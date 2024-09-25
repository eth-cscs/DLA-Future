//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/functions_sync.h>
#include <dlaf/communication/sync/broadcast.h>
#include <dlaf/eigensolver/reduction_to_band.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::comm;
using namespace dlaf::memory;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;

using pika::execution::experimental::any_sender;
using pika::execution::experimental::when_all_vector;
using pika::this_thread::experimental::sync_wait;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct ReductionToBandTest : public TestWithCommGrids {};

template <class T>
using ReductionToBandTestMC = ReductionToBandTest<T>;

TYPED_TEST_SUITE(ReductionToBandTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using ReductionToBandTestGPU = ReductionToBandTest<T>;

TYPED_TEST_SUITE(ReductionToBandTestGPU, MatrixElementTypes);
#endif

struct config_t {
  LocalElementSize size;
  TileElementSize block_size;
  SizeType band_size = block_size.rows();
};

// Structure of the input matrix
// Banded input matrices will have a band smaller than the target band_size
enum class InputMatrixStructure { full, banded };

std::vector<config_t> configs{
    {{0, 0}, {3, 3}},
    // full-tile band
    {{3, 3}, {3, 3}},    // single tile (nothing to do)
    {{6, 6}, {3, 3}},    // tile always full size (less room for distribution over ranks)
    {{9, 9}, {3, 3}},    // tile always full size (less room for distribution over ranks)
    {{12, 12}, {3, 3}},  // tile always full size (less room for distribution over ranks)
    {{24, 24}, {3, 3}},  // tile always full size (more room for distribution)
    {{40, 40}, {5, 5}},
    // tile incomplete
    {{8, 8}, {3, 3}},
    {{13, 13}, {3, 3}},
};

std::vector<config_t> configs_subband{
    {{0, 0}, {6, 6}, 2},  // empty matrix

    // half-tile band
    {{4, 4}, {4, 4}, 2},  // single tile
    {{12, 12}, {4, 4}, 2},
    {{42, 42}, {6, 6}, 3},
    {{13, 13}, {6, 6}, 3},  // tile incomplete

    // multi-band
    {{27, 27}, {9, 9}, 3},
    {{42, 42}, {12, 12}, 4},
    {{29, 29}, {9, 9}, 3},  // tile incomplete
};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().baseTileSize()};
}

template <class T>
void copyConjTrans(const Tile<const T, Device::CPU>& from, const Tile<T, Device::CPU>& to) {
  DLAF_ASSERT(from.size() == transposed(to.size()), from.size(), to.size());

  for (const TileElementIndex& index : common::iterate_range2d(from.size()))
    to(transposed(index)) = dlaf::conj(from(index));
}

template <class T>
void mirrorLowerOnDiag(const Tile<T, Device::CPU>& tile) {
  DLAF_ASSERT(square_size(tile), tile.size());

  copyConjTrans(tile, tile);
}

template <class T>
void setupHermitianBand(MatrixLocal<T>& matrix, const SizeType band_size) {
  DLAF_ASSERT(matrix.blockSize().rows() % band_size == 0, band_size, matrix.blockSize().rows());

  DLAF_ASSERT(square_blocksize(matrix), matrix.blockSize());
  DLAF_ASSERT(square_size(matrix), matrix.size());

  dlaf::common::internal::SingleThreadedBlasScope single;

  // 0-diagonal: mirror band
  // note: diagonal subtiles are correctly set just in the lower part by the algorithm
  for (SizeType k = 0; k < matrix.nrTiles().cols(); ++k) {
    const GlobalTileIndex kk(k, k);

    const auto& tile = matrix.tile(kk);

    const SizeType n = std::max<SizeType>(0, tile.size().rows() - band_size - 1);
    if (band_size < tile.size().rows())
      lapack::laset(blas::Uplo::Lower, n, n, T{0}, T{0}, tile.ptr({band_size + 1, 0}), tile.ld());

    mirrorLowerOnDiag(tile);
  }

  // 1-diagonal: setup band "edge" (and its tranposed)
  for (SizeType j = 0; j < matrix.nrTiles().cols() - 1; ++j) {
    const SizeType i = j + 1;
    const GlobalTileIndex ij(i, j);

    const auto& tile_l = matrix.tile(ij);

    if (tile_l.size().rows() > 1)
      lapack::laset(blas::Uplo::Lower, tile_l.size().rows() - 1, band_size, T(0), T(0),
                    tile_l.ptr({1, tile_l.size().cols() - band_size}), tile_l.ld());

    if (band_size < tile_l.size().cols())
      lapack::laset(blas::Uplo::General, tile_l.size().rows(), tile_l.size().cols() - band_size, T(0),
                    T(0), tile_l.ptr({0, 0}), tile_l.ld());

    copyConjTrans(matrix.tile(ij), matrix.tile(common::transposed(ij)));
  }

  // k-diagonal (with k >= 2): zero out both lower and upper out-of-band subtiles
  for (SizeType j = 0; j < matrix.nrTiles().cols() - 2; ++j) {
    for (SizeType i = j + 2; i < matrix.nrTiles().rows(); ++i) {
      const GlobalTileIndex ij(i, j);

      tile::internal::set0(matrix.tile(ij));
      tile::internal::set0(matrix.tile(common::transposed(ij)));
    }
  }
}

template <class T>
void splitReflectorsAndBand(MatrixLocal<const T>& mat_v, MatrixLocal<T>& mat_b,
                            const SizeType band_size) {
  DLAF_ASSERT_HEAVY(square_size(mat_v), mat_v.size());
  DLAF_ASSERT_HEAVY(square_size(mat_b), mat_b.size());

  DLAF_ASSERT(equal_size(mat_v, mat_b), mat_v.size(), mat_b.size());

  for (SizeType diag = 0; diag <= 1; ++diag) {
    for (SizeType i = diag; i < mat_v.nrTiles().rows(); ++i) {
      const GlobalTileIndex idx(i, i - diag);
      matrix::internal::copy(mat_v.tile_read(idx), mat_b.tile(idx));
    }
  }

  setupHermitianBand(mat_b, band_size);
}

template <class T>
auto allGatherTaus(const SizeType k, Matrix<T, Device::CPU>& mat_local_taus) {
  auto local_taus_tiles = sync_wait(when_all_vector(selectRead(
      mat_local_taus, common::iterate_range2d(LocalTileSize(mat_local_taus.nrTiles().rows(), 1)))));

  std::vector<T> taus;
  taus.reserve(to_sizet(k));
  for (const auto& t : local_taus_tiles) {
    std::copy(t.get().ptr(), t.get().ptr() + t.get().size().rows(), std::back_inserter(taus));
  }

  DLAF_ASSERT(to_SizeType(taus.size()) == k, taus.size(), k);

  return taus;
}

template <class T>
auto allGatherTaus(const SizeType k, Matrix<T, Device::CPU>& mat_taus,
                   comm::CommunicatorGrid& comm_grid) {
  const auto local_num_tiles = mat_taus.distribution().localNrTiles().rows();
  const auto num_tiles = mat_taus.distribution().nrTiles().rows();
  const auto local_num_tiles_expected =
      num_tiles / comm_grid.size().cols() +
      (comm_grid.rank().col() < (num_tiles % comm_grid.size().cols()) ? 1 : 0);
  EXPECT_EQ(local_num_tiles, local_num_tiles_expected);

  std::vector<T> taus;
  taus.reserve(to_sizet(k));

  for (SizeType i = 0; i < mat_taus.nrTiles().rows(); ++i) {
    const auto owner = mat_taus.rankGlobalTile(GlobalTileIndex(i, 0)).row();
    const bool is_owner = owner == comm_grid.rank().col();

    const auto chunk_size = mat_taus.tileSize(GlobalTileIndex(i, 0)).rows();

    if (is_owner) {
      auto tile_local = sync_wait(mat_taus.read(GlobalTileIndex(i, 0)));
      sync::broadcast::send(comm_grid.rowCommunicator(), common::make_data(tile_local.get()));
      std::copy(tile_local.get().ptr(), tile_local.get().ptr() + tile_local.get().size().rows(),
                std::back_inserter(taus));
    }
    else {
      Tile<T, Device::CPU> tile_local(TileElementSize(chunk_size, 1),
                                      MemoryView<T, Device::CPU>(chunk_size), chunk_size);
      sync::broadcast::receive_from(owner, comm_grid.rowCommunicator(), common::make_data(tile_local));
      std::copy(tile_local.ptr(), tile_local.ptr() + tile_local.size().rows(), std::back_inserter(taus));
    }
  }

  return taus;
}

// Verify equality of all the elements in the upper part of the matrices
template <class T>
auto checkUpperPartUnchanged(Matrix<const T, Device::CPU>& reference,
                             Matrix<const T, Device::CPU>& matrix_a) {
  auto merged_matrices = [&reference, &matrix_a](const GlobalElementIndex& index) {
    const auto& dist = reference.distribution();
    const auto ij_tile = dist.globalTileIndex(index);
    const auto ij_element_wrt_tile = dist.tileElementIndex(index);

    const bool is_in_upper = index.row() < index.col();

    if (!is_in_upper)
      return sync_wait(matrix_a.read(ij_tile)).get()(ij_element_wrt_tile);
    else
      return sync_wait(reference.read(ij_tile)).get()(ij_element_wrt_tile);
  };
  CHECK_MATRIX_NEAR(merged_matrices, matrix_a, 0, TypeUtilities<T>::error);
}

template <class T>
auto checkResult(const SizeType k, const SizeType band_size, Matrix<const T, Device::CPU>& reference,
                 const MatrixLocal<T>& mat_v, const MatrixLocal<T>& mat_b, const std::vector<T>& taus) {
  const GlobalElementIndex offset(band_size, 0);
  // Now that all input are collected locally, it's time to apply the transformation,
  // ...but just if there is any
  if (offset.isIn(mat_v.size())) {
    // Reduction to band returns a sequence of transformations applied from left and right to A
    // allowing to reduce the matrix A to a band matrix B
    //
    // Hn* ... H2* H1* A H1 H2 ... Hn
    // Q* A Q = B
    //
    // Applying the inverse of the same transformations, we can go from B to A
    // Q B Q* = A
    // Q = H1 H2 ... Hn
    // H1 H2 ... Hn B Hn* ... H2* H1*

    dlaf::common::internal::SingleThreadedBlasScope single;

    // apply from left...
    const GlobalElementIndex left_offset = offset;
    const GlobalElementSize left_size{mat_b.size().rows() - band_size, mat_b.size().cols()};
    lapack::unmqr(lapack::Side::Left, lapack::Op::NoTrans, left_size.rows(), left_size.cols(), k,
                  mat_v.ptr(offset), mat_v.ld(), taus.data(), mat_b.ptr(left_offset), mat_b.ld());

    // ... and from right
    const GlobalElementIndex right_offset = common::transposed(left_offset);
    const GlobalElementSize right_size = common::transposed(left_size);

    lapack::unmqr(lapack::Side::Right, lapack::Op::ConjTrans, right_size.rows(), right_size.cols(), k,
                  mat_v.ptr(offset), mat_v.ld(), taus.data(), mat_b.ptr(right_offset), mat_b.ld());
  }

  // Eventually, check the result obtained by applying the inverse transformation equals the original matrix
  auto result = [&dist = reference.distribution(),
                 &mat_local = mat_b](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };
  CHECK_MATRIX_NEAR(result, reference, 0,
                    std::max<SizeType>(1, mat_b.size().linear_size()) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testReductionToBandLocal(const LocalElementSize size, const TileElementSize block_size,
                              const SizeType band_size,
                              const InputMatrixStructure input_matrix_structure) {
  const SizeType k_reflectors = std::max(SizeType(0), size.rows() - band_size - 1);
  DLAF_ASSERT(block_size.rows() % band_size == 0, block_size.rows(), band_size);

  Distribution distribution({size.rows(), size.cols()}, block_size);

  // setup the reference input matrix
  Matrix<const T, Device::CPU> reference = [size = size, block_size = block_size, band_size,
                                            input_matrix_structure]() {
    Matrix<T, Device::CPU> reference(size, block_size);
    if (input_matrix_structure == InputMatrixStructure::banded)
      // Matrix already in band form, with band smaller than band_size
      matrix::util::set_random_hermitian_banded(reference, band_size - 1);
    else
      matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(distribution);
  copy(reference, mat_a_h);

  Matrix<T, Device::CPU> mat_local_taus = [&]() mutable {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    return eigensolver::internal::reduction_to_band<B, D, T>(mat_a.get(), band_size);
  }();

  ASSERT_EQ(mat_local_taus.blockSize().rows(), block_size.rows());

  checkUpperPartUnchanged(reference, mat_a_h);

  auto mat_v = allGather(blas::Uplo::Lower, mat_a_h);
  auto mat_b = makeLocal(mat_a_h);
  splitReflectorsAndBand(mat_v, mat_b, band_size);

  auto taus = allGatherTaus(k_reflectors, mat_local_taus);
  ASSERT_EQ(taus.size(), k_reflectors);

  checkResult(k_reflectors, band_size, reference, mat_v, mat_b, taus);
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessLocal) {
  for (const auto& config : configs) {
    const auto& [size, block_size, band_size] = config;

    for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded})
      testReductionToBandLocal<TypeParam, Backend::MC, Device::CPU>(size, block_size, band_size,
                                                                    input_matrix_structure);
  }
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessLocalSubBand) {
  for (const auto& config : configs_subband) {
    const auto& [size, block_size, band_size] = config;

    for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded}) {
      testReductionToBandLocal<TypeParam, Backend::MC, Device::CPU>(size, block_size, band_size,
                                                                    input_matrix_structure);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(ReductionToBandTestGPU, CorrectnessLocal) {
  for (const auto& config : configs) {
    const auto& [size, block_size, band_size] = config;

    for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded}) {
      testReductionToBandLocal<TypeParam, Backend::GPU, Device::GPU>(size, block_size, band_size,
                                                                     input_matrix_structure);
    }
  }
}

TYPED_TEST(ReductionToBandTestGPU, CorrectnessLocalSubBand) {
  for (const auto& config : configs_subband) {
    const auto& [size, block_size, band_size] = config;

    for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded}) {
      testReductionToBandLocal<TypeParam, Backend::GPU, Device::GPU>(size, block_size, band_size,
                                                                     input_matrix_structure);
    }
  }
}
#endif

template <class T, Device D, Backend B>
void testReductionToBand(comm::CommunicatorGrid& grid, const LocalElementSize size,
                         const TileElementSize block_size, const SizeType band_size,
                         const InputMatrixStructure input_matrix_structure) {
  const SizeType k_reflectors = std::max(SizeType(0), size.rows() - band_size - 1);
  DLAF_ASSERT(block_size.rows() % band_size == 0, block_size.rows(), band_size);

  Distribution distribution({size.rows(), size.cols()}, block_size, grid.size(), grid.rank(), {0, 0});

  // setup the reference input matrix
  Matrix<const T, Device::CPU> reference = [&]() {
    Matrix<T, Device::CPU> reference(distribution);
    if (input_matrix_structure == InputMatrixStructure::banded)
      // Matrix already in band form, with band smaller than band_size
      matrix::util::set_random_hermitian_banded(reference, band_size - 1);
    else
      matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> matrix_a_h(distribution);
  copy(reference, matrix_a_h);

  Matrix<T, Device::CPU> mat_local_taus = [&]() {
    MatrixMirror<T, D, Device::CPU> matrix_a(matrix_a_h);
    return eigensolver::internal::reduction_to_band<B>(grid, matrix_a.get(), band_size);
  }();

  ASSERT_EQ(mat_local_taus.blockSize().rows(), block_size.rows());

  checkUpperPartUnchanged(reference, matrix_a_h);

  // Wait for all work to finish before doing blocking communication
  pika::wait();

  auto mat_v = allGather(blas::Uplo::Lower, matrix_a_h, grid);
  auto mat_b = makeLocal(matrix_a_h);
  splitReflectorsAndBand(mat_v, mat_b, band_size);

  auto taus = allGatherTaus(k_reflectors, mat_local_taus, grid);
  ASSERT_EQ(taus.size(), k_reflectors);

  checkResult(k_reflectors, band_size, reference, mat_v, mat_b, taus);
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessDistributed) {
  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& [size, block_size, band_size] : configs) {
      for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded}) {
        testReductionToBand<TypeParam, Device::CPU, Backend::MC>(comm_grid, size, block_size, band_size,
                                                                 input_matrix_structure);
      }
    }
  }
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessDistributedSubBand) {
  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& [size, block_size, band_size] : configs_subband) {
      for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded}) {
        testReductionToBand<TypeParam, Device::CPU, Backend::MC>(comm_grid, size, block_size, band_size,
                                                                 input_matrix_structure);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(ReductionToBandTestGPU, CorrectnessDistributed) {
  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& [size, block_size, band_size] : configs) {
      for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded}) {
        testReductionToBand<TypeParam, Device::GPU, Backend::GPU>(comm_grid, size, block_size, band_size,
                                                                  input_matrix_structure);
      }
    }
  }
}

TYPED_TEST(ReductionToBandTestGPU, CorrectnessDistributedSubBand) {
  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& [size, block_size, band_size] : configs_subband) {
      for (auto input_matrix_structure : {InputMatrixStructure::full, InputMatrixStructure::banded}) {
        testReductionToBand<TypeParam, Device::GPU, Backend::GPU>(comm_grid, size, block_size, band_size,
                                                                  input_matrix_structure);
      }
    }
  }
}
#endif

template <class T>
struct CAReductionToBandTest : public TestWithCommGrids {};

template <class T>
using CAReductionToBandTestMC = ReductionToBandTest<T>;

TYPED_TEST_SUITE(CAReductionToBandTestMC, MatrixElementTypes);

template <class T>
MatrixLocal<T> allGatherT(Matrix<const T, Device::CPU>& source, comm::CommunicatorGrid& comm_grid) {
  // TODO tranposed distribution
  // DLAF_ASSERT(matrix::equal_process_grid(source, comm_grid), source, comm_grid);

  namespace tt = pika::this_thread::experimental;

  MatrixLocal<std::remove_const_t<T>> dest(source.size(), source.baseTileSize());

  const auto& dist_source = source.distribution();
  const auto rank = transposed(dist_source.rank_index());

  for (const auto& ij : iterate_range2d(dist_source.nr_tiles())) {
    const comm::Index2D owner = transposed(dist_source.rank_global_tile(ij));

    auto& dest_tile = dest.tile(ij);

    if (owner == rank) {
      const auto source_tile_holder = tt::sync_wait(source.read(ij));
      const auto& source_tile = source_tile_holder.get();
      comm::sync::broadcast::send(comm_grid.fullCommunicator(), source_tile);
      matrix::internal::copy(source_tile, dest_tile);
    }
    else {
      comm::sync::broadcast::receive_from(comm_grid.rankFullCommunicator(owner),
                                          comm_grid.fullCommunicator(), dest_tile);
    }
  }

  return MatrixLocal<T>(std::move(dest));
}

template <class T>
auto checkResult(const Distribution dist, const SizeType band_size,
                 Matrix<const T, Device::CPU>& reference, const MatrixLocal<T>& mat_b,
                 const MatrixLocal<T>& mat_hh_1st, const MatrixLocal<T>& taus_1st,
                 const MatrixLocal<T>& mat_hh_2nd, const std::vector<T>& taus_2nd) {
  const GlobalElementIndex offset(band_size, 0);
  // Now that all input are collected locally, it's time to apply the transformation,
  // ...but just if there is any
  if (offset.isIn(mat_hh_1st.size())) {
    dlaf::common::internal::SingleThreadedBlasScope single;

    const SizeType ntiles = mat_b.nrTiles().cols() - 1;

    // Apply in reverse order (blocked algorithm), which means both from last to first, inverting
    // intra-step too, i.e. 2nd first and 1st last.
    for (SizeType j = ntiles - 1; j >= 0; --j) {
      const SizeType i = j + 1;
      const SizeType i_el =
          dist.template global_element_from_global_tile_and_tile_element<Coord::Row>(i, 0);

      const std::size_t nranks_with_data =
          to_sizet(std::min<SizeType>(mat_b.nrTiles().rows() - i, dist.grid_size().rows()));

      // === 2nd pass
      // prepare workspace (height = max(nranks)) + reorder heads
      std::vector<comm::IndexT_MPI> col_rank_order(nranks_with_data, -1);
      const comm::IndexT_MPI first_rank = dist.template rank_global_tile<Coord::Row>(i);
      std::iota(col_rank_order.begin(), col_rank_order.end(), first_rank);
      std::transform(col_rank_order.begin(), col_rank_order.end(), col_rank_order.begin(),
                     [size = dist.grid_size().rows()](const comm::IndexT_MPI& value) {
                       return std::modulus<comm::IndexT_MPI>{}(value, size);
                     });

      // HH2
      const matrix::Distribution dist_hh_2nd = [&]() {
        using matrix::internal::distribution::global_tile_element_distance;
        const SizeType i_begin = i;
        const SizeType i_end = std::min<SizeType>(i + dist.grid_size().rows(), dist.nr_tiles().rows());
        const SizeType nrows = global_tile_element_distance<Coord::Row>(dist, i_begin, i_end);
        return matrix::Distribution({nrows, mat_b.blockSize().cols()}, mat_b.blockSize());
      }();
      MatrixLocal<T> hh_2nd(dist_hh_2nd.size(), dist_hh_2nd.block_size());

      const SizeType nrefls = [&]() {
        const SizeType reflector_size = hh_2nd.size().rows();
        return std::min(hh_2nd.size().cols(), reflector_size - 1);
      }();

      if (nrefls > 0) {
        for (SizeType i = 0; i < to_SizeType(col_rank_order.size()); ++i) {
          const SizeType ii = to_SizeType(col_rank_order[to_sizet(i)]);

          const bool is_last = i == (to_SizeType(col_rank_order.size()) - 1);
          const SizeType last_rows = hh_2nd.size().rows() % dist.block_size().rows();
          if (!is_last || last_rows == 0) {
            matrix::internal::copy(mat_hh_2nd.tile({ii, j}), hh_2nd.tile({i, 0}));
          }
          else {
            const auto& tile = mat_hh_2nd.tile({ii, j}).subTileReference(
                {{0, 0}, {last_rows, dist.block_size().cols()}});
            matrix::internal::copy(tile, hh_2nd.tile({i, 0}));
          }
        }

        // T2
        const SizeType j_el =
            dist.template global_element_from_global_tile_and_tile_element<Coord::Col>(j, 0);

        MatrixLocal<T> T_2nd({nrefls, nrefls}, mat_b.blockSize());
        lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, hh_2nd.size().rows(),
                      nrefls, hh_2nd.ptr(), hh_2nd.ld(), taus_2nd.data() + j_el, T_2nd.ptr(),
                      T_2nd.ld());

        // Apply HH2 from L and R
        lapack::larfb(lapack::Side::Left, lapack::Op::NoTrans, lapack::Direction::Forward,
                      lapack::StoreV::Columnwise, hh_2nd.size().rows(), mat_b.size().cols() - j_el,
                      nrefls, hh_2nd.ptr(), hh_2nd.ld(), T_2nd.ptr(), T_2nd.ld(),
                      mat_b.ptr({i_el, j_el}), mat_b.ld());
        lapack::larfb(lapack::Side::Right, lapack::Op::ConjTrans, lapack::Direction::Forward,
                      lapack::StoreV::Columnwise, mat_b.size().rows() - j_el, hh_2nd.size().rows(),
                      nrefls, hh_2nd.ptr(), hh_2nd.ld(), T_2nd.ptr(), T_2nd.ld(),
                      mat_b.ptr({j_el, i_el}), mat_b.ld());
      }

      // === 1st pass
      // HH1 (for all ranks)
      // prepare workspace (height = local matrix for each rank) with zeros to fill voids
      // Note: HH_1st workspaces is stored as a matrix where each column of tiles is for a specific rank.
      const matrix::Distribution dist_hh_1st({dist.size().rows() - i_el,
                                              to_SizeType(nranks_with_data) * dist.tile_size().cols()},
                                             dist.tile_size());
      MatrixLocal<T> hh_1st(dist_hh_1st.size(), dist_hh_1st.tile_size());

      std::size_t col_rank_current = 0;
      for (SizeType i_a = i; i_a < dist.nr_tiles().rows(); ++i_a, ++col_rank_current) {
        col_rank_current %= col_rank_order.size();

        const SizeType i_hh = i_a - i;

        for (SizeType j_hh = 0; j_hh < hh_1st.nrTiles().cols(); ++j_hh) {
          const auto& tile_hh = hh_1st.tile({i_hh, j_hh});

          if (j_hh == to_SizeType(col_rank_current)) {
            dlaf::matrix::internal::copy(mat_hh_1st.tile({i_a, j}), tile_hh);
          }
          else {
            dlaf::tile::internal::set0(tile_hh);
          }
        }
      }

      // Note: well-formed heads
      for (SizeType j = 0; j < hh_1st.nrTiles().cols(); ++j) {
        const auto& tile_hh = hh_1st.tile({j, j});
        dlaf::tile::internal::laset(blas::Uplo::Upper, T(0), T(1), tile_hh);
      }

      // Note: apply one HH1 per time, independently, order not relevant
      for (SizeType col_rank = 0; col_rank < to_SizeType(col_rank_order.size()); ++col_rank) {
        const SizeType rank = to_SizeType(col_rank_order[to_sizet(col_rank)]);

        const SizeType i_begin = col_rank;
        const SizeType i_end_gap = dist_hh_1st.nr_tiles().rows();
        const SizeType i_end = i_begin + dlaf::util::ceilDiv(dist_hh_1st.nr_tiles().rows() - i_begin,
                                                             to_SizeType(col_rank_order.size()));
        using matrix::internal::distribution::global_tile_element_distance;

        const SizeType refl_size = global_tile_element_distance<Coord::Row>(dist_hh_1st, i_begin, i_end);

        const auto& hh_1st_head = hh_1st.tile({col_rank, col_rank});
        const SizeType nrefls = std::min(refl_size - 1, hh_1st_head.size().cols());

        if (nrefls <= 0)
          continue;

        // Compute T1
        const auto& tile_taus = taus_1st.tile({j, rank});

        const SizeType refl_size_gap =
            global_tile_element_distance<Coord::Row>(dist_hh_1st, i_begin, i_end_gap);

        MatrixLocal<T> T_1st({nrefls, nrefls}, mat_b.blockSize());
        lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, refl_size_gap, nrefls,
                      hh_1st_head.ptr(), hh_1st_head.ld(), tile_taus.ptr(), T_1st.ptr(), T_1st.ld());

        // Apply HH1 (of a rank) from L and R
        {
          const SizeType m = dist.size().rows() - (i + col_rank) * dist.tile_size().rows();
          const SizeType n = dist.size().cols() - j * dist.tile_size().cols();
          lapack::larfb(lapack::Side::Left, lapack::Op::NoTrans, lapack::Direction::Forward,
                        lapack::StoreV::Columnwise, m, n, nrefls, hh_1st_head.ptr(), hh_1st.ld(),
                        T_1st.ptr(), T_1st.ld(), mat_b.tile({i + col_rank, j}).ptr(), mat_b.ld());
        }
        {
          const SizeType m = dist.size().rows() - j * dist.tile_size().rows();
          const SizeType n = dist.size().cols() - (i + col_rank) * dist.tile_size().cols();
          lapack::larfb(lapack::Side::Right, lapack::Op::ConjTrans, lapack::Direction::Forward,
                        lapack::StoreV::Columnwise, m, n, nrefls, hh_1st_head.ptr(), hh_1st.ld(),
                        T_1st.ptr(), T_1st.ld(), mat_b.tile({j, i + col_rank}).ptr(), mat_b.ld());
        }
      }
    }
  }

  // Eventually, check the result obtained by applying the inverse transformation equals the original matrix
  auto result = [&dist = reference.distribution(),
                 &mat_local = mat_b](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, reference, 0,
                    std::max<SizeType>(1, mat_b.size().linear_size()) * TypeUtilities<T>::error);
}

template <class T, Device D, Backend B>
void testCAReductionToBand(comm::CommunicatorGrid& grid, const LocalElementSize size,
                           const TileElementSize block_size, const SizeType band_size,
                           const InputMatrixStructure input_matrix_structure) {
  const SizeType k_reflectors = std::max(SizeType(0), size.rows() - band_size - 1);
  DLAF_ASSERT(block_size.rows() % band_size == 0, block_size.rows(), band_size);

  const Distribution dist({size.rows(), size.cols()}, block_size, grid.size(), grid.rank(), {0, 0});

  // setup the reference input matrix
  Matrix<const T, Device::CPU> reference = [&]() {
    Matrix<T, Device::CPU> reference(dist);
    if (input_matrix_structure == InputMatrixStructure::banded)
      // Matrix already in band form, with band smaller than band_size
      matrix::util::set_random_hermitian_banded(reference, band_size - 1);
    else
      matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> matrix_a_h(dist);
  copy(reference, matrix_a_h);

  eigensolver::internal::CARed2BandResult<T, D> red2band_result = [&]() {
    MatrixMirror<T, D, Device::CPU> matrix_a(matrix_a_h);
    return eigensolver::internal::ca_reduction_to_band<B>(grid, matrix_a.get(), band_size);
  }();

  ASSERT_EQ(red2band_result.taus_1st.block_size().rows(), block_size.rows());
  ASSERT_EQ(red2band_result.taus_2nd.block_size().rows(), block_size.rows());

  checkUpperPartUnchanged(reference, matrix_a_h);

  // Wait for all work to finish before doing blocking communication
  pika::wait();

  auto mat_hh_1st = allGather(blas::Uplo::Lower, matrix_a_h, grid);

  auto taus_1st = allGatherT(red2band_result.taus_1st, grid);
  ASSERT_EQ(taus_1st.size().rows(), k_reflectors);
  ASSERT_EQ(taus_1st.size().cols(), grid.size().rows());

  auto mat_hh_2nd = allGather(blas::Uplo::General, red2band_result.hh_2nd, grid);

  auto taus_2nd = allGatherTaus(k_reflectors, red2band_result.taus_2nd, grid);
  ASSERT_EQ(taus_2nd.size(), k_reflectors);

  auto mat_band = makeLocal(matrix_a_h);
  splitReflectorsAndBand(mat_hh_1st, mat_band, band_size);

  checkResult(dist, band_size, reference, mat_band, mat_hh_1st, taus_1st, mat_hh_2nd, taus_2nd);
}

TYPED_TEST(CAReductionToBandTestMC, CorrectnessDistributed) {
  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& [size, block_size, band_size] : configs) {
      for (auto input_matrix_structure : {InputMatrixStructure::full}) {
        testCAReductionToBand<TypeParam, Device::CPU, Backend::MC>(comm_grid, size, block_size,
                                                                   band_size, input_matrix_structure);
      }
    }
  }
}
