//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/reduction_to_band.h"

#include <cmath>

#include <gtest/gtest.h>
#include <lapack/util.hh>
#include <pika/future.hpp>
#include <pika/modules/threadmanager.hpp>
#include <pika/runtime.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct ReductionToBandTest : public TestWithCommGrids {};

template <class T>
using ReductionToBandTestMC = ReductionToBandTest<T>;

TYPED_TEST_SUITE(ReductionToBandTestMC, MatrixElementTypes);

struct config_t {
  LocalElementSize size;
  TileElementSize block_size;
  SizeType band_size = block_size.rows();
};

std::vector<config_t> configs{
    {{0, 0}, {3, 3}},
    // full-tile band
    {{3, 3}, {3, 3}},    // single tile (nothing to do)
    {{12, 12}, {3, 3}},  // tile always full size (less room for distribution over ranks)
    {{13, 13}, {3, 3}},  // tile incomplete
    {{24, 24}, {3, 3}},  // tile always full size (more room for distribution)
    {{40, 40}, {5, 5}},
};

std::vector<config_t> configs_subband{
    {{0, 0}, {6, 6}, 2},
    // sub-tile band
    {{4, 4}, {4, 4}, 2},    // single tile
    {{12, 12}, {4, 4}, 2},  // tile always full size (less room for distribution over ranks)
    {{13, 13}, {3, 3}, 3},  // tile incomplete
    {{24, 24}, {3, 3}, 3},  // tile always full size (more room for distribution)
    {{40, 40}, {5, 5}, 5},
};

template <class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
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
  DLAF_ASSERT(square_size(matrix), matrix.blockSize());

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

    if (band_size < tile_l.size().rows())
      lapack::laset(blas::Uplo::General, tile_l.size().rows(), band_size, T(0), T(0), tile_l.ptr({0, 0}),
                    tile_l.ld());

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
void splitReflectorsAndBand(MatrixLocal<T>& mat_v, MatrixLocal<T>& mat_b, const SizeType band_size) {
  DLAF_ASSERT_HEAVY(square_size(mat_v), mat_v.size());
  DLAF_ASSERT_HEAVY(square_size(mat_b), mat_b.size());

  DLAF_ASSERT(equal_size(mat_v, mat_b), mat_v.size(), mat_b.size());

  for (SizeType diag = 0; diag <= 1; ++diag) {
    for (SizeType i = diag; i < mat_v.nrTiles().rows(); ++i) {
      const GlobalTileIndex idx(i, i - diag);
      matrix::internal::copy(mat_v.tile(idx), mat_b.tile(idx));
    }
  }

  setupHermitianBand(mat_b, band_size);
}

template <class T>
auto allGatherTaus(const SizeType k, const SizeType band_size,
                   std::vector<pika::shared_future<common::internal::vector<T>>> fut_local_taus) {
  std::vector<T> taus;
  taus.reserve(to_sizet(k));

  pika::wait_all(fut_local_taus);
  auto local_taus = pika::unwrap(fut_local_taus);

  const auto n_chunks = std::ceil(float(k) / band_size);

  for (auto index_chunk = 0; index_chunk < n_chunks; ++index_chunk) {
    std::vector<T> chunk_data;
    const auto index_chunk_local = to_sizet(index_chunk);
    chunk_data = local_taus.at(index_chunk_local);

    // copy each chunk contiguously
    std::copy(chunk_data.begin(), chunk_data.end(), std::back_inserter(taus));
  }

  return taus;
}

template <class T>
auto allGatherTaus(const SizeType k, const SizeType chunk_size, const SizeType band_size,
                   std::vector<pika::shared_future<common::internal::vector<T>>> fut_local_taus,
                   comm::CommunicatorGrid comm_grid) {
  std::vector<T> taus;
  taus.reserve(to_sizet(k));

  pika::wait_all(fut_local_taus);
  auto local_taus = pika::unwrap(fut_local_taus);

  DLAF_ASSERT(band_size == chunk_size, band_size, chunk_size);

  const auto n_chunks = std::ceil(float(k) / chunk_size);

  for (auto index_chunk = 0; index_chunk < n_chunks; ++index_chunk) {
    const auto owner = index_chunk % comm_grid.size().cols();
    const bool is_owner = owner == comm_grid.rank().col();

    const auto this_chunk_size = std::min(k - index_chunk * chunk_size, chunk_size);

    std::vector<T> chunk_data;
    if (is_owner) {
      const auto index_chunk_local = to_sizet(index_chunk / comm_grid.size().cols());
      chunk_data = local_taus.at(index_chunk_local);
      sync::broadcast::send(comm_grid.rowCommunicator(),
                            common::make_data(chunk_data.data(),
                                              static_cast<SizeType>(chunk_data.size())));
    }
    else {
      chunk_data.resize(to_sizet(this_chunk_size));
      sync::broadcast::receive_from(owner, comm_grid.rowCommunicator(),
                                    common::make_data(chunk_data.data(),
                                                      static_cast<SizeType>(chunk_data.size())));
    }

    // copy each chunk contiguously
    std::copy(chunk_data.begin(), chunk_data.end(), std::back_inserter(taus));
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
      return matrix_a.read(ij_tile).get()(ij_element_wrt_tile);
    else
      return reference.read(ij_tile).get()(ij_element_wrt_tile);
  };
  CHECK_MATRIX_NEAR(merged_matrices, matrix_a, 0, TypeUtilities<T>::error);
}

template <class T>
auto checkResult(const SizeType k, const SizeType band_size, Matrix<const T, Device::CPU>& reference,
                 MatrixLocal<T> const& mat_v, MatrixLocal<T> const& mat_b, std::vector<T> const& taus) {
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
                              const SizeType band_size) {
  const SizeType k_reflectors = std::max(SizeType(0), size.rows() - band_size - 1);
  DLAF_ASSERT(block_size.rows() % band_size == 0, block_size.rows(), band_size);

  Distribution distribution({size.rows(), size.cols()}, block_size);

  // setup the reference input matrix
  Matrix<const T, Device::CPU> reference = [size = size, block_size = block_size]() {
    Matrix<T, Device::CPU> reference(size, block_size);
    matrix::util::set_random_hermitian(reference);
    return reference;
  }();

  Matrix<T, Device::CPU> mat_a_h(distribution);
  copy(reference, mat_a_h);

  common::internal::vector<pika::shared_future<common::internal::vector<T>>> local_taus;
  {
    MatrixMirror<T, D, Device::CPU> mat_a(mat_a_h);
    local_taus = eigensolver::reductionToBand<B, D, T>(mat_a.get(), band_size);
  }

  checkUpperPartUnchanged(reference, mat_a_h);

  auto mat_v = allGather(blas::Uplo::Lower, mat_a_h);
  auto mat_b = makeLocal(mat_a_h);
  splitReflectorsAndBand(mat_v, mat_b, band_size);

  auto taus = allGatherTaus(k_reflectors, band_size, local_taus);
  EXPECT_EQ(taus.size(), k_reflectors);

  checkResult(k_reflectors, band_size, reference, mat_v, mat_b, taus);
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessLocal) {
  for (const auto& config : configs) {
    const auto& [size, block_size, band_size] = config;

    testReductionToBandLocal<TypeParam, Backend::MC, Device::CPU>(size, block_size, band_size);
  }
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessLocalSubBand) {
  for (const auto& config : configs_subband) {
    const auto& [size, block_size, band_size] = config;

    testReductionToBandLocal<TypeParam, Backend::MC, Device::CPU>(size, block_size, band_size);
  }
}

TYPED_TEST(ReductionToBandTestMC, CorrectnessDistributed) {
  constexpr Device device = Device::CPU;

  for (auto&& comm_grid : this->commGrids()) {
    for (const auto& config : configs) {
      const auto& [size, block_size, band_size] = config;

      const SizeType k_reflectors = std::max(SizeType(0), size.rows() - band_size - 1);
      DLAF_ASSERT(block_size.rows() % band_size == 0, block_size.rows(), band_size);

      Distribution distribution({size.rows(), size.cols()}, block_size, comm_grid.size(),
                                comm_grid.rank(), {0, 0});

      // setup the reference input matrix
      Matrix<const TypeParam, device> reference = [&]() {
        Matrix<TypeParam, device> reference(distribution);
        matrix::util::set_random_hermitian(reference);
        return reference;
      }();

      Matrix<TypeParam, device> matrix_a(distribution);
      copy(reference, matrix_a);

      auto local_taus = eigensolver::reductionToBand<Backend::MC>(comm_grid, matrix_a);
      pika::threads::get_thread_manager().wait();

      checkUpperPartUnchanged(reference, matrix_a);

      auto mat_v = allGather(blas::Uplo::Lower, matrix_a, comm_grid);
      auto mat_b = makeLocal(matrix_a);
      splitReflectorsAndBand(mat_v, mat_b, band_size);

      auto taus = allGatherTaus(k_reflectors, block_size.cols(), band_size, local_taus, comm_grid);
      DLAF_ASSERT(to_SizeType(taus.size()) == k_reflectors, taus.size(), k_reflectors);

      checkResult(k_reflectors, band_size, reference, mat_v, mat_b, taus);
    }
  }
}
