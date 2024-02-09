//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cmath>

#include <pika/execution.hpp>
#include <pika/init.hpp>

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
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/eigensolver/reduction_utils.h>
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
    {{12, 12}, {3, 3}},  // tile always full size (less room for distribution over ranks)
    {{13, 13}, {3, 3}},  // tile incomplete
    {{24, 24}, {3, 3}},  // tile always full size (more room for distribution)
    {{40, 40}, {5, 5}},
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
