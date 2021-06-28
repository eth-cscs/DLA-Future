//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/factorization/cholesky.h"

#include "gtest/gtest.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class CholeskyTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(CholeskyTest, MatrixElementTypes);

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<LocalElementSize> square_sizes({{10, 10}, {25, 25}, {12, 12}, {0, 0}});
const std::vector<TileElementSize> square_block_sizes({{3, 3}, {5, 5}});

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T, Backend B, Device D>
void testCholesky(blas::Uplo uplo, LocalElementSize size, TileElementSize block_size) {
  std::function<T(const GlobalElementIndex&)> el, res;
  std::tie(el, res) = getCholeskySetters<GlobalElementIndex, T>(uplo);

  Matrix<T, Device::CPU> mat_h(size, block_size);
  set(mat_h, el);

  {
    MatrixMirror<T, D, Device::CPU> mat(mat_h);
    factorization::cholesky<B, D, T>(uplo, mat.get());
  }

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testCholesky(comm::CommunicatorGrid comm_grid, blas::Uplo uplo, LocalElementSize size,
                  TileElementSize block_size) {
  std::function<T(const GlobalElementIndex&)> el, res;
  std::tie(el, res) = getCholeskySetters<GlobalElementIndex, T>(uplo);

  // Matrix to undergo Cholesky decomposition
  Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                         std::min(1, comm_grid.size().cols() - 1));
  GlobalElementSize sz = globalTestSize(size);
  Distribution distribution(sz, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_h(std::move(distribution));
  set(mat_h, el);

  {
    MatrixMirror<T, D, Device::CPU> mat(mat_h);
    factorization::cholesky<B, D, T>(comm_grid, uplo, mat.get());
  }

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(CholeskyTest, CorrectnessLocal) {
  for (const auto& size : square_sizes) {
    for (const auto& block_size : square_block_sizes) {
      for (const auto& uplo : blas_uplos) {
        testCholesky<TypeParam, Backend::MC, Device::CPU>(uplo, size, block_size);
#ifdef DLAF_WITH_CUDA
        testCholesky<TypeParam, Backend::GPU, Device::GPU>(uplo, size, block_size);
#endif
      }
    }
  }
}

TYPED_TEST(CholeskyTest, CorrectnessDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : square_sizes) {
      for (const auto& block_size : square_block_sizes) {
        for (const auto& uplo : blas_uplos) {
          testCholesky<TypeParam, Backend::MC, Device::CPU>(comm_grid, uplo, size, block_size);
#ifdef DLAF_WITH_CUDA
          testCholesky<TypeParam, Backend::GPU, Device::GPU>(comm_grid, uplo, size, block_size);
#endif
        }
      }
    }
  }
}
