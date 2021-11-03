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

#include <functional>
#include <tuple>

#include <gtest/gtest.h>
#include <hpx/include/threadmanager.hpp>
#include <hpx/runtime.hpp>

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
class CholeskyTestMC : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};
TYPED_TEST_SUITE(CholeskyTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <typename Type>
class CholeskyTestGPU : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};
TYPED_TEST_SUITE(CholeskyTestGPU, MatrixElementTypes);
#endif

TYPED_TEST_SUITE(CholeskyTest, MatrixElementTypes);

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {5, 8}, {34, 34},                    // m <= mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m > mb
};

template <class T, Backend B, Device D>
void testCholesky(const blas::Uplo uplo, const SizeType m, const SizeType mb) {
  std::function<T(const GlobalElementIndex&)> el, res;

  const LocalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_h(size, block_size);

  std::tie(el, res) = getCholeskySetters<GlobalElementIndex, T>(uplo);

  set(mat_h, el);

  {
    MatrixMirror<T, D, Device::CPU> mat(mat_h);
    factorization::cholesky<B, D, T>(uplo, mat.get());
  }

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);
}

template <class T, Backend B, Device D>
void testCholesky(comm::CommunicatorGrid grid, const blas::Uplo uplo, const SizeType m,
                  const SizeType mb) {
  std::function<T(const GlobalElementIndex&)> el, res;

  const GlobalElementSize size(m, m);
  const TileElementSize block_size(mb, mb);
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution distribution(size, block_size, grid.size(), grid.rank(), src_rank_index);
  Matrix<T, Device::CPU> mat_h(std::move(distribution));

  std::tie(el, res) = getCholeskySetters<GlobalElementIndex, T>(uplo);

  set(mat_h, el);

  {
    MatrixMirror<T, D, Device::CPU> mat(mat_h);
    factorization::cholesky<B, D, T>(grid, uplo, mat.get());
  }

  CHECK_MATRIX_NEAR(res, mat_h, 4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error,
                    4 * (mat_h.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(CholeskyTestMC, CorrectnessLocal) {
  SizeType m, mb;

  for (auto uplo : blas_uplos) {
    for (auto sz : sizes) {
      std::tie(m, mb) = sz;
      testCholesky<TypeParam, Backend::MC, Device::CPU>(uplo, m, mb);
    }
  }
}

TYPED_TEST(CholeskyTestMC, CorrectnessDistributed) {
  SizeType m, mb;

  for (const auto& comm_grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto sz : sizes) {
        std::tie(m, mb) = sz;
        testCholesky<TypeParam, Backend::MC, Device::CPU>(comm_grid, uplo, m, mb);
        hpx::threads::get_thread_manager().wait();
      }
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(CholeskyTestGPU, CorrectnessLocal) {
  SizeType m, mb;

  for (auto uplo : blas_uplos) {
    for (auto sz : sizes) {
      std::tie(m, mb) = sz;
      testCholesky<TypeParam, Backend::GPU, Device::GPU>(uplo, m, mb);
    }
  }
}

TYPED_TEST(CholeskyTestGPU, CorrectnessDistributed) {
  SizeType m, mb;

  for (const auto& comm_grid : this->commGrids()) {
    for (auto uplo : blas_uplos) {
      for (auto sz : sizes) {
        std::tie(m, mb) = sz;
        testCholesky<TypeParam, Backend::GPU, Device::GPU>(comm_grid, uplo, m, mb);
        hpx::threads::get_thread_manager().wait();
      }
    }
  }
}
#endif
