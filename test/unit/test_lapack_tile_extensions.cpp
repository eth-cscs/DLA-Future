//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

#include <dlaf/lapack/tile.h>

#include "test_lapack_tile_extensions/test_lauum_workspace.h"
#include "test_lapack_tile_extensions/test_trtri_workspace.h"

#include <gtest/gtest.h>

#include <dlaf_test/matrix/util_tile.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;

const std::vector<blas::Diag> blas_diags({blas::Diag::Unit, blas::Diag::NonUnit});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

template <class T, Device D>
class TileOperationsTest : public ::testing::Test {};

template <class T>
using TileOperationsTestMC = TileOperationsTest<T, Device::CPU>;

template <class T>
using RealTileOperationsTestMC = TileOperationsTest<T, Device::CPU>;

TYPED_TEST_SUITE(TileOperationsTestMC, MatrixElementTypes);
TYPED_TEST_SUITE(RealTileOperationsTestMC, RealMatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using TileOperationsTestGPU = TileOperationsTest<T, Device::GPU>;

TYPED_TEST_SUITE(TileOperationsTestGPU, MatrixElementTypes);
#endif

// Tuple elements:  n, extra_lda
std::vector<std::tuple<SizeType, SizeType>> lauum_sizes = {
    {0, 0}, {0, 2},  // 0 size
    {1, 0}, {12, 1}, {17, 3}, {11, 0}, {128, 0},
};

TYPED_TEST(TileOperationsTestMC, LauumWorkspace) {
  using Type = TypeParam;

  for (const auto uplo : blas_uplos) {
    for (const auto& [n, extra_lda] : lauum_sizes) {
      test_lauum_workspace<Type, Device::CPU>(uplo, n, extra_lda);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, LauumWorkspace) {
  using Type = TypeParam;

  for (const auto uplo : blas_uplos) {
    for (const auto& [n, extra_lda] : lauum_sizes) {
      test_lauum_workspace<Type, Device::GPU>(uplo, n, extra_lda);
    }
  }
}
#endif

// Tuple elements:  n, extra_lda
std::vector<std::tuple<SizeType, SizeType>> trtri_sizes = {
    {0, 0}, {0, 2},  // 0 size
    {1, 0}, {12, 1}, {17, 3}, {11, 0}, {128, 0},
};

TYPED_TEST(TileOperationsTestMC, TrtriWorkspace) {
  using Type = TypeParam;

  for (const auto uplo : blas_uplos) {
    for (const auto diag : blas_diags) {
      for (const auto& [n, extra_lda] : trtri_sizes) {
        test_trtri_workspace<Type, Device::CPU>(uplo, diag, n, extra_lda);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, TrtriWorkspace) {
  using Type = TypeParam;

  for (const auto uplo : blas_uplos) {
    for (const auto diag : blas_diags) {
      for (const auto& [n, extra_lda] : trtri_sizes) {
        test_trtri_workspace<Type, Device::GPU>(uplo, diag, n, extra_lda);
      }
    }
  }
}
#endif
