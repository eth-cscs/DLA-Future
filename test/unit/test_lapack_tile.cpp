//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/lapack_tile.h"

#include "gtest/gtest.h"
#include "dlaf_test/util_types.h"

#include "test_lapack_tile/test_lange.h"
#include "test_lapack_tile/test_lantr.h"
#include "test_lapack_tile/test_potrf.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});
const std::vector<lapack::Norm> lapack_norms({lapack::Norm::Fro, lapack::Norm::Inf, lapack::Norm::Max,
                                              lapack::Norm::One, lapack::Norm::Two});

template <typename Type>
class TileOperationsTest : public ::testing::Test {};

TYPED_TEST_SUITE(TileOperationsTest, MatrixElementTypes);

TYPED_TEST(TileOperationsTest, lange) {
  SizeType m, n, extra_lda;

  std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                 {1, 1, 0},   {12, 8, 1},  {8, 12, 1},
                                                                 {12, 12, 1}, {11, 17, 3}, {11, 17, 0},
                                                                 {17, 11, 3}, {17, 11, 0}, {17, 17, 3},
                                                                 {11, 11, 0}};

  for (const auto norm : lapack_norms) {
    // lange cannot be tested for norm2
    if (norm == lapack::Norm::Two)
      continue;

    for (const auto& size : sizes) {
      std::tie(m, n, extra_lda) = size;

      dlaf::test::lange::run<TypeParam>(norm, TileElementSize{m, n}, extra_lda);
    }
  }
}

TYPED_TEST(TileOperationsTest, lantr) {
  SizeType m, n, extra_lda;

  std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                 {1, 1, 0},   {17, 11, 3}, {17, 11, 0},
                                                                 {17, 17, 3}, {17, 17, 3}, {11, 11, 0}};

  for (const auto norm : lapack_norms) {
    for (const auto uplo : blas_uplos) {
      for (const auto diag : blas_diags) {
        for (const auto& size : sizes) {
          std::tie(m, n, extra_lda) = size;

          // transpose rectangular matrix to be useful for upper triangular case
          if (blas::Uplo::Upper == uplo)
            std::swap(m, n);

          testLantr<TypeParam>(norm, uplo, diag, m, n, extra_lda);
        }
      }
    }
  }
}

TYPED_TEST(TileOperationsTest, Potrf) {
  using Type = TypeParam;

  SizeType n, extra_lda;

  std::vector<std::tuple<SizeType, SizeType>> sizes = {{0, 0}, {0, 2},  // 0 size
                                                       {1, 0}, {12, 1}, {17, 3}, {11, 0}};

  for (const auto uplo : blas_uplos) {
    for (const auto& size : sizes) {
      std::tie(n, extra_lda) = size;

      // Test version non returning info
      testPotrf<Type, false>(uplo, n, extra_lda);

      // Test version returning info
      testPotrf<Type, true>(uplo, n, extra_lda);
    }
  }
}

TYPED_TEST(TileOperationsTest, PotrfArgumentsExceptions) {
  using Type = TypeParam;

  TileElementSize size_a(0, 0);
  SizeType extra_lda;

  std::vector<std::tuple<TileElementSize, SizeType>> sizes = {{{0, 1}, 0},   {{1, 0}, 2},  // 0 size
                                                              {{3, 7}, 1},   {{12, 9}, 1},
                                                              {{64, 32}, 3}, {{2, 32}, 0}};

  for (const auto uplo : blas_uplos) {
    for (const auto& size : sizes) {
      std::tie(size_a, extra_lda) = size;
      // Test version non returning info
      testPotrfArgExceptions<Type, false>(uplo, size_a, extra_lda);

      // Test version returning info
      testPotrfArgExceptions<Type, true>(uplo, size_a, extra_lda);
    }
  }
}

TYPED_TEST(TileOperationsTest, PotrfNonPositiveDefinite) {
  using Type = TypeParam;

  SizeType n, extra_lda;

  std::vector<std::tuple<SizeType, SizeType>> sizes = {{1, 0}, {12, 1}, {17, 3}, {11, 0}};

  for (const auto uplo : blas_uplos) {
    for (const auto& size : sizes) {
      std::tie(n, extra_lda) = size;

      // Test version non returning info
      testPotrfNonPosDef<Type, false>(uplo, n, extra_lda);

      // Test version returning info
      testPotrfNonPosDef<Type, true>(uplo, n, extra_lda);
    }
  }
}
