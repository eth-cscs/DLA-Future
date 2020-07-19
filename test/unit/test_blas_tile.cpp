//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/blas_tile.h"

#include "gtest/gtest.h"
#include "dlaf_test/util_types.h"

#include "test_blas_tile/test_gemm.h"
#include "test_blas_tile/test_herk.h"
#include "test_blas_tile/test_trsm.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace testing;

const std::vector<blas::Diag> blas_diags({blas::Diag::Unit, blas::Diag::NonUnit});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

template <typename Type>
class TileOperationsTest : public ::testing::Test {};

TYPED_TEST_SUITE(TileOperationsTest, MatrixElementTypes);

TYPED_TEST(TileOperationsTest, Gemm) {
  using Type = TypeParam;

  SizeType m, n, k, extra_lda, extra_ldb, extra_ldc;

  std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType, SizeType>> sizes = {
      {0, 0, 0, 0, 0, 0},                                               // all 0 sizes
      {7, 0, 0, 3, 1, 0},  {0, 5, 0, 0, 0, 1},    {0, 0, 11, 1, 1, 2},  // two 0 sizes
      {0, 5, 13, 1, 0, 1}, {7, 0, 4, 1, 2, 0},    {3, 11, 0, 0, 1, 0},  // one 0 size
      {1, 1, 1, 0, 3, 0},  {1, 12, 1, 1, 0, 7},   {17, 12, 16, 1, 3, 0}, {11, 23, 8, 0, 3, 4},
      {6, 9, 12, 1, 1, 1}, {32, 32, 32, 0, 0, 0}, {32, 32, 32, 4, 5, 7},
  };

  for (const auto op_a : blas_ops) {
    for (const auto op_b : blas_ops) {
      for (const auto& size : sizes) {
        std::tie(m, n, k, extra_lda, extra_ldb, extra_ldc) = size;

        // Test a and b const Tiles.
        testGemm<Type>(op_a, op_b, m, n, k, extra_lda, extra_ldb, extra_ldc);

        // Test a and b non const Tiles.
        testGemm<Type, Type>(op_a, op_b, m, n, k, extra_lda, extra_ldb, extra_ldc);
      }
    }
  }
}

TYPED_TEST(TileOperationsTest, Herk) {
  using Type = TypeParam;

  auto herk_blas_ops = blas_ops;
  // [c,z]herk do not allow op = Trans
  if (std::is_same<Type, ComplexType<Type>>::value)
    herk_blas_ops = {blas::Op::NoTrans, blas::Op::ConjTrans};
  SizeType n, k, extra_lda, extra_ldc;

  std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes =
      {{0, 0, 0, 0},                 // all 0 sizes
       {0, 5, 1, 0},  {7, 0, 1, 2},  // one 0 size
       {1, 1, 0, 3},  {1, 12, 1, 0},  {17, 12, 1, 3}, {11, 23, 0, 3},
       {9, 12, 1, 1}, {32, 32, 0, 0}, {32, 32, 4, 7}};

  for (const auto uplo : blas_uplos) {
    for (const auto op : herk_blas_ops) {
      for (const auto& size : sizes) {
        std::tie(n, k, extra_lda, extra_ldc) = size;

        // Test a const Tile.
        testHerk<Type>(uplo, op, n, k, extra_lda, extra_ldc);

        // Test a non const Tile.
        testHerk<Type, Type>(uplo, op, n, k, extra_lda, extra_ldc);
      }
    }
  }
}

TYPED_TEST(TileOperationsTest, Trsm) {
  using Type = TypeParam;

  SizeType m, n, extra_lda, extra_ldb;

  std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes =
      {{0, 0, 0, 0},                 // all 0 sizes
       {0, 5, 1, 0},  {7, 0, 1, 2},  // one 0 size
       {1, 1, 0, 3},  {1, 12, 1, 0},  {17, 12, 1, 3}, {11, 23, 0, 3},
       {9, 12, 1, 1}, {32, 32, 0, 0}, {32, 32, 4, 7}};

  for (const auto side : blas_sides) {
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& size : sizes) {
            std::tie(m, n, extra_lda, extra_ldb) = size;

            // Test a const Tile.
            testTrsm<TileElementIndex, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);

            // Test a non const Tile.
            testTrsm<TileElementIndex, Type, Type>(side, uplo, op, diag, m, n, extra_lda, extra_ldb);
          }
        }
      }
    }
  }
}
