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

TYPED_TEST(TileOperationsTest, GemmExceptions) {
  using Type = TypeParam;

  TileElementSize size_op_a(0, 0), size_op_b(0, 0), size_c(0, 0);
  SizeType extra_lda, extra_ldb, extra_ldc;

  std::vector<std::tuple<TileElementSize, TileElementSize, TileElementSize, SizeType, SizeType, SizeType>>
      sizes = {
          {{13, 20}, {20, 11}, {15, 11}, 0, 5, 7},  // different m
          {{15, 23}, {23, 11}, {13, 11}, 1, 2, 2},  // different m
          {{17, 20}, {20, 19}, {17, 17}, 2, 0, 3},  // different n
          {{17, 24}, {24, 17}, {17, 19}, 0, 0, 0},  // different n
          {{17, 7}, {13, 11}, {17, 11}, 7, 2, 2},   // different k
          {{17, 13}, {7, 11}, {17, 11}, 7, 1, 0},   // different k
          // Correct sizes which do not throw are: {{m, k}, {k, n}, {m, n}, >=0, >= 0, >= 0}
      };
  for (const auto op_a : blas_ops) {
    for (const auto op_b : blas_ops) {
      for (const auto& size : sizes) {
        std::tie(size_op_a, size_op_b, size_c, extra_lda, extra_ldb, extra_ldc) = size;

        // Test a and b const Tiles.
        testGemmExceptions<Type>(op_a, op_b, size_op_a, size_op_b, size_c, extra_lda, extra_ldb,
                                 extra_ldc);

        // Test a and b non const Tiles.
        testGemmExceptions<Type, Type>(op_a, op_b, size_op_a, size_op_b, size_c, extra_lda, extra_ldb,
                                       extra_ldc);
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

TYPED_TEST(TileOperationsTest, HerkExceptions) {
  using Type = TypeParam;

  auto herk_blas_ops = blas_ops;
  TileElementSize size_op_a(0, 0), size_c(0, 0);
  SizeType extra_lda, extra_ldc;

  std::vector<std::tuple<TileElementSize, TileElementSize, SizeType, SizeType>> sizes = {
      {{1, 0}, {0, 0}, 0, 0},     {{6, 12}, {5, 5}, 1, 0},
      {{10, 16}, {11, 11}, 1, 2}, {{13, 21}, {0, 0}, 1, 22},  // different n
      {{0, 0}, {0, 1}, 0, 0},     {{5, 12}, {5, 6}, 1, 0},
      {{11, 16}, {12, 11}, 1, 2}, {{0, 0}, {1, 0}, 1, 22},  // c not square
      // Correct sizes which do not throw are: {{n, k}, {n, n}, >= 0, >= 0}
  };

  for (const auto uplo : blas_uplos) {
    for (const auto op : blas_ops) {
      for (const auto& size : sizes) {
        std::tie(size_op_a, size_c, extra_lda, extra_ldc) = size;

        // Test a const Tile.
        testHerkExceptions<Type>(uplo, op, size_op_a, size_c, extra_lda, extra_ldc);

        // Test a non const Tile.
        testHerkExceptions<Type, Type>(uplo, op, size_op_a, size_c, extra_lda, extra_ldc);
      }
    }
  }

  // [c,z]herk do not allow op = Trans
  if (std::is_same<Type, ComplexType<Type>>::value) {
    std::vector<std::tuple<TileElementSize, TileElementSize, SizeType, SizeType>> ok_sizes = {
        {{0, 0}, {0, 0}, 0, 0},
        {{5, 12}, {5, 5}, 1, 0},
        {{11, 16}, {11, 11}, 1, 2},
    };

    auto op = blas::Op::Trans;
    for (const auto uplo : blas_uplos) {
      for (const auto& size : ok_sizes) {
        std::tie(size_op_a, size_c, extra_lda, extra_ldc) = size;

        // Test a const Tile.
        testHerkExceptions<Type>(uplo, op, size_op_a, size_c, extra_lda, extra_ldc);

        // Test a non const Tile.
        testHerkExceptions<Type, Type>(uplo, op, size_op_a, size_c, extra_lda, extra_ldc);
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

TYPED_TEST(TileOperationsTest, TrsmExceptions) {
  using Type = TypeParam;

  TileElementSize size_a(0, 0), size_b(0, 0);
  SizeType extra_lda, extra_ldb;

  std::vector<std::tuple<TileElementSize, TileElementSize, SizeType, SizeType>> sizes_left = {
      {{0, 0}, {1, 0}, 0, 0},     {{5, 5}, {6, 12}, 1, 0},
      {{11, 11}, {16, 11}, 1, 2}, {{0, 0}, {7, 7}, 1, 22},  // different m
      {{1, 0}, {0, 0}, 0, 0},     {{5, 6}, {5, 6}, 1, 0},
      {{11, 16}, {16, 11}, 1, 2}, {{6, 7}, {8, 9}, 1, 22},  // a not square
      // Correct sizes which do not throw are: {{m, m}, {m, n}, >= 0, >= 0}
  };
  std::vector<std::tuple<TileElementSize, TileElementSize, SizeType, SizeType>> sizes_right;

  for (const auto side : blas_sides) {
    auto& sizes = side == blas::Side::Left ? sizes_left : sizes_right;
    for (const auto uplo : blas_uplos) {
      for (const auto op : blas_ops) {
        for (const auto diag : blas_diags) {
          for (const auto& size : sizes) {
            std::tie(size_a, size_b, extra_lda, extra_ldb) = size;

            // Test a const Tile.
            testTrsmExceptions<Type>(side, uplo, op, diag, size_a, size_b, extra_lda, extra_ldb);

            // Test a non const Tile.
            testTrsmExceptions<Type, Type>(side, uplo, op, diag, size_a, size_b, extra_lda, extra_ldb);
          }
        }
      }
    }
  }
}
