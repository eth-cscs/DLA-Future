//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/lapack_tile.h"

#include "gtest/gtest.h"
#include "dlaf_test/util_types.h"

#include "test_lapack_tile/test_hegst.h"
#include "test_lapack_tile/test_lange.h"
#include "test_lapack_tile/test_lantr.h"
#include "test_lapack_tile/test_potrf.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});
const std::vector<lapack::Norm> lapack_norms({lapack::Norm::Fro, lapack::Norm::Inf, lapack::Norm::Max,
                                              lapack::Norm::One, lapack::Norm::Two});

template <class T>
Tile<T, Device::CPU> allocate_tile(TileElementSize size, SizeType extra_lda) {
  using dlaf::memory::MemoryView;

  SizeType lda = std::max<SizeType>(1, size.rows()) + extra_lda;

  MemoryView<T, Device::CPU> mem_a(lda * size.cols());
  return {size, std::move(mem_a), lda};
}

template <typename Type>
class TileOperationsTest : public ::testing::Test {};

TYPED_TEST_SUITE(TileOperationsTest, MatrixElementTypes);

TYPED_TEST(TileOperationsTest, Hegst) {
  using Type = TypeParam;

  SizeType m, extra_ld;

  std::vector<std::tuple<SizeType, SizeType>> sizes = {{0, 0},  {3, 0},  {5, 3},  {9, 0}, {9, 1},
                                                       {17, 0}, {17, 7}, {32, 0}, {32, 4}};

  std::vector<int> itypes = {1, 2, 3};

  for (const auto& size : sizes) {
    for (const auto& uplo : blas_uplos) {
      for (const auto& itype : itypes) {
        std::tie(m, extra_ld) = size;
        testHegst<TileElementIndex, Type>(itype, uplo, m, extra_ld);
      }
    }
  }
}

TYPED_TEST(TileOperationsTest, lange) {
  SizeType m, n, extra_lda;

  std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                 {1, 1, 0},   {12, 8, 1},  {8, 12, 1},
                                                                 {12, 12, 1}, {11, 17, 3}, {11, 17, 0},
                                                                 {17, 11, 3}, {17, 11, 0}, {17, 17, 3},
                                                                 {11, 11, 0}};

  for (const auto& size : sizes) {
    std::tie(m, n, extra_lda) = size;

    auto tile = allocate_tile<TypeParam>(TileElementSize{m, n}, extra_lda);

    for (const auto norm : lapack_norms) {
      // lange does not support norm2
      if (norm == lapack::Norm::Two)
        continue;

      dlaf::test::lange::run<TypeParam>(norm, tile);
    }
  }
}

TYPED_TEST(TileOperationsTest, lantr) {
  SizeType m, n, extra_lda;

  std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                 {1, 1, 0},   {17, 11, 3}, {17, 11, 0},
                                                                 {17, 17, 3}, {17, 17, 3}, {11, 11, 0}};

  for (const auto& size : sizes) {
    std::tie(m, n, extra_lda) = size;

    for (const auto uplo : blas_uplos) {
      // transpose rectangular matrix to be useful for upper triangular case
      if (blas::Uplo::Upper == uplo)
        std::swap(m, n);

      auto tile = allocate_tile<TypeParam>(TileElementSize{m, n}, extra_lda);

      for (const auto norm : lapack_norms) {
        // lantr does not support norm2
        if (norm == lapack::Norm::Two)
          continue;

        for (const auto diag : blas_diags) {
          dlaf::test::lantr::run<TypeParam>(norm, uplo, diag, tile);
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

TYPED_TEST(TileOperationsTest, Lacpy) {
  using Scalar = TypeParam;
  using Tile_t = Tile<Scalar, Device::CPU>;
  using ConstTile_t = Tile<const Scalar, Device::CPU>;

  TileElementSize region(3, 3);
  TileElementIndex in_idx(1, 2);
  ConstTile_t in_tile = createTile<Scalar>([](TileElementIndex idx) { return idx.row() + idx.col(); },
                                           TileElementSize(5, 5), 5);
  TileElementIndex out_idx(2, 3);
  Tile_t out_tile = createTile<Scalar>([](TileElementIndex) { return 2; }, TileElementSize(7, 7), 7);

  tile::lacpy(region, in_idx, in_tile, out_idx, out_tile);

  double eps = std::numeric_limits<double>::epsilon();

  ASSERT_TRUE(std::abs(Scalar(1 + 2) - out_tile(TileElementIndex(2, 3))) < eps);
  ASSERT_TRUE(std::abs(Scalar(2 + 2) - out_tile(TileElementIndex(3, 3))) < eps);
  ASSERT_TRUE(std::abs(Scalar(3 + 2) - out_tile(TileElementIndex(4, 3))) < eps);

  ASSERT_TRUE(std::abs(Scalar(1 + 3) - out_tile(TileElementIndex(2, 4))) < eps);
  ASSERT_TRUE(std::abs(Scalar(2 + 3) - out_tile(TileElementIndex(3, 4))) < eps);
  ASSERT_TRUE(std::abs(Scalar(3 + 3) - out_tile(TileElementIndex(4, 4))) < eps);

  ASSERT_TRUE(std::abs(Scalar(1 + 4) - out_tile(TileElementIndex(2, 5))) < eps);
  ASSERT_TRUE(std::abs(Scalar(2 + 4) - out_tile(TileElementIndex(3, 5))) < eps);
  ASSERT_TRUE(std::abs(Scalar(3 + 4) - out_tile(TileElementIndex(4, 5))) < eps);
}
