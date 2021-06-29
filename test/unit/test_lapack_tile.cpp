//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/lapack/tile.h"

#include "gtest/gtest.h"

#include "test_lapack_tile/test_hegst.h"
#include "test_lapack_tile/test_lange.h"
#include "test_lapack_tile/test_lantr.h"
#include "test_lapack_tile/test_potrf.h"
#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;

const std::vector<blas::Diag> blas_diags({blas::Diag::Unit, blas::Diag::NonUnit});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<lapack::MatrixType> lapack_matrices({lapack::MatrixType::General,
                                                       lapack::MatrixType::Lower,
                                                       lapack::MatrixType::Upper});
const std::vector<lapack::Norm> lapack_norms({lapack::Norm::Fro, lapack::Norm::Inf, lapack::Norm::Max,
                                              lapack::Norm::One, lapack::Norm::Two});

template <typename Type>
class TileOperationsTestMC : public ::testing::Test {};

TYPED_TEST_SUITE(TileOperationsTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <typename Type>
class TileOperationsTestGPU : public ::testing::Test {};

TYPED_TEST_SUITE(TileOperationsTestGPU, MatrixElementTypes);
#endif

// Tuple elements:  n, extra_lda, extra_ldb
std::vector<std::tuple<SizeType, SizeType, SizeType>> hegst_sizes = {{0, 0, 0},  {3, 0, 0},  {5, 3, 5},
                                                                     {9, 0, 7},  {9, 1, 0},  {17, 0, 3},
                                                                     {17, 7, 0}, {32, 0, 0}, {32, 4, 5}};

TYPED_TEST(TileOperationsTestMC, Hegst) {
  using Type = TypeParam;
  SizeType m, extra_lda, extra_ldb;

  std::vector<int> itypes = {1, 2, 3};

  for (const auto& uplo : blas_uplos) {
    for (const auto& itype : itypes) {
      for (const auto& size : hegst_sizes) {
        std::tie(m, extra_lda, extra_ldb) = size;
        testHegst<Type, Device::CPU>(itype, uplo, m, extra_lda, extra_ldb);
      }
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TileOperationsTestGPU, Hegst) {
  using Type = TypeParam;
  SizeType m, extra_lda, extra_ldb;

  std::vector<int> itypes = {1, 2, 3};

  for (const auto& uplo : blas_uplos) {
    for (const auto& itype : itypes) {
      for (const auto& size : hegst_sizes) {
        std::tie(m, extra_lda, extra_ldb) = size;
        testHegst<Type, Device::GPU>(itype, uplo, m, extra_lda, extra_ldb);
      }
    }
  }
}
#endif

TYPED_TEST(TileOperationsTestMC, lange) {
  SizeType m, n, extra_lda;

  std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                 {1, 1, 0},   {12, 8, 1},  {8, 12, 1},
                                                                 {12, 12, 1}, {11, 17, 3}, {11, 17, 0},
                                                                 {17, 11, 3}, {17, 11, 0}, {17, 17, 3},
                                                                 {11, 11, 0}};

  for (const auto& size : sizes) {
    std::tie(m, n, extra_lda) = size;

    const SizeType lda = std::max<SizeType>(1, m) + extra_lda;
    auto tile = createTile<TypeParam, Device::CPU>(TileElementSize{m, n}, lda);

    for (const auto norm : lapack_norms) {
      // lange does not support norm2
      if (norm == lapack::Norm::Two)
        continue;

      dlaf::test::lange::run<TypeParam>(norm, tile);
    }
  }
}

TYPED_TEST(TileOperationsTestMC, lantr) {
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

      const SizeType lda = std::max<SizeType>(1, m) + extra_lda;
      auto tile = createTile<TypeParam, Device::CPU>(TileElementSize{m, n}, lda);

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

// Tuple elements:  n, extra_lda
std::vector<std::tuple<SizeType, SizeType>> potrf_sizes = {{0, 0}, {0, 2},  // 0 size
                                                           {1, 0}, {12, 1}, {17, 3}, {11, 0}};

TYPED_TEST(TileOperationsTestMC, Potrf) {
  using Type = TypeParam;
  SizeType n, extra_lda;

  for (const auto uplo : blas_uplos) {
    for (const auto& size : potrf_sizes) {
      std::tie(n, extra_lda) = size;

      // Test version non returning info
      testPotrf<Type, Device::CPU, false>(uplo, n, extra_lda);

      // Test version returning info
      testPotrf<Type, Device::CPU, true>(uplo, n, extra_lda);
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TileOperationsTestGPU, Potrf) {
  using Type = TypeParam;
  SizeType n, extra_lda;

  for (const auto uplo : blas_uplos) {
    for (const auto& size : potrf_sizes) {
      std::tie(n, extra_lda) = size;

      // Test version non returning info
      testPotrf<Type, Device::GPU, false>(uplo, n, extra_lda);

      // Test version returning info
      testPotrf<Type, Device::GPU, true>(uplo, n, extra_lda);
    }
  }
}
#endif

TYPED_TEST(TileOperationsTestMC, PotrfNonPositiveDefinite) {
  using Type = TypeParam;
  SizeType n, extra_lda;

  for (const auto uplo : blas_uplos) {
    for (const auto& size : potrf_sizes) {
      std::tie(n, extra_lda) = size;
      if (n == 0)
        continue;

      // Only test version returning info
      testPotrfNonPosDef<Type, Device::CPU>(uplo, n, extra_lda);
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TileOperationsTestGPU, PotrfNonPositiveDefinite) {
  using Type = TypeParam;
  SizeType n, extra_lda;

  for (const auto uplo : blas_uplos) {
    for (const auto& size : potrf_sizes) {
      std::tie(n, extra_lda) = size;
      if (n == 0)
        continue;

      // Only test version returning info
      testPotrfNonPosDef<Type, Device::GPU>(uplo, n, extra_lda);
    }
  }
}
#endif

TYPED_TEST(TileOperationsTestMC, Lacpy) {
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

std::vector<std::tuple<SizeType, SizeType, SizeType>> setsizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                  {1, 1, 0},   {17, 11, 3}, {17, 11, 0},
                                                                  {17, 17, 3}, {17, 17, 3}, {11, 11, 0}};

TYPED_TEST(TileOperationsTestMC, Laset) {
  SizeType m, n, extra_lda;

  for (const auto& size : setsizes) {
    for (const auto mtype : lapack_matrices) {
      std::tie(m, n, extra_lda) = size;
      const SizeType lda = std::max<SizeType>(1, m) + extra_lda;
      Tile<TypeParam, Device::CPU> tile =
          createTile<TypeParam>([](TileElementIndex idx) { return idx.row() + idx.col(); },
                                TileElementSize(m, n), lda);

      auto res = [mtype](const TileElementIndex& index) {
        const double i = index.row();
        const double j = index.col();
        if ((mtype == lapack::MatrixType::Lower && i > j) ||
            (mtype == lapack::MatrixType::Upper && i < j) ||
            (mtype == lapack::MatrixType::General && i != j))
          return TypeUtilities<TypeParam>::element(0.0, 0.0);

        if (i == j)
          return TypeUtilities<TypeParam>::element(1.0, 0.0);

        return TypeUtilities<TypeParam>::element(i + j, 0.0);
      };

      tile::laset<TypeParam>(mtype, 0.f, 1.f, tile);
      CHECK_TILE_NEAR(res, tile, 4 * (n + 1) * TypeUtilities<TypeParam>::error,
                      4 * (n + 1) * TypeUtilities<TypeParam>::error);
    }
  }
}

TYPED_TEST(TileOperationsTestMC, Set0) {
  SizeType m, n, extra_lda;

  for (const auto& size : setsizes) {
    std::tie(m, n, extra_lda) = size;
    const SizeType lda = std::max<SizeType>(1, m) + extra_lda;
    Tile<TypeParam, Device::CPU> tile =
        createTile<TypeParam>([](TileElementIndex idx) { return idx.row() + idx.col(); },
                              TileElementSize(m, n), lda);

    auto res = [](const TileElementIndex& index) { return TypeUtilities<TypeParam>::element(0.0, 0.0); };

    tile::set0(tile);
    CHECK_TILE_EQ(res, tile);
  }
}
