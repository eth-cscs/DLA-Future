//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
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
#include "test_lapack_tile/test_laset.h"
#include "test_lapack_tile/test_potrf.h"
#include "dlaf_test/matrix/util_tile.h"

#include <cmath>

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;

const std::vector<blas::Diag> blas_diags({blas::Diag::Unit, blas::Diag::NonUnit});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Uplo> blas_geuplos({blas::Uplo::General, blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<lapack::Norm> lapack_norms({lapack::Norm::Fro, lapack::Norm::Inf, lapack::Norm::Max,
                                              lapack::Norm::One, lapack::Norm::Two});

template <class T, Device D>
class TileOperationsTest : public ::testing::Test {};

template <class T>
using TileOperationsTestMC = TileOperationsTest<T, Device::CPU>;

template <class T>
using RealTileOperationsTestMC = TileOperationsTest<T, Device::CPU>;

TYPED_TEST_SUITE(TileOperationsTestMC, MatrixElementTypes);
TYPED_TEST_SUITE(RealTileOperationsTestMC, RealMatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <class T>
using TileOperationsTestGPU = TileOperationsTest<T, Device::GPU>;

TYPED_TEST_SUITE(TileOperationsTestGPU, MatrixElementTypes);
#endif

// Tuple elements:  n, extra_lda, extra_ldb
std::vector<std::tuple<SizeType, SizeType, SizeType>> hegst_sizes = {{0, 0, 0},  {3, 0, 0},  {5, 3, 5},
                                                                     {9, 0, 7},  {9, 1, 0},  {17, 0, 3},
                                                                     {17, 7, 0}, {32, 0, 0}, {32, 4, 5}};

TYPED_TEST(TileOperationsTestMC, Hegst) {
  using Type = TypeParam;

  std::vector<int> itypes = {1, 2, 3};

  for (const auto& uplo : blas_uplos) {
    for (const auto& itype : itypes) {
      for (const auto& [m, extra_lda, extra_ldb] : hegst_sizes) {
        testHegst<Type, Device::CPU>(itype, uplo, m, extra_lda, extra_ldb);
      }
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TileOperationsTestGPU, Hegst) {
  using Type = TypeParam;

  std::vector<int> itypes = {1, 2, 3};

  for (const auto& uplo : blas_uplos) {
    for (const auto& itype : itypes) {
      for (const auto& [m, extra_lda, extra_ldb] : hegst_sizes) {
        testHegst<Type, Device::GPU>(itype, uplo, m, extra_lda, extra_ldb);
      }
    }
  }
}
#endif

TYPED_TEST(TileOperationsTestMC, lange) {
  std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                 {1, 1, 0},   {12, 8, 1},  {8, 12, 1},
                                                                 {12, 12, 1}, {11, 17, 3}, {11, 17, 0},
                                                                 {17, 11, 3}, {17, 11, 0}, {17, 17, 3},
                                                                 {11, 11, 0}};

  for (const auto& [m, n, extra_lda] : sizes) {
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
  std::vector<std::tuple<SizeType, SizeType, SizeType>> sizes = {{0, 0, 0},   {0, 0, 2},  // 0 size
                                                                 {1, 1, 0},   {17, 11, 3}, {17, 11, 0},
                                                                 {17, 17, 3}, {17, 17, 3}, {11, 11, 0}};

  for (auto [m, n, extra_lda] : sizes) {
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

  for (const auto uplo : blas_uplos) {
    for (const auto& [n, extra_lda] : potrf_sizes) {
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

  for (const auto uplo : blas_uplos) {
    for (const auto& [n, extra_lda] : potrf_sizes) {
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

  for (const auto uplo : blas_uplos) {
    for (const auto& [n, extra_lda] : potrf_sizes) {
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

  for (const auto uplo : blas_uplos) {
    for (const auto& [n, extra_lda] : potrf_sizes) {
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

std::vector<std::tuple<SizeType, SizeType, SizeType>> setsizes = {  //
    {0, 0, 0},   {0, 1, 2},   {3, 0, 1},   {1, 1, 0},    {17, 11, 3},  {17, 11, 0},
    {17, 17, 3}, {17, 17, 3}, {11, 11, 0}, {117, 35, 3}, {68, 117, 3}, {121, 121, 0}};

TYPED_TEST(TileOperationsTestMC, Laset) {
  using Type = TypeParam;

  const auto alpha = TypeUtilities<Type>::element(-3.5, 8.72);
  for (const auto beta : {alpha, TypeUtilities<Type>::element(-1.25, -7.21)}) {
    for (const auto& [m, n, extra_lda] : setsizes) {
      for (const auto uplo : blas_geuplos) {
        testLaset<Type, Device::CPU>(uplo, m, n, alpha, beta, extra_lda);
      }
    }
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TileOperationsTestGPU, Laset) {
  using Type = TypeParam;

  const auto alpha = TypeUtilities<Type>::element(-3.5, 8.72);
  for (const auto beta : {alpha, TypeUtilities<Type>::element(-1.25, -7.21)}) {
    for (const auto& [m, n, extra_lda] : setsizes) {
      for (const auto uplo : blas_geuplos) {
        testLaset<Type, Device::GPU>(uplo, m, n, alpha, beta, extra_lda);
      }
    }
  }
}

#endif

TYPED_TEST(TileOperationsTestMC, Set0) {
  using Type = TypeParam;

  for (const auto& [m, n, extra_lda] : setsizes) {
    testSet0<Type, Device::CPU>(m, n, extra_lda);
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(TileOperationsTestGPU, Set0) {
  using Type = TypeParam;

  for (const auto& [m, n, extra_lda] : setsizes) {
    testSet0<Type, Device::GPU>(m, n, extra_lda);
  }
}
#endif

TYPED_TEST(RealTileOperationsTestMC, Stedc) {
  using dlaf::matrix::test::createTile;

  using RealParam = BaseType<TypeParam>;

  constexpr double pi = 3.14159265358979323846;

  SizeType sz = 10;

  // Tridiagonal tile : 1D Laplacian
  auto tridiag_f = [](const TileElementIndex& idx) {
    if (idx.col() == 0) {
      // diagonal
      return RealParam(2);
    }
    else {
      // off-diagoanl
      return RealParam(-1);
    }
  };
  auto tridiag = createTile<RealParam, Device::CPU>(std::move(tridiag_f), TileElementSize(sz, 2), sz);

  auto evecs = createTile<TypeParam, Device::CPU>(TileElementSize(sz, sz), sz);
  set(evecs, TypeParam(0));

  tile::internal::stedc(tridiag, evecs);

  // Note that only the first column is relevant but to avoid copying to a separate buffer or 1D tile, we
  // also set the expected values as returned by `stedc` in the second column where the off-diagonal is
  // stored as well as the unused last entry.
  auto expected_tridiag_f = [sz](const TileElementIndex& idx) {
    if (idx.col() == 0) {
      // the diagonal (first column) holds the eigenvalues
      return RealParam(2 * (1 - std::cos(pi * (idx.row() + 1) / (sz + 1))));
    }
    else if (idx.col() == 1 && idx.row() == sz - 1) {
      // the last element of the second column is unused and is left unchanged
      return RealParam(-1);
    }
    else {
      // the off-diagonal is set to zero by `stedc`
      return RealParam(0);
    }
  };
  auto expected_tridiag =
      createTile<RealParam, Device::CPU>(std::move(expected_tridiag_f), TileElementSize(sz, 2), sz);

  auto expected_evecs_f = [sz](const TileElementIndex& idx) {
    SizeType j = idx.col() + 1;
    SizeType k = idx.row() + 1;
    return TypeParam(std::sqrt(2.0 / (sz + 1)) * std::sin(j * k * pi / (sz + 1)));
  };

  // Eigenvalues
  CHECK_TILE_NEAR(expected_tridiag, tridiag, sz * TypeUtilities<RealParam>::error,
                  sz * TypeUtilities<RealParam>::error);

  // Eigenvectors are unique up to a sign, this makes sure evecs have the same signs as the expected evecs
  for (SizeType i = 0; i < sz; ++i) {
    TileElementIndex idx(0, i);
    if (dlaf::util::sameSign(expected_evecs_f(idx), evecs(idx)))
      continue;

    tile::internal::scaleCol(TypeParam(-1), i, evecs);
  }
  CHECK_TILE_NEAR(expected_evecs_f, evecs, 1e-6, 1e-6);
}

TYPED_TEST(TileOperationsTestMC, ScaleCol) {
  TileElementSize tile_size{5, 5};
  auto tile_fn = [](const TileElementIndex& idx) { return TypeParam(idx.row() + idx.col()); };
  auto tile = createTile<TypeParam, Device::CPU>(std::move(tile_fn), tile_size, tile_size.rows());
  TypeParam alpha = 4.2;
  SizeType col = 3;

  tile::internal::scaleCol(alpha, col, tile);

  auto expected_tile_fn = [col, alpha](const TileElementIndex& idx) {
    TypeParam factor = (idx.col() == col) ? alpha : TypeParam(1);
    return TypeParam(idx.row() + idx.col()) * factor;
  };

  CHECK_TILE_NEAR(expected_tile_fn, tile, TypeUtilities<TypeParam>::error,
                  TypeUtilities<TypeParam>::error);
}
