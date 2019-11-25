//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/mc/cholesky_local.h"

#include "gtest/gtest.h"
#include "dlaf/matrix.h"
#include "dlaf_test/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace matrix_test;
using namespace testing;

template <typename Type>
class CholeskyTest : public ::testing::Test {};

TYPED_TEST_CASE(CholeskyTest, MatrixElementTypes);

std::vector<GlobalElementSize> square_sizes({{10, 10}, {25, 25}, {12, 12}, {0, 0}});
std::vector<GlobalElementSize> rectangular_sizes({{10, 20}, {50, 20}, {0, 10}, {20, 0}});
std::vector<TileElementSize> square_block_sizes({{5, 5}, {20, 20}});
std::vector<TileElementSize> rectangular_block_sizes({{10, 30}, {20, 10}});

TYPED_TEST(CholeskyTest, Correctness) {
  using Type = TypeParam;

  // Note: The tile elements are chosen such that:
  // - res_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // a_ij = Sum_k(res_ik * ConjTrans(res)_kj) =
  //      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
  // where k = 0 .. min(i,j)
  // Therefore,
  // a_ij = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(-i+j))
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    if (i < j)
      return TypeUtilities<TypeParam>::element(-9.9, 0.0);

    return TypeUtilities<TypeParam>::polar(std::exp2(-(i + j)) / 3 *
                                               (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                           -i + j);
  };

  // Analytical results
  auto res = [](const TileElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    if (i < j)
      return TypeUtilities<TypeParam>::element(-9.9, 0.0);

    return TypeUtilities<TypeParam>::polar(std::exp2(-std::abs(i - j)), -i + j);
  };

  for (const auto& size : square_sizes) {
    for (const auto& block_size : square_block_sizes) {
      // Matrix to undergo Cholesky decomposition
      Matrix<Type, Device::CPU> mat(size, block_size);
      set(mat, el);

      EXPECT_NO_THROW(cholesky_local(blas::Uplo::Lower, mat));

      CHECK_MATRIX_NEAR(res, mat, 4 * (mat.size().rows() + 1) * TypeUtilities<TypeParam>::error,
                        4 * (mat.size().rows() + 1) * TypeUtilities<TypeParam>::error);
    }
  }
}

TYPED_TEST(CholeskyTest, NoSquareMatrixException) {
  using Type = TypeParam;

  // Check for rectangular sizes
  for (const auto& size : rectangular_sizes) {
    for (const auto& block_size : square_block_sizes) {
      Matrix<Type, Device::CPU> mat(size, block_size);

      EXPECT_THROW(cholesky_local(blas::Uplo::Lower, mat), std::invalid_argument);
    }
  }
}

TYPED_TEST(CholeskyTest, NoSquareBlockException) {
  using Type = TypeParam;

  // Check for rectangular sizes
  for (const auto& size : square_sizes) {
    for (const auto& block_size : rectangular_block_sizes) {
      Matrix<Type, Device::CPU> mat(size, block_size);

      EXPECT_THROW(cholesky_local(blas::Uplo::Lower, mat), std::invalid_argument);
    }
  }
}

TYPED_TEST(CholeskyTest, NonPositiveDefiniteException) {}
