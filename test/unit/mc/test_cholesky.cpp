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

#include <typeinfo>
#include "gtest/gtest.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf_test/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace matrix_test;
using namespace testing;

template <typename Type>
class CholeskyTest : public ::testing::Test {};

TYPED_TEST_CASE(CholeskyTest, MatrixElementTypes);

std::vector<GlobalElementSize> square_sizes({{10, 10}, {25, 25}, {0, 0}});
std::vector<GlobalElementSize> rectangular_sizes({{10, 20}, {50, 20}, {0, 10}, {20, 0}});
std::vector<TileElementSize> square_block_sizes({{5, 5}, {20, 20}});
std::vector<TileElementSize> rectangular_block_sizes({{10, 30}, {20, 10}});

TYPED_TEST(CholeskyTest, Correctness) {
  using Type = TypeParam;

  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + 0.001 * j, j - 0.01 * i);
  };

  for (const auto& size : square_sizes) {
    for (const auto& block_size : square_block_sizes) {
      Matrix<Type, Device::CPU> mat(size, block_size);

      set(mat, el);

      EXPECT_NO_THROW(cholesky_local(mat));
    }
  }
}

  TYPED_TEST(CholeskyTest, NoSquareMatrixException) {
    using Type = TypeParam;

    auto el = [](const GlobalElementIndex& index) {
      SizeType i = index.row();
      SizeType j = index.col();
      return TypeUtilities<Type>::element(i + 0.001 * j, j - 0.01 * i);
    };

    // Check for square sizes
    for (const auto& size : square_sizes) {
      for (const auto& block_size : square_block_sizes) {
        Matrix<Type, Device::CPU> mat(size, block_size);

        set(mat, el);

        EXPECT_NO_THROW(cholesky_local(mat));
      }
    }

    // Check for rectangular sizes
    for (const auto& size : rectangular_sizes) {
      for (const auto& block_size : square_block_sizes) {
        Matrix<Type, Device::CPU> mat(size, block_size);

        EXPECT_THROW(cholesky_local(mat), std::invalid_argument);
      }
    }
  }

  TYPED_TEST(CholeskyTest, NoSquareBlockException) {
    using Type = TypeParam;

    auto el = [](const GlobalElementIndex& index) {
      SizeType i = index.row();
      SizeType j = index.col();
      return TypeUtilities<Type>::element(i + 0.001 * j, j - 0.01 * i);
    };

    // Check for square sizes
    for (const auto& size : square_sizes) {
      for (const auto& block_size : square_block_sizes) {
        Matrix<Type, Device::CPU> mat(size, block_size);

        set(mat, el);

        EXPECT_NO_THROW(cholesky_local(mat));
      }
    }

    // Check for rectangular sizes
    for (const auto& size : square_sizes) {
      for (const auto& block_size : rectangular_block_sizes) {
        Matrix<Type, Device::CPU> mat(size, block_size);

        EXPECT_THROW(cholesky_local(mat), std::invalid_argument);
      }
    }
  }

  TYPED_TEST(CholeskyTest, NonPositiveDefiniteException) {}
