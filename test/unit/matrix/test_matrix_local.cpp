//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/util_math.h"
#include "dlaf_test/matrix/matrix_local.h"

#include <vector>

#include <gtest/gtest.h>

#include "dlaf_test/util_types.h"
#include "dlaf_test/matrix/util_matrix_local.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

template <typename Type>
class MatrixLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixLocalTest, MatrixElementTypes);

struct TestSizes {
  GlobalElementSize size;
  TileElementSize block_size;
};

const std::vector<TestSizes> sizes_tests({
    //{{0, 0}, {11, 13}},
    {{3, 0}, {1, 2}},
    //{{0, 1}, {7, 32}},
    {{15, 18}, {5, 9}},
    {{6, 6}, {2, 2}},
    {{3, 4}, {24, 15}},
    {{16, 24}, {3, 5}},
});

TYPED_TEST(MatrixLocalTest, ConstructorAndShape) {
  for (const auto& test : sizes_tests) {
    const GlobalTileSize nrTiles{
      dlaf::util::ceilDiv(test.size.rows(), test.block_size.rows()),
      dlaf::util::ceilDiv(test.size.cols(), test.block_size.cols()),
    };

    const MatrixLocal<const TypeParam> mat(test.size, test.block_size);

    EXPECT_EQ(test.size, mat.size());
    EXPECT_EQ(test.block_size, mat.blockSize());

    EXPECT_EQ(nrTiles, mat.nrTiles());

    EXPECT_EQ(test.size.rows(), mat.ld());
  }
}

TYPED_TEST(MatrixLocalTest, Set) {
  auto el = [](const GlobalElementIndex& index) {
    const auto i = index.row();
    const auto j = index.col();
    return TypeUtilities<TypeParam>::element(i + j / 1024., j - i / 128.);
  };

  for (const auto& test : sizes_tests) {
    const GlobalTileSize nrTiles{
      dlaf::util::ceilDiv(test.size.rows(), test.block_size.rows()),
      dlaf::util::ceilDiv(test.size.cols(), test.block_size.cols()),
    };

    MatrixLocal<TypeParam> mat(test.size, test.block_size);

    set(mat, el);

    CHECK_MATRIX_NEAR(el, mat, 1e-3, 1e-3);
  }
}

