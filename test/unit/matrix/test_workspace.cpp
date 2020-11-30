//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/workspace.h"

#include <vector>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/hpx_main.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_futures.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;

template <typename Type>
class WorkspaceLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(WorkspaceLocalTest, MatrixElementTypes);

TYPED_TEST(WorkspaceLocalTest, Basic) {
  using hpx::util::unwrapping;

  Distribution dist(LocalElementSize{2, 1}, TileElementSize{1, 1});

  constexpr auto VALUE = TypeUtilities<TypeParam>::element(26, 10);

  // init order matters!
  Matrix<TypeParam, Device::CPU> matrix(dist);
  matrix::util::set(matrix, [VALUE](auto&&) { return VALUE; });

  Workspace<TypeParam, Device::CPU> ws(dist);

  ws({0, 0}).then(unwrapping([](auto&& tile) {
    tile({0, 0}) = TypeUtilities<TypeParam>::element(13, 26);
  }));
  ws({1, 0}).then(unwrapping([](auto&& tile) {
    tile({0, 0}) = TypeUtilities<TypeParam>::element(5, 10);
  }));
  ws.read({0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtilities<TypeParam>::element(13, 26), tile({0, 0}));
  }));
  ws.read({1, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtilities<TypeParam>::element(5, 10), tile({0, 0}));
  }));

  for (const auto& index : common::iterate_range2d(matrix.distribution().localNrTiles()))
    matrix.read(index).then(unwrapping([VALUE](auto&& tile) { EXPECT_EQ(VALUE, tile({0, 0})); }));

  ws.set_tile({1, 0}, matrix.read(LocalTileIndex{0, 0}));

  // EXPECT_DEATH(ws.set_tile({1, 0}, matrix.read(LocalTileIndex{0, 0})), "you cannot set it again");

  ws.read({0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtilities<TypeParam>::element(13, 26), tile({0, 0}));
  }));
  ws.read({1, 0}).then(unwrapping([VALUE](auto&& tile) { EXPECT_EQ(VALUE, tile({0, 0})); }));
}
