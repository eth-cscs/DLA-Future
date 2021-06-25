//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/extra_buffers.h"

#include <vector>

#include <gtest/gtest.h>
#include <hpx/include/util.hpp>

#include "dlaf/matrix/matrix.h"

#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace testing;

template <typename Type>
class ExtraBuffersTest : public ::testing::Test {};

TYPED_TEST_SUITE(ExtraBuffersTest, MatrixElementTypes);

TYPED_TEST(ExtraBuffersTest, AccessBuffers) {
  using TypeUtils = TypeUtilities<TypeParam>;

  const TileElementSize blocksize(1, 1);
  Matrix<TypeParam, Device::CPU> matrix({1, 1}, blocksize);
  matrix::test::set(matrix, [](auto&&) { return TypeUtils::element(1, 0); });

  const SizeType tot_buffers = 10;
  ExtraBuffers<TypeParam> buffers(matrix(LocalTileIndex{0, 0}), tot_buffers - 1, blocksize);

  for (auto i = 0; i < tot_buffers; ++i) {
    buffers.get_buffer(i).then(hpx::unwrapping([i](const auto& tile) {
      matrix::test::set(tile, [i](auto&&) { return TypeUtils::element(i, 0); });
    }));
  }

  for (auto i = 0; i < tot_buffers; ++i) {
    auto value_func = [i /*, tot_buffers*/](const TileElementIndex&) {
      return TypeUtils::element((i % tot_buffers), 0);
    };
    CHECK_TILE_EQ(std::move(value_func), buffers.get_buffer(i).get());
  }

  buffers.reduce();
}

TYPED_TEST(ExtraBuffersTest, BasicUsage) {
  const TileElementSize blocksize(1, 1);
  Matrix<TypeParam, Device::CPU> matrix({1, 1}, blocksize);
  matrix::test::set(matrix, [](auto&&) { return 1; });

  ExtraBuffers<TypeParam> buffers(matrix(LocalTileIndex{0, 0}), 2, blocksize);

  matrix::test::set(buffers.get_buffer(1).get(), [](auto&&) { return 3; });
  matrix::test::set(buffers.get_buffer(2).get(), [](auto&&) { return 5; });

  buffers.reduce();

  CHECK_MATRIX_EQ([](const GlobalElementIndex&) { return TypeUtilities<TypeParam>::element(9, 0); },
                  matrix);
}

TYPED_TEST(ExtraBuffersTest, FutureOrder) {
  using hpx::unwrapping;
  using TypeUtils = TypeUtilities<TypeParam>;

  const TileElementSize blocksize(1, 1);
  Matrix<TypeParam, Device::CPU> matrix({1, 1}, blocksize);
  matrix::test::set(matrix, [](auto&&) { return 1; });

  ExtraBuffers<TypeParam> buffers(matrix(LocalTileIndex{0, 0}), 2, blocksize);

  matrix.read(LocalTileIndex{0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtils::element(4, 0), tile({0, 0}));
  }));

  buffers.get_buffer(0).then(unwrapping([](auto tile) { tile({0, 0}) += TypeUtils::element(1, 0); }));
  buffers.get_buffer(0).then(unwrapping([](auto tile) { tile({0, 0}) *= TypeUtils::element(2, 0); }));

  matrix(LocalTileIndex(0, 0)).then(unwrapping([](auto tile) {
    tile({0, 0}) = TypeUtils::element(13, 0);
  }));

  auto f = matrix.read(LocalTileIndex{0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtils::element(13, 0), tile({0, 0}));
  }));

  buffers.reduce();

  f.get();
}
