//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <stdexcept>
#include "gtest/gtest.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/util_math.h"

using namespace dlaf;
using namespace testing;

std::vector<
    std::tuple<GlobalElementSize, TileElementSize, SizeType, std::size_t, std::size_t, std::size_t>>
    values({{{31, 17}, {7, 11}, 31, 7, 341, 527},      // Scalapack like layout
            {{31, 17}, {7, 11}, 33, 7, 363, 559},      // with padding (ld)
            {{31, 17}, {7, 11}, 47, 11, 517, 799},     // with padding (row)
            {{31, 17}, {7, 11}, 31, 7, 348, 534},      // with padding (col)
            {{29, 41}, {13, 11}, 13, 143, 429, 1667},  // Tile like layout
            {{29, 41}, {13, 11}, 17, 183, 549, 2135},  // with padding (ld)
            {{29, 41}, {13, 11}, 13, 146, 438, 1700},  // with padding (row)
            {{29, 41}, {13, 11}, 13, 143, 436, 1688},  // with padding (col)
            {{29, 41}, {13, 11}, 13, 143, 419, 1637},  // compressed col_offset
            {{0, 0}, {1, 1}, 1, 1, 1, 0}});
std::vector<std::tuple<GlobalElementSize, TileElementSize, SizeType, std::size_t, std::size_t>> wrong_values(
    {
        {{31, 17}, {7, 11}, 30, 7, 341},     // ld, row_offset combo is wrong
        {{31, 17}, {7, 11}, 31, 6, 341},     // row_offset is wrong
        {{31, 17}, {7, 11}, 31, 7, 340},     // col_offset is wrong
        {{29, 41}, {13, 11}, 12, 143, 419},  // ld is wrong
        {{29, 41}, {13, 11}, 13, 142, 419},  // ld, row_offset combo is wrong
        {{29, 41}, {13, 11}, 13, 143, 418},  // col_offset is wrong
        {{-1, 0}, {1, 1}, 1, 1, 1},          // wrong size
        {{0, -1}, {1, 1}, 1, 1, 1},          // wrong size
        {{0, 0}, {0, 1}, 1, 1, 1},           // wrong block_size
        {{0, 0}, {1, 0}, 1, 1, 1},           // wrong block_size
        {{0, 0}, {1, 1}, 0, 1, 1},           // wrong ld
        {{0, 0}, {1, 1}, 1, 0, 1},           // wrong row_offset
        {{0, 0}, {1, 1}, 1, 1, 0}            // wrong col_offset
    });

TEST(LayoutInfoTest, Constructor) {
  using util::size_t::mul;

  for (const auto& v : values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto row_offset = std::get<3>(v);
    auto col_offset = std::get<4>(v);
    auto min_memory = std::get<5>(v);

    matrix::LayoutInfo layout(size, block_size, ld, row_offset, col_offset);

    EXPECT_EQ(size, layout.size());
    EXPECT_EQ(LocalTileSize(util::ceilDiv(size.rows(), block_size.rows()),
                            util::ceilDiv(size.cols(), block_size.cols())),
              layout.nrTiles());
    EXPECT_EQ(block_size, layout.blockSize());
    EXPECT_EQ(ld, layout.ldTile());

    EXPECT_EQ(min_memory, layout.minMemSize());
    for (SizeType j = 0; j < layout.nrTiles().cols(); ++j) {
      for (SizeType i = 0; i < layout.nrTiles().rows(); ++i) {
        std::size_t offset = mul(i, row_offset) + mul(j, col_offset);
        EXPECT_EQ(offset, layout.tileOffset(LocalTileIndex(i, j)));
      }
    }
  }
}

TEST(LayoutInfoTest, ConstructorException) {
  for (const auto& v : wrong_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto row_offset = std::get<3>(v);
    auto col_offset = std::get<4>(v);

    EXPECT_THROW(matrix::LayoutInfo(size, block_size, ld, row_offset, col_offset),
                 std::invalid_argument);
  }
}
