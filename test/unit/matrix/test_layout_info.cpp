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

const std::vector<
    std::tuple<LocalElementSize, TileElementSize, SizeType, std::size_t, std::size_t, std::size_t>>
    values({{{31, 17}, {7, 11}, 31, 7, 341, 527},      // Scalapack like layout
            {{31, 17}, {32, 11}, 31, 31, 341, 527},    // only one row of tiles
            {{31, 17}, {7, 11}, 33, 7, 363, 559},      // with padding (ld)
            {{31, 17}, {7, 11}, 47, 11, 517, 799},     // with padding (row)
            {{31, 17}, {7, 11}, 31, 7, 348, 534},      // with padding (col)
            {{29, 41}, {13, 11}, 13, 143, 429, 1667},  // Tile like layout
            {{29, 41}, {13, 11}, 17, 183, 549, 2135},  // with padding (ld)
            {{29, 41}, {13, 11}, 13, 146, 438, 1700},  // with padding (row)
            {{29, 41}, {13, 11}, 13, 143, 436, 1688},  // with padding (col)
            {{29, 41}, {13, 11}, 13, 143, 419, 1637},  // compressed col_offset
            {{0, 0}, {1, 1}, 1, 1, 1, 0}});

const std::vector<std::tuple<LocalElementSize, TileElementSize, SizeType, std::size_t, std::size_t>>
    wrong_values({
        {{31, 17}, {7, 11}, 30, 7, 341},     // ld, row_offset combo is wrong
        {{31, 17}, {32, 11}, 30, 7, 341},    // ld is wrong
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
  using util::size_t::sum;

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
      SizeType jb = std::min(block_size.cols(), size.cols() - j * block_size.cols());
      for (SizeType i = 0; i < layout.nrTiles().rows(); ++i) {
        SizeType ib = std::min(block_size.rows(), size.rows() - i * block_size.rows());
        LocalTileIndex tile_index(i, j);
        TileElementSize tile_size(ib, jb);

        std::size_t offset = mul(i, row_offset) + mul(j, col_offset);
        EXPECT_EQ(offset, layout.tileOffset(tile_index));
        EXPECT_EQ(tile_size, layout.tileSize(tile_index));
        std::size_t min_mem = sum(ib, mul(ld, jb - 1));
        EXPECT_EQ(min_mem, layout.minTileMemSize(tile_index));
        EXPECT_EQ(min_mem, layout.minTileMemSize(tile_size));
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

const std::vector<std::tuple<LocalElementSize, TileElementSize, SizeType, std::size_t, std::size_t, bool>>
    comp_values({
        {{25, 25}, {5, 5}, 50, 8, 1000, true},   // Original
        {{23, 25}, {5, 5}, 50, 8, 1000, false},  // different size
        {{25, 25}, {6, 5}, 50, 8, 1000, false},  // different block_size
        {{25, 25}, {5, 5}, 40, 8, 1000, false},  // different ld
        {{25, 25}, {5, 5}, 50, 6, 1000, false},  // different row_offset
        {{25, 25}, {5, 5}, 50, 8, 900, false},   // different col_offset
    });

TEST(LayoutInfoTest, ComparisonOperator) {
  matrix::LayoutInfo layout0({25, 25}, {5, 5}, 50, 8, 1000);

  for (const auto& v : comp_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto row_offset = std::get<3>(v);
    auto col_offset = std::get<4>(v);
    auto is_equal = std::get<5>(v);

    matrix::LayoutInfo layout(size, block_size, ld, row_offset, col_offset);

    if (is_equal) {
      EXPECT_TRUE(layout0 == layout);
      EXPECT_FALSE(layout0 != layout);
    }
    else {
      EXPECT_FALSE(layout0 == layout);
      EXPECT_TRUE(layout0 != layout);
    }
  }
}

const std::vector<
    std::tuple<LocalElementSize, TileElementSize, SizeType, std::size_t, std::size_t, std::size_t>>
    col_major_values({
        {{31, 17}, {7, 11}, 31, 7, 341, 527},     // packed ld
        {{31, 17}, {7, 11}, 33, 7, 363, 559},     // padded ld
        {{29, 41}, {13, 11}, 29, 13, 319, 1189},  // packed ld
        {{29, 41}, {13, 11}, 35, 13, 385, 1429},  // padded ld
        {{1, 0}, {1, 1}, 1, 1, 1, 0},             // empty matrix
        {{0, 1}, {1, 1}, 1, 1, 1, 0},             // empty matrix
        {{0, 0}, {1, 1}, 1, 1, 1, 0},             // empty matrix
    });

TEST(LayoutInfoTest, ColMajorLayout) {
  for (const auto& v : col_major_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto row_offset = std::get<3>(v);
    auto col_offset = std::get<4>(v);
    auto min_memory = std::get<5>(v);

    matrix::LayoutInfo exp_layout(size, block_size, ld, row_offset, col_offset);
    matrix::LayoutInfo layout = colMajorLayout(size, block_size, ld);

    EXPECT_EQ(exp_layout, layout);
    EXPECT_EQ(min_memory, layout.minMemSize());
  }
}

const std::vector<std::tuple<LocalElementSize, TileElementSize, SizeType, SizeType, std::size_t,
                             std::size_t, std::size_t, bool>>
    tile_values({
        {{31, 17}, {7, 11}, 7, 5, 77, 385, 731, true},       // basic tile layout
        {{31, 17}, {7, 11}, 11, 5, 121, 605, 1147, false},   // padded ld
        {{31, 17}, {7, 11}, 7, 7, 77, 539, 885, false},      // padded ld
        {{29, 41}, {13, 11}, 13, 3, 143, 429, 1667, true},   // basic tile layout
        {{29, 41}, {13, 11}, 17, 3, 187, 561, 2179, false},  // padded ld
        {{29, 41}, {13, 11}, 13, 4, 143, 572, 2096, false},  // padded tiles_per_col
        {{1, 0}, {1, 1}, 1, 0, 1, 1, 0, true},               // empty matrix
        {{0, 1}, {1, 1}, 1, 0, 1, 1, 0, true},               // empty matrix
        {{0, 0}, {1, 1}, 1, 0, 1, 1, 0, true},               // empty matrix
    });

TEST(LayoutInfoTest, TileLayout) {
  for (const auto& v : tile_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto tiles_per_col = std::get<3>(v);
    auto row_offset = std::get<4>(v);
    auto col_offset = std::get<5>(v);
    auto min_memory = std::get<6>(v);
    auto is_basic = std::get<7>(v);

    matrix::LayoutInfo exp_layout(size, block_size, ld, row_offset, col_offset);
    if (is_basic) {
      matrix::LayoutInfo layout_basic = tileLayout(size, block_size);
      EXPECT_EQ(exp_layout, layout_basic);
      EXPECT_EQ(min_memory, layout_basic.minMemSize());
    }
    matrix::LayoutInfo layout = tileLayout(size, block_size, ld, tiles_per_col);
    EXPECT_EQ(exp_layout, layout);
    EXPECT_EQ(min_memory, layout.minMemSize());
  }
}
