//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <tuple>
#include <utility>
#include <vector>

#include <dlaf/matrix/col_major_layout.h>
#include <dlaf/matrix/index.h>
#include <dlaf/util_math.h>

#include <gtest/gtest.h>

using namespace dlaf;
using namespace testing;

const std::vector<std::tuple<LocalElementSize, TileElementSize, SizeType, SizeType>> values(
    // size, tile_size, ld, min_memory
    {{{31, 17}, {7, 11}, 31, 527},   // Scalapack like layout
     {{31, 17}, {32, 11}, 31, 527},  // only one row of tiles
     {{31, 17}, {7, 11}, 33, 559},   // with padding (ld)
     {{5, 0}, {5, 5}, 1, 0},         // empty matrix with rows > 1 and ld == 1
     {{0, 0}, {1, 1}, 1, 0}});

TEST(ColMajorLayoutTest, Constructor) {
  for (const auto& [size, tile_size, ld, min_memory] : values) {
    matrix::Distribution distribution(size, tile_size);

    matrix::ColMajorLayout layout(distribution, ld);

    EXPECT_EQ(distribution, layout.distribution());
    EXPECT_EQ(size, layout.size());
    EXPECT_EQ(LocalTileSize(util::ceilDiv(size.rows(), tile_size.rows()),
                            util::ceilDiv(size.cols(), tile_size.cols())),
              layout.nr_tiles());
    EXPECT_EQ(tile_size, layout.tile_size());
    EXPECT_EQ(min_memory, layout.min_mem_size());

    for (SizeType j = 0; j < layout.nr_tiles().cols(); ++j) {
      SizeType jb = std::min(tile_size.cols(), size.cols() - j * tile_size.cols());
      for (SizeType i = 0; i < layout.nr_tiles().rows(); ++i) {
        SizeType ib = std::min(tile_size.rows(), size.rows() - i * tile_size.rows());
        LocalTileIndex tile_index(i, j);
        TileElementSize tile_size_of_ij(ib, jb);

        SizeType offset = i * tile_size.rows() + j * tile_size.cols() * ld;
        EXPECT_EQ(offset, layout.tile_offset(tile_index));
        EXPECT_EQ(tile_size_of_ij, layout.tile_size_of(tile_index));
        SizeType min_mem = ib + ld * (jb - 1);
        EXPECT_EQ(min_mem, layout.min_tile_mem_size(tile_index));
      }
    }
  }
}

const matrix::ColMajorLayout layout0({{25, 25}, {5, 5}}, 50);
const std::vector<std::tuple<LocalElementSize, TileElementSize, SizeType, bool>> comp_values({
    // size, tile_size, ld, is_equal
    {{25, 25}, {5, 5}, 50, true},   // Original
    {{23, 25}, {5, 5}, 50, false},  // different size
    {{25, 25}, {6, 5}, 50, false},  // different block_size
    {{25, 25}, {5, 5}, 40, false},  // different ld
});

TEST(LayoutInfoTest, ComparisonOperator) {
  for (const auto& [size, tile_size, ld, is_equal] : comp_values) {
    matrix::ColMajorLayout layout({size, tile_size}, ld);

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
