//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>

#include <dlaf/matrix/util_distribution.h>
#include <dlaf/types.h>

#include <gtest/gtest.h>

using namespace dlaf;
using namespace dlaf::util::matrix;
using namespace testing;

struct Parameters {
  // Distribution settings
  SizeType tile_size;
  SizeType tiles_per_block;
  int rank;
  int grid_size;
  int src_rank;
  // Valid indices
  SizeType global_element;
  SizeType global_tile;
  SizeType rank_tile;
  SizeType local_tile;
  SizeType local_tile_next;
  SizeType tile_element;
  // offsets
  SizeType tile_offset;
  SizeType tile_element_offset;
};

TEST(UtilDistribution, IndexConversion) {
  std::vector<Parameters> tests = {
      // {tile_size, tiles_per_block, rank, grid_size, src_rank, global_element, global_tile,
      // rank_tile, local_tile, local_tile_next, tile_element, tile_offset, tile_element_offset}
      {10, 1, 0, 1, 0, 31, 3, 0, 3, 3, 1, 0, 0},    {10, 1, 0, 5, 0, 102, 10, 0, 2, 2, 2, 0, 0},
      {10, 1, 1, 5, 0, 124, 12, 2, -1, 3, 4, 0, 0}, {10, 1, 4, 5, 3, 124, 12, 0, -1, 3, 4, 0, 0},
      {25, 1, 0, 1, 0, 231, 9, 0, 9, 9, 6, 0, 0},   {25, 1, 0, 5, 0, 102, 4, 4, -1, 1, 2, 0, 0},
      {25, 1, 3, 5, 4, 102, 4, 3, 0, 0, 2, 0, 0},   {25, 1, 4, 5, 3, 0, 0, 3, -1, 0, 0, 0, 0},
      {25, 1, 0, 5, 3, 0, 0, 3, -1, 0, 0, 0, 0},    {25, 1, 3, 5, 3, 0, 0, 3, 0, 0, 0, 0, 0},

      {10, 3, 0, 1, 0, 31, 3, 0, 3, 3, 1, 0, 0},    {10, 2, 0, 5, 0, 102, 10, 0, 2, 2, 2, 0, 0},
      {10, 4, 1, 5, 0, 124, 12, 3, -1, 4, 4, 0, 0}, {10, 4, 4, 5, 3, 124, 12, 1, -1, 4, 4, 0, 0},
      {25, 5, 0, 1, 0, 231, 9, 0, 9, 9, 6, 0, 0},   {25, 4, 0, 5, 0, 652, 26, 1, -1, 8, 2, 0, 0},
      {25, 4, 1, 5, 0, 652, 26, 1, 6, 6, 2, 0, 0},  {25, 4, 2, 5, 0, 652, 26, 1, -1, 4, 2, 0, 0},
      {25, 3, 3, 5, 2, 102, 4, 3, 1, 1, 2, 0, 0},   {25, 3, 4, 5, 3, 0, 0, 3, -1, 0, 0, 0, 0},
      {25, 2, 0, 5, 3, 0, 0, 3, -1, 0, 0, 0, 0},    {25, 2, 3, 5, 3, 0, 0, 3, 0, 0, 0, 0, 0},

      {10, 1, 0, 1, 0, 31, 3, 0, 3, 3, 7, 0, 6},    {10, 1, 0, 5, 0, 98, 10, 0, 2, 2, 1, 0, 3},
      {25, 1, 0, 1, 0, 224, 9, 0, 9, 9, 6, 0, 7},   {25, 1, 0, 5, 0, 102, 4, 4, -1, 1, 24, 0, 22},
      {10, 3, 0, 1, 0, 21, 2, 0, 2, 2, 1, 1, 0},    {10, 2, 0, 5, 0, 88, 9, 0, 1, 1, 2, 1, 4},
      {10, 4, 1, 5, 0, 102, 10, 3, -1, 4, 4, 2, 2}, {10, 4, 4, 5, 3, 94, 9, 1, -1, 4, 4, 3, 0},
      {25, 4, 1, 5, 0, 582, 24, 1, 6, 6, 2, 2, 20}, {25, 4, 2, 5, 0, 553, 23, 1, -1, 4, 2, 3, 24},
  };

  for (const auto& test : tests) {
    EXPECT_EQ(test.global_tile,
              tile_from_element(test.global_element, test.tile_size, test.tile_element_offset));
    EXPECT_EQ(test.tile_element,
              tile_element_from_element(test.global_element, test.tile_size, test.tile_element_offset));
    EXPECT_EQ(test.global_element,
              element_from_tile_and_tile_element(test.global_tile, test.tile_element, test.tile_size,
                                                 test.tile_element_offset));
    EXPECT_EQ(test.rank_tile, rank_global_tile(test.global_tile, test.tiles_per_block, test.grid_size,
                                               test.src_rank, test.tile_offset));
    EXPECT_EQ(test.local_tile,
              local_tile_from_global_tile(test.global_tile, test.tiles_per_block, test.grid_size,
                                          test.rank, test.src_rank, test.tile_offset));
    EXPECT_EQ(test.local_tile_next,
              next_local_tile_from_global_tile(test.global_tile, test.tiles_per_block, test.grid_size,
                                               test.rank, test.src_rank, test.tile_offset));
    if (test.local_tile >= 0) {
      EXPECT_EQ(test.global_tile,
                global_tile_from_local_tile(test.local_tile, test.tiles_per_block, test.grid_size,
                                            test.rank, test.src_rank, test.tile_offset));
    }
  }
}
