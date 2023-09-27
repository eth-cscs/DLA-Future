//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <array>
#include <stdexcept>

#include <dlaf/matrix/distribution.h>

#include <gtest/gtest.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace testing;

TEST(DistributionTest, DistanceToAdjacentTile) {
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 3));

    // Columns
    //
    // Inside a tile
    EXPECT_EQ(2, distr.distanceToAdjacentTile<Coord::Col>(7));
    // At the start of tile
    EXPECT_EQ(3, distr.distanceToAdjacentTile<Coord::Col>(6));
    // At the end of tile
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Col>(5));
    // At the beginning of matrix
    EXPECT_EQ(3, distr.distanceToAdjacentTile<Coord::Col>(0));
    // At the end of matrix
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Col>(9));

    // Rows
    //
    // Inside a tile
    EXPECT_EQ(2, distr.distanceToAdjacentTile<Coord::Row>(1));
    // At the start of tile
    EXPECT_EQ(3, distr.distanceToAdjacentTile<Coord::Row>(3));
    // At the end of tile
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Row>(8));
    // At the beginning of matrix
    EXPECT_EQ(3, distr.distanceToAdjacentTile<Coord::Row>(0));
    // At the end of matrix
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Row>(9));
  }

  // block_size > size && offset != {0, 0}
  {
    matrix::Distribution distr(LocalElementSize(7, 7), TileElementSize(20, 20),
                               GlobalElementIndex(1, 15));

    // Columns
    //
    EXPECT_EQ(5, distr.distanceToAdjacentTile<Coord::Col>(0));
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Col>(4));
    EXPECT_EQ(2, distr.distanceToAdjacentTile<Coord::Col>(5));
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Col>(6));

    // Rows
    //
    EXPECT_EQ(7, distr.distanceToAdjacentTile<Coord::Row>(0));
    EXPECT_EQ(6, distr.distanceToAdjacentTile<Coord::Row>(1));
    EXPECT_EQ(2, distr.distanceToAdjacentTile<Coord::Row>(5));
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Row>(6));
  }

  // offset != {0, 0}
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 3),
                               GlobalElementIndex(1, 5));

    // Columns
    //
    // Inside a tile
    EXPECT_EQ(2, distr.distanceToAdjacentTile<Coord::Col>(5));
    // At the start of tile
    EXPECT_EQ(3, distr.distanceToAdjacentTile<Coord::Col>(4));
    // At the end of tile
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Col>(6));
    // At the beginning of matrix
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Col>(0));
    // At the end of matrix
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Col>(9));

    // Rows
    //
    // Inside a tile
    EXPECT_EQ(2, distr.distanceToAdjacentTile<Coord::Row>(3));
    // At the start of tile
    EXPECT_EQ(3, distr.distanceToAdjacentTile<Coord::Row>(2));
    // At the end of tile
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Row>(4));
    // At the beginning of matrix
    EXPECT_EQ(2, distr.distanceToAdjacentTile<Coord::Row>(0));
    // At the end of matrix
    EXPECT_EQ(1, distr.distanceToAdjacentTile<Coord::Row>(9));
  }
}

TEST(DistributionTest, GlobalTileLinearIndex) {
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 3));

    // Start of matrix
    EXPECT_EQ(0, distr.globalTileLinearIndex(GlobalElementIndex(0, 0)));
    // End of matrix
    EXPECT_EQ(15, distr.globalTileLinearIndex(GlobalElementIndex(9, 9)));
    EXPECT_EQ(5, distr.globalTileLinearIndex(GlobalElementIndex(5, 5)));
    EXPECT_EQ(7, distr.globalTileLinearIndex(GlobalElementIndex(9, 4)));
  }

  // offset!= {0, 0}
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 4),
                               GlobalElementIndex(1, 5));

    // Corners of matrix
    EXPECT_EQ(0, distr.globalTileLinearIndex(GlobalElementIndex(0, 0)));
    EXPECT_EQ(8, distr.globalTileLinearIndex(GlobalElementIndex(0, 9)));
    EXPECT_EQ(3, distr.globalTileLinearIndex(GlobalElementIndex(9, 0)));
    EXPECT_EQ(11, distr.globalTileLinearIndex(GlobalElementIndex(9, 9)));
    // Interior of matrix
    EXPECT_EQ(1, distr.globalTileLinearIndex(GlobalElementIndex(4, 2)));
    EXPECT_EQ(2, distr.globalTileLinearIndex(GlobalElementIndex(5, 2)));
    EXPECT_EQ(5, distr.globalTileLinearIndex(GlobalElementIndex(4, 3)));
    EXPECT_EQ(6, distr.globalTileLinearIndex(GlobalElementIndex(5, 3)));
  }
}

struct ParametersLocalDistanceFromGlobalTile {
  // Distribution settings
  GlobalElementSize size;
  TileElementSize block_size;
  comm::Index2D rank;
  comm::Size2D grid_size;
  comm::Index2D src_rank;
  GlobalElementIndex offset;
  // Valid indices
  GlobalTileIndex global_tile_begin;
  GlobalTileIndex global_tile_end;
  LocalElementSize local_element_distance;
};

struct ParametersSubDistribution {
  // Distribution settings
  GlobalElementSize size;
  TileElementSize block_size;
  comm::Index2D rank;
  comm::Size2D grid_size;
  comm::Index2D src_rank;
  GlobalElementIndex offset;
  // Sub-distribution settings
  GlobalElementIndex sub_origin;
  GlobalElementSize sub_size;
  // Valid indices
  GlobalElementIndex global_element;
  GlobalTileIndex global_tile;
  comm::Index2D rank_tile;
  std::array<SizeType, 2> local_tile;  // can be an invalid LocalTileIndex
};

const std::vector<ParametersSubDistribution> tests_sub_distribution = {
    // {size, block_size, rank, grid_size, src_rank, offset, sub_origin, sub_size,
    // global_element, global_tile, rank_tile, local_tile}
    // clang-format off
    // Empty distribution
    {{0, 0}, {2, 5}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {2, 5}, {0, 0}, {1, 1}, {0, 0}, {4, 8}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},

    // Empty sub-distribution
    {{3, 4}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{3, 4}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {2, 3}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{5, 9}, {3, 2}, {1, 1}, {2, 4}, {0, 2}, {1, 1}, {4, 5}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},

    // Sub-distribution == distribution
    {{3, 4}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {3, 4}, {1, 3}, {0, 1}, {0, 0}, {0, 1}},
    {{5, 9}, {3, 2}, {1, 1}, {2, 4}, {0, 2}, {1, 1}, {0, 0}, {5, 9}, {1, 3}, {0, 2}, {0, 0}, {-1, -1}},
    {{123, 59}, {32, 16}, {3, 3}, {5, 7}, {3, 1}, {1, 1}, {0, 0}, {123, 59}, {30, 30}, {0, 1}, {3, 2}, {0, -1}},

    // Other sub-distributions
    {{3, 4}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {1, 2}, {2, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{3, 4}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {1, 2}, {2, 1}, {1, 0}, {1, 0}, {0, 0}, {1, 0}},
    {{5, 9}, {3, 2}, {1, 1}, {2, 4}, {0, 2}, {1, 1}, {3, 4}, {2, 3}, {0, 0}, {0, 0}, {1, 0}, {0, -1}},
    {{5, 9}, {3, 2}, {1, 1}, {2, 4}, {0, 2}, {1, 1}, {3, 4}, {2, 3}, {1, 2}, {0, 1}, {1, 1}, {0, 0}},
    {{123, 59}, {32, 16}, {3, 3}, {5, 7}, {3, 1}, {1, 1}, {50, 17}, {40, 20}, {20, 10}, {1, 0}, {0, 2}, {-1, -1}},
    // clang-format on
};

TEST(DistributionTest, SubDistribution) {
  for (const auto& test : tests_sub_distribution) {
    Distribution dist(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    const SubDistributionSpec spec{test.sub_origin, test.sub_size};
    Distribution sub_dist(dist, spec);

    EXPECT_EQ(sub_dist.size(), test.sub_size);

    EXPECT_EQ(sub_dist.blockSize(), dist.blockSize());
    EXPECT_EQ(sub_dist.baseTileSize(), dist.baseTileSize());
    EXPECT_EQ(sub_dist.rankIndex(), dist.rankIndex());
    EXPECT_EQ(sub_dist.commGridSize(), dist.commGridSize());

    EXPECT_LE(sub_dist.localSize().rows(), dist.localSize().rows());
    EXPECT_LE(sub_dist.localSize().cols(), dist.localSize().cols());
    EXPECT_LE(sub_dist.localNrTiles().rows(), dist.localNrTiles().rows());
    EXPECT_LE(sub_dist.localNrTiles().cols(), dist.localNrTiles().cols());
    EXPECT_LE(sub_dist.nrTiles().rows(), dist.nrTiles().rows());
    EXPECT_LE(sub_dist.nrTiles().cols(), dist.nrTiles().cols());

    if (!test.sub_size.isEmpty()) {
      EXPECT_EQ(sub_dist.globalTileIndex(test.global_element), test.global_tile);
      EXPECT_EQ(sub_dist.rankGlobalTile(sub_dist.globalTileIndex(test.global_element)), test.rank_tile);

      EXPECT_EQ(sub_dist.localTileFromGlobalElement<Coord::Row>(test.global_element.get<Coord::Row>()),
                test.local_tile[0]);
      EXPECT_EQ(sub_dist.localTileFromGlobalElement<Coord::Col>(test.global_element.get<Coord::Col>()),
                test.local_tile[1]);
    }
  }
}
