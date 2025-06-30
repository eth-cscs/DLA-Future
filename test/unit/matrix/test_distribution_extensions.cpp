//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <array>
#include <stdexcept>
#include <vector>

#include <dlaf/matrix/distribution.h>

#include <gtest/gtest.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal::distribution;
using namespace testing;

struct ParametersIndices {
  // Distribution settings
  GlobalElementSize size;
  GlobalElementSize block_size;
  TileElementSize tile_size;
  comm::Index2D rank;
  comm::Size2D grid_size;
  comm::Index2D src_rank;
  GlobalElementIndex offset;
  // Valid indices
  GlobalElementIndex global_element;
  GlobalTileIndex global_tile;
  comm::Index2D rank_tile;
  std::array<SizeType, 2> local_element;  // can be an invalid LocalElementIndex
  std::array<SizeType, 2> local_tile;     // can be an invalid LocalTileIndex
  LocalTileIndex local_tile_next;
  TileElementIndex tile_element;
};

const std::vector<ParametersIndices> tests_indices = {
    // {size, block_size, tile_size, rank, grid_size, src_rank, offset, global_element, global_tile,
    // rank_tile, local_element, local_tile, local_tile_next, tile_element}
    // clang-format off
    {{121, 232}, {10, 25}, {5, 5}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {31, 231}, {6, 46}, {0, 0}, {31, 231}, {6, 46}, {6, 46}, {1, 1}},
    {{133, 111}, {13, 25}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {0, 0}, {77, 102}, {5, 4}, {0, 3}, {-1, 2}, {-1, 0}, {1, 0}, {12, 2}},
    {{13, 130}, {25, 10}, {5, 10}, {4, 0}, {5, 5}, {3, 0}, {0, 0}, {0, 102}, {0, 10}, {3, 0}, {-1, 22}, {-1, 2}, {0, 2}, {0, 2}},
    {{134, 300}, {32, 64}, {32, 32}, {2, 3}, {3, 5}, {2, 0}, {0, 0}, {113, 229}, {3, 7}, {2, 3}, {49, 37}, {1, 1}, {1, 1}, {17, 5}},

    // offset != {0, 0}
    {{121, 232}, {10, 25}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {10, 25}, {31, 231}, {3, 9}, {0, 0}, {31, 231}, {3, 9}, {3, 9}, {1, 6}},
    {{121, 232}, {10, 25}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {30, 230}, {3, 9}, {0, 0}, {30, 230}, {3, 9}, {3, 9}, {1, 6}},
    {{121, 232}, {10, 25}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {11, 1}, {30, 230}, {3, 9}, {0, 0}, {30, 230}, {3, 9}, {3, 9}, {1, 6}},
    {{121, 232}, {10, 25}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {1, 30}, {30, 226}, {3, 9}, {0, 0}, {30, 226}, {3, 9}, {3, 9}, {1, 6}},
    {{133, 111}, {13, 25}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {52, 125}, {77, 102}, {5, 4}, {0, 3}, {-1, 2}, {-1, 0}, {1, 0}, {12, 2}},
    {{133, 111}, {13, 25}, {13, 25}, {1, 3}, {4, 5}, {2, 3}, {13, 25}, {77, 102}, {5, 4}, {0, 3}, {-1, 2}, {-1, 0}, {1, 0}, {12, 2}},
    {{133, 111}, {13, 25}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {13, 25}, {77, 102}, {5, 4}, {1, 4}, {25, -1}, {1, -1}, {1, 1}, {12, 2}},
    {{133, 111}, {13, 25}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {18, 30}, {72, 97}, {5, 4}, {1, 4}, {25, -1}, {1, -1}, {1, 1}, {12, 2}},
    {{13, 130}, {25, 10}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {1, 1}, {0, 101}, {0, 10}, {3, 0}, {-1, 21}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {125, 50}, {0, 102}, {0, 10}, {3, 0}, {-1, 22}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {25, 10}, {4, 0}, {5, 5}, {2, 4}, {150, 60}, {0, 102}, {0, 10}, {3, 0}, {-1, 22}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {25, 10}, {0, 1}, {5, 5}, {3, 0}, {150, 60}, {0, 102}, {0, 10}, {4, 1}, {-1, 22}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {30, 15}, {0, 97}, {0, 10}, {4, 1}, {0, -1}, {0, -1}, {0, 2}, {0, 2}},
    {{134, 300}, {32, 64}, {32, 64}, {2, 3}, {3, 5}, {2, 0}, {96, 320}, {113, 229}, {3, 3}, {2, 3}, {49, 37}, {1, 0}, {1, 0}, {17, 37}},
    {{134, 300}, {32, 64}, {32, 64}, {2, 0}, {3, 5}, {2, 0}, {31, 63}, {0, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{134, 300}, {32, 64}, {32, 64}, {0, 1}, {3, 5}, {2, 0}, {31, 63}, {1, 1}, {1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{121, 232}, {10, 25}, {2, 5}, {0, 0}, {1, 1}, {0, 0}, {10, 25}, {31, 231}, {15, 46}, {0, 0}, {31, 231}, {15, 46}, {15, 46}, {1, 1}},
    {{121, 232}, {10, 25}, {2, 5}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {30, 230}, {15, 46}, {0, 0}, {30, 230}, {15, 46}, {15, 46}, {1, 1}},
    {{121, 232}, {10, 25}, {2, 5}, {0, 0}, {1, 1}, {0, 0}, {11, 1}, {30, 230}, {15, 46}, {0, 0}, {30, 230}, {15, 46}, {15, 46}, {1, 1}},
    {{121, 232}, {10, 25}, {2, 5}, {0, 0}, {1, 1}, {0, 0}, {1, 30}, {30, 226}, {15, 45}, {0, 0}, {30, 226}, {15, 45}, {15, 45}, {1, 1}},
    {{13, 130}, {25, 10}, {5, 5}, {4, 0}, {5, 5}, {3, 0}, {1, 1}, {0, 101}, {0, 20}, {3, 0}, {-1, 21}, {-1, 4}, {0, 4}, {0, 2}},
    {{13, 130}, {25, 10}, {5, 5}, {4, 0}, {5, 5}, {3, 0}, {125, 50}, {0, 102}, {0, 20}, {3, 0}, {-1, 22}, {-1, 4}, {0, 4}, {0, 2}},
    {{13, 130}, {25, 10}, {5, 5}, {4, 0}, {5, 5}, {2, 4}, {150, 60}, {0, 102}, {0, 20}, {3, 0}, {-1, 22}, {-1, 4}, {0, 4}, {0, 2}},
    {{13, 130}, {25, 10}, {5, 5}, {0, 1}, {5, 5}, {3, 0}, {150, 60}, {0, 102}, {0, 20}, {4, 1}, {-1, 22}, {-1, 4}, {0, 4}, {0, 2}},
    {{13, 130}, {25, 10}, {5, 5}, {4, 0}, {5, 5}, {3, 0}, {30, 15}, {0, 97}, {0, 19}, {4, 1}, {0, -1}, {0, -1}, {0, 4}, {0, 2}},
    {{134, 300}, {32, 64}, {16, 8}, {2, 3}, {3, 5}, {2, 0}, {96, 320}, {113, 229}, {7, 28}, {2, 3}, {49, 37}, {3, 4}, {3, 4}, {1, 5}},
    {{134, 300}, {32, 64}, {16, 8}, {2, 0}, {3, 5}, {2, 0}, {31, 63}, {0, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{134, 300}, {32, 64}, {16, 8}, {0, 1}, {3, 5}, {2, 0}, {31, 63}, {1, 1}, {1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    // clang-format on
};

template <Coord rc>
void test_indices_with_rank(const Distribution& obj, const ParametersIndices& test) {
  SizeType local_element = rc == Coord::Row ? test.local_element[0] : test.local_element[1];
  SizeType local_tile = rc == Coord::Row ? test.local_tile[0] : test.local_tile[1];

  if (local_tile >= 0) {
    EXPECT_EQ(local_tile, local_tile_from_global_tile_any_rank<rc>(obj, test.global_tile.get<rc>()));

    EXPECT_EQ(test.global_element.get<rc>(),
              global_element_from_local_element_on_rank<rc>(obj, test.rank.get<rc>(), local_element));

    EXPECT_EQ(test.global_tile.get<rc>(),
              global_tile_from_local_tile_on_rank<rc>(obj, test.rank.get<rc>(), local_tile));

    EXPECT_EQ(local_tile,
              local_tile_from_local_element_on_rank<rc>(obj, test.rank.get<rc>(), local_element));

    EXPECT_EQ(test.tile_element.get<rc>(),
              tile_element_from_local_element_on_rank<rc>(obj, test.rank.get<rc>(), local_element));
  }
}

TEST(DistributionExtensionsTest, IndexConversionWithRank) {
  for (const auto& test : tests_indices) {
    for (int rank = 0; rank < test.grid_size.rows(); ++rank) {
      Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size,
                       {rank, test.rank.col()}, test.src_rank, test.offset);

      test_indices_with_rank<Coord::Row>(obj, test);
    }
    for (int rank = 0; rank < test.grid_size.cols(); ++rank) {
      Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size,
                       {test.rank.row(), rank}, test.src_rank, test.offset);

      test_indices_with_rank<Coord::Col>(obj, test);
    }
  }
}

TEST(DistributionExtensionsTest, DistanceToAdjacentTile) {
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 3));

    // Columns
    //
    // Inside a tile
    EXPECT_EQ(2, distance_to_adjacent_tile<Coord::Col>(distr, 7));
    // At the start of tile
    EXPECT_EQ(3, distance_to_adjacent_tile<Coord::Col>(distr, 6));
    // At the end of tile
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Col>(distr, 5));
    // At the beginning of matrix
    EXPECT_EQ(3, distance_to_adjacent_tile<Coord::Col>(distr, 0));
    // At the end of matrix
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Col>(distr, 9));

    // Rows
    //
    // Inside a tile
    EXPECT_EQ(2, distance_to_adjacent_tile<Coord::Row>(distr, 1));
    // At the start of tile
    EXPECT_EQ(3, distance_to_adjacent_tile<Coord::Row>(distr, 3));
    // At the end of tile
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Row>(distr, 8));
    // At the beginning of matrix
    EXPECT_EQ(3, distance_to_adjacent_tile<Coord::Row>(distr, 0));
    // At the end of matrix
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Row>(distr, 9));
  }

  // block_size > size && offset != {0, 0}
  {
    matrix::Distribution distr(LocalElementSize(7, 7), TileElementSize(20, 20),
                               GlobalElementIndex(1, 15));

    // Columns
    //
    EXPECT_EQ(5, distance_to_adjacent_tile<Coord::Col>(distr, 0));
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Col>(distr, 4));
    EXPECT_EQ(2, distance_to_adjacent_tile<Coord::Col>(distr, 5));
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Col>(distr, 6));

    // Rows
    //
    EXPECT_EQ(7, distance_to_adjacent_tile<Coord::Row>(distr, 0));
    EXPECT_EQ(6, distance_to_adjacent_tile<Coord::Row>(distr, 1));
    EXPECT_EQ(2, distance_to_adjacent_tile<Coord::Row>(distr, 5));
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Row>(distr, 6));
  }

  // offset != {0, 0}
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 3),
                               GlobalElementIndex(1, 5));

    // Columns
    //
    // Inside a tile
    EXPECT_EQ(2, distance_to_adjacent_tile<Coord::Col>(distr, 5));
    // At the start of tile
    EXPECT_EQ(3, distance_to_adjacent_tile<Coord::Col>(distr, 4));
    // At the end of tile
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Col>(distr, 6));
    // At the beginning of matrix
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Col>(distr, 0));
    // At the end of matrix
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Col>(distr, 9));

    // Rows
    //
    // Inside a tile
    EXPECT_EQ(2, distance_to_adjacent_tile<Coord::Row>(distr, 3));
    // At the start of tile
    EXPECT_EQ(3, distance_to_adjacent_tile<Coord::Row>(distr, 2));
    // At the end of tile
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Row>(distr, 4));
    // At the beginning of matrix
    EXPECT_EQ(2, distance_to_adjacent_tile<Coord::Row>(distr, 0));
    // At the end of matrix
    EXPECT_EQ(1, distance_to_adjacent_tile<Coord::Row>(distr, 9));
  }
}

TEST(DistributionExtensionsTest, GlobalTileLinearIndex) {
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 3));

    // Start of matrix
    EXPECT_EQ(0, global_tile_linear_index(distr, GlobalElementIndex(0, 0)));
    // End of matrix
    EXPECT_EQ(15, global_tile_linear_index(distr, GlobalElementIndex(9, 9)));
    EXPECT_EQ(5, global_tile_linear_index(distr, GlobalElementIndex(5, 5)));
    EXPECT_EQ(7, global_tile_linear_index(distr, GlobalElementIndex(9, 4)));
  }

  // offset!= {0, 0}
  {
    matrix::Distribution distr(LocalElementSize(10, 10), TileElementSize(3, 4),
                               GlobalElementIndex(1, 5));

    // Corners of matrix
    EXPECT_EQ(0, global_tile_linear_index(distr, GlobalElementIndex(0, 0)));
    EXPECT_EQ(8, global_tile_linear_index(distr, GlobalElementIndex(0, 9)));
    EXPECT_EQ(3, global_tile_linear_index(distr, GlobalElementIndex(9, 0)));
    EXPECT_EQ(11, global_tile_linear_index(distr, GlobalElementIndex(9, 9)));
    // Interior of matrix
    EXPECT_EQ(1, global_tile_linear_index(distr, GlobalElementIndex(4, 2)));
    EXPECT_EQ(2, global_tile_linear_index(distr, GlobalElementIndex(5, 2)));
    EXPECT_EQ(5, global_tile_linear_index(distr, GlobalElementIndex(4, 3)));
    EXPECT_EQ(6, global_tile_linear_index(distr, GlobalElementIndex(5, 3)));
  }
}
