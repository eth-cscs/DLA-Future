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

struct ParametersConstructor {
  // Distribution settings
  GlobalElementSize size;
  TileElementSize block_size;
  comm::Index2D rank;
  comm::Size2D grid_size;
  comm::Index2D src_rank;
  GlobalElementIndex offset;
  // Derived params
  GlobalTileSize global_tiles;
  LocalTileSize local_tiles;
  LocalElementSize local_size;
};

const std::vector<ParametersConstructor> tests_constructor = {
    // {size, block_size, rank, grid_size, src_rank, offset, global_tiles, local_tiles, local_size}
    {{0, 0}, {13, 17}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {13, 17}, {2, 1}, {3, 2}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 128}, {12, 11}, {1, 0}, {3, 1}, {0, 0}, {0, 0}, {0, 12}, {0, 12}, {0, 128}},
    {{25, 0}, {14, 7}, {0, 1}, {3, 2}, {1, 1}, {0, 0}, {2, 0}, {0, 0}, {0, 0}},
    {{1, 1}, {16, 16}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}},
    {{1, 32}, {13, 21}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {1, 2}, {0, 1}, {0, 11}},
    {{13, 16}, {13, 16}, {5, 7}, {9, 8}, {2, 3}, {0, 0}, {1, 1}, {0, 0}, {0, 0}},
    {{523, 111}, {19, 11}, {2, 5}, {9, 8}, {2, 3}, {0, 0}, {28, 11}, {4, 2}, {67, 12}},
    {{71, 3750}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {0, 0}, {2, 30}, {0, 5}, {0, 550}},
    {{1020, 34}, {16, 32}, {0, 0}, {1, 6}, {0, 0}, {0, 0}, {64, 2}, {64, 1}, {1020, 32}},
    {{1024, 1024}, {32, 32}, {3, 2}, {6, 4}, {1, 1}, {0, 0}, {32, 32}, {5, 8}, {160, 256}},
    {{160, 192}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {0, 0}, {5, 6}, {2, 2}, {64, 64}},

    // offset != {0, 0}
    {{0, 0}, {13, 17}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {3, 3}, {2, 1}, {3, 2}, {1, 1}, {4, 1}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {13, 17}, {2, 1}, {3, 2}, {0, 1}, {2, 1}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 128}, {12, 11}, {1, 0}, {3, 1}, {0, 0}, {2, 3}, {0, 12}, {0, 12}, {0, 128}},
    {{25, 0}, {14, 7}, {0, 1}, {3, 2}, {1, 1}, {3, 3}, {2, 0}, {0, 0}, {0, 0}},
    {{1, 1}, {16, 16}, {0, 0}, {1, 1}, {0, 0}, {17, 17}, {1, 1}, {1, 1}, {1, 1}},
    {{1, 32}, {13, 21}, {2, 1}, {3, 2}, {0, 0}, {1, 1}, {1, 2}, {0, 1}, {0, 12}},
    {{1, 32}, {13, 21}, {2, 1}, {3, 2}, {2, 1}, {1, 1}, {1, 2}, {1, 1}, {1, 20}},
    {{10, 15}, {5, 5}, {1, 1}, {2, 2}, {1, 0}, {3, 7}, {3, 4}, {2, 2}, {5, 8}},
    {{13, 16}, {13, 16}, {4, 5}, {9, 8}, {2, 3}, {32, 32}, {2, 1}, {1, 1}, {7, 16}},
    {{13, 16}, {13, 16}, {5, 5}, {9, 8}, {2, 3}, {32, 32}, {2, 1}, {1, 1}, {6, 16}},
    {{13, 16}, {13, 16}, {5, 7}, {9, 8}, {2, 3}, {32, 32}, {2, 1}, {1, 0}, {6, 0}},
    {{523, 111}, {19, 11}, {2, 5}, {9, 8}, {2, 3}, {10, 10}, {29, 11}, {4, 2}, {66, 22}},
    {{71, 3750}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {1, 1}, {2, 30}, {0, 5}, {0, 551}},
    {{71, 3750}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {448, 0}, {2, 30}, {0, 5}, {0, 550}},
    {{71, 3750}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {0, 768}, {2, 30}, {0, 5}, {0, 550}},
    {{1020, 34}, {16, 32}, {0, 0}, {1, 6}, {0, 0}, {8, 8}, {65, 2}, {65, 1}, {1020, 24}},
    {{1024, 1024}, {32, 32}, {3, 2}, {6, 4}, {1, 1}, {48, 48}, {33, 33}, {6, 9}, {192, 256}},
    {{160, 192}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {16, 16}, {6, 7}, {2, 2}, {48, 48}},
    {{160, 192}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {24, 8}, {6, 7}, {2, 2}, {40, 56}},
    {{160, 192}, {32, 32}, {1, 1}, {4, 4}, {0, 0}, {24, 8}, {6, 7}, {2, 2}, {56, 64}},
    {{160, 192}, {32, 32}, {0, 0}, {4, 4}, {3, 3}, {24, 8}, {6, 7}, {2, 2}, {56, 64}},
};

TEST(DistributionTest, DefaultConstructor) {
  Distribution obj;

  EXPECT_EQ(GlobalElementSize(0, 0), obj.size());
  EXPECT_EQ(TileElementSize(1, 1), obj.blockSize());
  EXPECT_EQ(comm::Index2D(0, 0), obj.rankIndex());
  EXPECT_EQ(comm::Size2D(1, 1), obj.commGridSize());
  EXPECT_EQ(comm::Index2D(0, 0), obj.sourceRankIndex());

  EXPECT_EQ(LocalElementSize(0, 0), obj.localSize());
  EXPECT_EQ(GlobalTileSize(0, 0), obj.nrTiles());
  EXPECT_EQ(LocalTileSize(0, 0), obj.localNrTiles());
}

TEST(DistributionTest, ConstructorLocal) {
  for (const auto& test : tests_constructor) {
    if (test.grid_size == comm::Size2D(1, 1)) {
      Distribution obj(test.local_size, test.block_size, test.offset);

      EXPECT_EQ(test.size, obj.size());
      EXPECT_EQ(test.block_size, obj.blockSize());
      EXPECT_EQ(test.rank, obj.rankIndex());
      EXPECT_EQ(test.grid_size, obj.commGridSize());
      EXPECT_EQ(test.src_rank, obj.sourceRankIndex());

      EXPECT_EQ(test.global_tiles, obj.nrTiles());
      EXPECT_EQ(test.local_size, obj.localSize());
      EXPECT_EQ(test.local_tiles, obj.localNrTiles());
    }
  }
}

TEST(DistributionTest, Constructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);

    EXPECT_EQ(test.size, obj.size());
    EXPECT_EQ(test.block_size, obj.blockSize());
    EXPECT_EQ(test.rank, obj.rankIndex());
    EXPECT_EQ(test.grid_size, obj.commGridSize());
    decltype(obj.sourceRankIndex()) expected_source_rank_index =
        {static_cast<int>((test.src_rank.get<Coord::Row>() +
                           (test.offset.get<Coord::Row>() / test.block_size.get<Coord::Row>())) %
                          test.grid_size.get<Coord::Row>()),
         static_cast<int>((test.src_rank.get<Coord::Col>() +
                           (test.offset.get<Coord::Col>() / test.block_size.get<Coord::Row>())) %
                          test.grid_size.get<Coord::Col>())};
    EXPECT_EQ(expected_source_rank_index, obj.sourceRankIndex());

    EXPECT_EQ(test.global_tiles, obj.nrTiles());
    EXPECT_EQ(test.local_size, obj.localSize());
    EXPECT_EQ(test.local_tiles, obj.localNrTiles());

    // An offset split into tile and element offsets should produce the same distribution
    Distribution obj_tile_offset(test.size, test.block_size, test.grid_size, test.rank, test.src_rank,
                                 GlobalTileIndex(test.offset.row() / test.block_size.rows(),
                                                 test.offset.col() / test.block_size.cols()),
                                 GlobalElementIndex(test.offset.row() % test.block_size.rows(),
                                                    test.offset.col() % test.block_size.cols()));
    EXPECT_EQ(obj, obj_tile_offset);
  }
}

TEST(DistributionTest, ComparisonOperator) {
  for (const auto& test : tests_constructor) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    Distribution obj_eq(test.size, test.block_size, test.grid_size, test.rank, test.src_rank,
                        test.offset);

    EXPECT_TRUE(obj == obj_eq);
    EXPECT_FALSE(obj != obj_eq);

    std::vector<Distribution> objs_ne;
    objs_ne.emplace_back(GlobalElementSize(test.size.rows() + 1, test.size.cols()), test.block_size,
                         test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(GlobalElementSize(test.size.rows(), test.size.cols() + 1), test.block_size,
                         test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, TileElementSize(test.block_size.rows() + 1, test.block_size.cols()),
                         test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, TileElementSize(test.block_size.rows(), test.block_size.cols() + 1),
                         test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, test.block_size,
                         comm::Size2D{test.grid_size.rows() + 1, test.grid_size.cols()}, test.rank,
                         test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, test.block_size,
                         comm::Size2D{test.grid_size.rows(), test.grid_size.cols() + 1}, test.rank,
                         test.src_rank, test.offset);
    if (test.rank.row() < test.grid_size.rows() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size,
                           comm::Index2D(test.rank.row() + 1, test.rank.col()), test.src_rank,
                           test.offset);
    }
    if (test.rank.col() < test.grid_size.cols() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size,
                           comm::Index2D(test.rank.row(), test.rank.col() + 1), test.src_rank,
                           test.offset);
    }
    if (test.src_rank.row() < test.grid_size.rows() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size, test.rank,
                           comm::Index2D(test.src_rank.row() + 1, test.src_rank.col()), test.offset);
    }
    if (test.src_rank.col() < test.grid_size.cols() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size, test.rank,
                           comm::Index2D(test.src_rank.row(), test.src_rank.col() + 1), test.offset);
    }

    for (const auto& obj_ne : objs_ne) {
      EXPECT_TRUE(obj != obj_ne);
      EXPECT_FALSE(obj == obj_ne);
    }
  }
}

TEST(DistributionTest, CopyConstructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    EXPECT_EQ(obj0, obj);

    Distribution obj_copy(obj);
    EXPECT_EQ(obj0, obj);
    EXPECT_EQ(obj, obj_copy);
  }
}

TEST(DistributionTest, MoveConstructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    EXPECT_EQ(obj0, obj);

    Distribution obj_move(std::move(obj));
    EXPECT_EQ(Distribution(), obj);
    EXPECT_EQ(obj0, obj_move);
  }
}

TEST(DistributionTest, CopyAssignment) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    EXPECT_EQ(obj0, obj);

    Distribution obj_copy;
    obj_copy = obj;
    EXPECT_EQ(obj0, obj);
    EXPECT_EQ(obj, obj_copy);
  }
}

TEST(DistributionTest, MoveAssignment) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);
    EXPECT_EQ(obj0, obj);

    Distribution obj_move;
    obj_move = std::move(obj);
    EXPECT_EQ(Distribution(), obj);
    EXPECT_EQ(obj0, obj_move);
  }
}

struct ParametersIndices {
  // Distribution settings
  GlobalElementSize size;
  TileElementSize block_size;
  comm::Index2D rank;
  comm::Size2D grid_size;
  comm::Index2D src_rank;
  GlobalElementIndex offset;
  // Valid indices
  GlobalElementIndex global_element;
  GlobalTileIndex global_tile;
  comm::Index2D rank_tile;
  std::array<SizeType, 2> local_tile;  // can be an invalid LocalTileIndex
  LocalTileIndex local_tile_next;
  TileElementIndex tile_element;
};

const std::vector<ParametersIndices> tests_indices = {
    // {size, block_size, rank, grid_size, src_rank, offset, global_element, global_tile,
    // rank_tile, local_tile, local_tile_next, tile_element}
    // clang-format off
    {{121, 232}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {31, 231}, {3, 9}, {0, 0}, {3, 9}, {3, 9}, {1, 6}},
    {{133, 111}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {0, 0}, {77, 102}, {5, 4}, {0, 3}, {-1, 0}, {1, 0}, {12, 2}},
    {{13, 130}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {0, 0}, {0, 102}, {0, 10}, {3, 0}, {-1, 2}, {0, 2}, {0, 2}},
    {{134, 300}, {32, 64}, {2, 3}, {3, 5}, {2, 0}, {0, 0}, {113, 229}, {3, 3}, {2, 3}, {1, 0}, {1, 0}, {17, 37}},

    // offset != {0, 0}
    {{121, 232}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {10, 25}, {31, 231}, {3, 9}, {0, 0}, {3, 9}, {3, 9}, {1, 6}},
    {{121, 232}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {30, 230}, {3, 9}, {0, 0}, {3, 9}, {3, 9}, {1, 6}},
    {{121, 232}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {11, 1}, {30, 230}, {3, 9}, {0, 0}, {3, 9}, {3, 9}, {1, 6}},
    {{121, 232}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {1, 30}, {30, 226}, {3, 9}, {0, 0}, {3, 9}, {3, 9}, {1, 6}},
    {{133, 111}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {52, 125}, {77, 102}, {5, 4}, {0, 3}, {-1, 0}, {1, 0}, {12, 2}},
    {{133, 111}, {13, 25}, {1, 3}, {4, 5}, {2, 3}, {13, 25}, {77, 102}, {5, 4}, {0, 3}, {-1, 0}, {1, 0}, {12, 2}},
    {{133, 111}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {13, 25}, {77, 102}, {5, 4}, {1, 4}, {1, -1}, {1, 1}, {12, 2}},
    {{133, 111}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {18, 30}, {72, 97}, {5, 4}, {1, 4}, {1, -1}, {1, 1}, {12, 2}},
    {{13, 130}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {1, 1}, {0, 101}, {0, 10}, {3, 0}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {125, 50}, {0, 102}, {0, 10}, {3, 0}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {4, 0}, {5, 5}, {2, 4}, {150, 60}, {0, 102}, {0, 10}, {3, 0}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {0, 1}, {5, 5}, {3, 0}, {150, 60}, {0, 102}, {0, 10}, {4, 1}, {-1, 2}, {0, 2}, {0, 2}},
    {{13, 130}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {30, 15}, {0, 97}, {0, 10}, {4, 1}, {0, -1}, {0, 2}, {0, 2}},
    {{134, 300}, {32, 64}, {2, 3}, {3, 5}, {2, 0}, {96, 320}, {113, 229}, {3, 3}, {2, 3}, {1, 0}, {1, 0}, {17, 37}},
    {{134, 300}, {32, 64}, {2, 0}, {3, 5}, {2, 0}, {31, 63}, {0, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{134, 300}, {32, 64}, {0, 1}, {3, 5}, {2, 0}, {31, 63}, {1, 1}, {1, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}},
    // clang-format on
};

template <Coord rc>
void testIndex(const Distribution& obj, const ParametersIndices& test) {
  SizeType local_tile = rc == Coord::Row ? test.local_tile[0] : test.local_tile[1];

  EXPECT_EQ(test.global_element.get<rc>(), obj.globalElementFromGlobalTileAndTileElement<rc>(
                                               test.global_tile.get<rc>(), test.tile_element.get<rc>()));
  EXPECT_EQ(test.rank_tile.get<rc>(), obj.rankGlobalElement<rc>(test.global_element.get<rc>()));
  EXPECT_EQ(test.rank_tile.get<rc>(), obj.rankGlobalTile<rc>(test.global_tile.get<rc>()));

  EXPECT_EQ(test.global_tile.get<rc>(),
            obj.globalTileFromGlobalElement<rc>(test.global_element.get<rc>()));

  EXPECT_EQ(local_tile, obj.localTileFromGlobalElement<rc>(test.global_element.get<rc>()));
  EXPECT_EQ(local_tile, obj.localTileFromGlobalTile<rc>(test.global_tile.get<rc>()));

  EXPECT_EQ(test.local_tile_next.get<rc>(),
            obj.nextLocalTileFromGlobalElement<rc>(test.global_element.get<rc>()));
  EXPECT_EQ(test.local_tile_next.get<rc>(),
            obj.nextLocalTileFromGlobalTile<rc>(test.global_tile.get<rc>()));

  EXPECT_EQ(test.tile_element.get<rc>(),
            obj.tileElementFromGlobalElement<rc>(test.global_element.get<rc>()));

  if (local_tile >= 0) {
    EXPECT_EQ(test.global_element.get<rc>(),
              obj.globalElementFromLocalTileAndTileElement<rc>(local_tile, test.tile_element.get<rc>()));
    EXPECT_EQ(test.global_tile.get<rc>(), obj.globalTileFromLocalTile<rc>(local_tile));
  }
}

TEST(DistributionTest, IndexConversions) {
  for (const auto& test : tests_indices) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);

    testIndex<Coord::Row>(obj, test);
    testIndex<Coord::Col>(obj, test);
  }
}

TEST(DistributionTest, Index2DConversions) {
  for (const auto& test : tests_indices) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);

    EXPECT_EQ(test.global_element, obj.globalElementIndex(test.global_tile, test.tile_element));
    EXPECT_EQ(test.global_tile, obj.globalTileIndex(test.global_element));
    EXPECT_EQ(test.rank_tile, obj.rankGlobalTile(test.global_tile));
    EXPECT_EQ(test.tile_element, obj.tileElementIndex(test.global_element));

    if (test.rank == test.rank_tile) {
      LocalTileIndex local_tile(test.local_tile);
      EXPECT_EQ(test.global_tile, obj.globalTileIndex(local_tile));
      EXPECT_EQ(local_tile, obj.localTileIndex(test.global_tile));
    }
  }
}

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

TEST(DistributionTest, TileSizeFromGlobalElement) {
  {
    matrix::Distribution distr(LocalElementSize(8, 7), TileElementSize(3, 2));

    // Rows
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Row>(1));
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Row>(5));
    EXPECT_EQ(2, distr.tileSizeFromGlobalElement<Coord::Row>(7));

    // Cols
    EXPECT_EQ(2, distr.tileSizeFromGlobalElement<Coord::Col>(1));
    EXPECT_EQ(2, distr.tileSizeFromGlobalElement<Coord::Col>(3));
    EXPECT_EQ(1, distr.tileSizeFromGlobalElement<Coord::Col>(6));
  }

  // offset!= {0, 0}
  {
    matrix::Distribution distr(LocalElementSize(8, 7), TileElementSize(3, 4), GlobalElementIndex(2, 5));

    // Rows
    EXPECT_EQ(1, distr.tileSizeFromGlobalElement<Coord::Row>(0));
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Row>(1));
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Row>(3));
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Row>(4));
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Row>(6));
    EXPECT_EQ(1, distr.tileSizeFromGlobalElement<Coord::Row>(7));

    // Cols
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Col>(0));
    EXPECT_EQ(3, distr.tileSizeFromGlobalElement<Coord::Col>(2));
    EXPECT_EQ(4, distr.tileSizeFromGlobalElement<Coord::Col>(3));
    EXPECT_EQ(4, distr.tileSizeFromGlobalElement<Coord::Col>(6));
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

const std::vector<ParametersLocalDistanceFromGlobalTile> tests_local_distance_from_global_tile = {
    // {size, block_size, rank, grid_size, src_rank, offset, global_tile_begin, global_tile_end,
    // local_element_distance}

    // block_size > size
    {{3, 4}, {15, 35}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{3, 4}, {15, 35}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {3, 4}},
    {{3, 4}, {15, 35}, {0, 0}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {3, 4}},
    {{3, 4}, {15, 35}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {0, 0}},
    {{3, 4}, {15, 35}, {2, 1}, {3, 2}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {0, 4}},

    // block_size > size, non-zero offset
    {{3, 4}, {15, 35}, {0, 0}, {1, 1}, {0, 0}, {5, 7}, {0, 0}, {0, 0}, {0, 0}},
    {{3, 4}, {15, 35}, {0, 0}, {1, 1}, {0, 0}, {5, 7}, {0, 0}, {1, 1}, {3, 4}},
    {{3, 4}, {15, 35}, {0, 0}, {3, 2}, {0, 0}, {5, 7}, {0, 0}, {1, 1}, {3, 4}},
    {{3, 4}, {15, 35}, {2, 1}, {3, 2}, {0, 0}, {5, 7}, {0, 0}, {1, 1}, {0, 0}},
    {{3, 4}, {15, 35}, {2, 1}, {3, 2}, {1, 1}, {5, 7}, {0, 0}, {1, 1}, {0, 4}},

    // block_size > size, non-zero offset
    {{3, 4}, {15, 35}, {0, 0}, {1, 1}, {0, 0}, {14, 34}, {0, 0}, {0, 0}, {0, 0}},
    {{3, 4}, {15, 35}, {0, 0}, {1, 1}, {0, 0}, {14, 34}, {0, 0}, {1, 1}, {1, 1}},
    {{3, 4}, {15, 35}, {0, 0}, {1, 1}, {0, 0}, {14, 34}, {0, 0}, {2, 2}, {3, 4}},
    {{3, 4}, {15, 35}, {0, 0}, {3, 2}, {0, 0}, {14, 34}, {0, 0}, {2, 2}, {1, 1}},
    {{3, 4}, {15, 35}, {2, 1}, {3, 2}, {0, 0}, {14, 34}, {0, 0}, {2, 2}, {0, 3}},
    {{3, 4}, {15, 35}, {2, 1}, {3, 2}, {1, 1}, {14, 34}, {0, 0}, {2, 2}, {2, 1}},
    {{3, 4}, {15, 35}, {1, 0}, {3, 2}, {1, 1}, {14, 34}, {0, 0}, {2, 2}, {1, 3}},

    // local only
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {3, 4}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {2, 2}, {4, 4}, {6, 8}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {5, 9}, {15, 35}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {4, 8}, {5, 9}, {3, 3}},

    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {2, 2}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {2, 2}, {0, 0}, {1, 1}, {1, 2}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {2, 2}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {2, 2}, {2, 2}, {4, 4}, {6, 8}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {2, 2}, {0, 0}, {5, 9}, {13, 34}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {2, 2}, {0, 0}, {6, 10}, {15, 35}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {2, 2}, {5, 9}, {6, 10}, {2, 1}},

    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {7, 6}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {7, 6}, {0, 0}, {1, 1}, {2, 2}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {7, 6}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {7, 6}, {2, 2}, {4, 4}, {6, 8}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {7, 6}, {0, 0}, {5, 9}, {14, 34}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {7, 6}, {0, 0}, {6, 10}, {15, 35}},
    {{15, 35}, {3, 4}, {0, 0}, {1, 1}, {0, 0}, {7, 6}, {5, 9}, {6, 10}, {1, 1}},

    // distributed
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {3, 4}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {0, 0}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {0, 0}, {2, 2}, {4, 4}, {3, 4}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {5, 9}, {6, 19}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {0, 0}, {4, 8}, {5, 9}, {0, 3}},

    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {1, 1}, {1, 2}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {2, 2}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {2, 2}, {2, 2}, {4, 4}, {3, 4}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {5, 9}, {4, 18}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {6, 10}, {4, 18}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {2, 2}, {5, 9}, {6, 10}, {0, 0}},

    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {1, 1}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {7, 6}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {7, 6}, {2, 2}, {4, 4}, {0, 4}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {5, 9}, {6, 16}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {6, 10}, {6, 17}},
    {{15, 35}, {3, 4}, {0, 0}, {3, 2}, {0, 0}, {7, 6}, {5, 9}, {6, 10}, {0, 1}},

    // additionally non-zero rank
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {1, 1}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {2, 2}, {4, 4}, {3, 4}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {0, 0}, {5, 9}, {3, 16}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {4, 8}, {5, 9}, {0, 0}},

    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {1, 1}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {2, 2}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {2, 2}, {2, 2}, {4, 4}, {3, 4}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {5, 9}, {3, 16}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {2, 2}, {0, 0}, {6, 10}, {5, 17}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {2, 2}, {5, 9}, {6, 10}, {2, 1}},

    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {1, 1}, {2, 2}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {7, 6}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {7, 6}, {2, 2}, {4, 4}, {3, 4}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {5, 9}, {5, 18}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {7, 6}, {0, 0}, {6, 10}, {5, 18}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {0, 0}, {7, 6}, {5, 9}, {6, 10}, {0, 0}},

    // additionally non-zero source rank
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {0, 4}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {0, 0}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {0, 0}, {2, 2}, {4, 4}, {0, 4}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {0, 0}, {0, 0}, {5, 9}, {6, 19}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {0, 0}, {4, 8}, {5, 9}, {3, 3}},

    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {2, 2}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {2, 2}, {0, 0}, {1, 1}, {0, 2}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {2, 2}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {2, 2}, {2, 2}, {4, 4}, {0, 4}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {2, 2}, {0, 0}, {5, 9}, {6, 18}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {2, 2}, {0, 0}, {6, 10}, {6, 18}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {2, 2}, {5, 9}, {6, 10}, {0, 0}},

    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {7, 6}, {0, 0}, {0, 0}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {7, 6}, {0, 0}, {1, 1}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {7, 6}, {3, 5}, {3, 5}, {0, 0}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {7, 6}, {2, 2}, {4, 4}, {3, 4}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {7, 6}, {0, 0}, {5, 9}, {3, 16}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {7, 6}, {0, 0}, {6, 10}, {4, 17}},
    {{15, 35}, {3, 4}, {2, 1}, {3, 2}, {1, 1}, {7, 6}, {5, 9}, {6, 10}, {1, 1}},
};

TEST(DistributionTest, LocalElementDistanceFromGlobalTile) {
  for (const auto& test : tests_local_distance_from_global_tile) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank, test.offset);

    EXPECT_EQ(test.local_element_distance,
              obj.localElementDistanceFromGlobalTile(test.global_tile_begin, test.global_tile_end));
  }
}

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
    // clang-format off
    {{123, 59}, {32, 16}, {3, 3}, {5, 7}, {3, 1}, {1, 1}, {0, 0}, {123, 59}, {30, 30}, {0, 1}, {3, 2}, {0, -1}},
    // clang-format on
    // Other sub-distributions
    {{3, 4}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {1, 2}, {2, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{3, 4}, {2, 2}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {1, 2}, {2, 1}, {1, 0}, {1, 0}, {0, 0}, {1, 0}},
    {{5, 9}, {3, 2}, {1, 1}, {2, 4}, {0, 2}, {1, 1}, {3, 4}, {2, 3}, {0, 0}, {0, 0}, {1, 0}, {0, -1}},
    {{5, 9}, {3, 2}, {1, 1}, {2, 4}, {0, 2}, {1, 1}, {3, 4}, {2, 3}, {1, 2}, {0, 1}, {1, 1}, {0, 0}},
    // clang-format off
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
