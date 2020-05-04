//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/distribution.h"

#include <array>
#include <stdexcept>
#include "gtest/gtest.h"

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
  // Derived params
  GlobalTileSize global_tiles;
  LocalTileSize local_tiles;
  LocalElementSize local_size;
};

class TestDistribution : public Distribution {
  TestDistribution(Distribution d) : Distribution(d) {}

public:
  static LocalElementSize testLocalSize(const Distribution& d) {
    return TestDistribution(d).localSize();
  }
};

const std::vector<ParametersConstructor> tests_constructor = {
    // {size, block_size, rank, grid_size, src_rank, global_tiles, local_tiles, local_size}
    {{0, 0}, {13, 17}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {13, 17}, {2, 1}, {3, 2}, {0, 1}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 128}, {12, 11}, {1, 0}, {3, 1}, {0, 0}, {0, 12}, {0, 12}, {0, 128}},
    {{25, 0}, {14, 7}, {0, 1}, {3, 2}, {1, 1}, {2, 0}, {0, 0}, {0, 0}},
    {{1, 1}, {16, 16}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {1, 1}, {1, 1}},
    {{1, 32}, {13, 21}, {2, 1}, {3, 2}, {0, 0}, {1, 2}, {0, 1}, {0, 11}},
    {{13, 16}, {13, 16}, {5, 7}, {9, 8}, {2, 3}, {1, 1}, {0, 0}, {0, 0}},
    {{523, 111}, {19, 11}, {2, 5}, {9, 8}, {2, 3}, {28, 11}, {4, 2}, {67, 12}},
    {{71, 3750}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {2, 30}, {0, 5}, {0, 550}},
    {{1020, 34}, {16, 32}, {0, 0}, {1, 6}, {0, 0}, {64, 2}, {64, 1}, {1020, 32}},
    {{1024, 1024}, {32, 32}, {3, 2}, {6, 4}, {1, 1}, {32, 32}, {5, 8}, {160, 256}},
};

TEST(DistributionTest, DefaultConstructor) {
  Distribution obj;

  EXPECT_EQ(GlobalElementSize(0, 0), obj.size());
  EXPECT_EQ(TileElementSize(1, 1), obj.blockSize());
  EXPECT_EQ(comm::Index2D(0, 0), obj.rankIndex());
  EXPECT_EQ(comm::Size2D(1, 1), obj.commGridSize());
  EXPECT_EQ(comm::Index2D(0, 0), obj.sourceRankIndex());

  EXPECT_EQ(LocalElementSize(0, 0), TestDistribution::testLocalSize(obj));
  EXPECT_EQ(GlobalTileSize(0, 0), obj.nrTiles());
  EXPECT_EQ(LocalTileSize(0, 0), obj.localNrTiles());
}

TEST(DistributionTest, ConstructorLocal) {
  for (const auto& test : tests_constructor) {
    if (test.grid_size == comm::Size2D(1, 1)) {
      Distribution obj(test.local_size, test.block_size);

      EXPECT_EQ(test.size, obj.size());
      EXPECT_EQ(test.block_size, obj.blockSize());
      EXPECT_EQ(test.rank, obj.rankIndex());
      EXPECT_EQ(test.grid_size, obj.commGridSize());
      EXPECT_EQ(test.src_rank, obj.sourceRankIndex());

      EXPECT_EQ(test.global_tiles, obj.nrTiles());
      EXPECT_EQ(test.local_size, TestDistribution::testLocalSize(obj));
      EXPECT_EQ(test.local_tiles, obj.localNrTiles());
    }
  }
}

TEST(DistributionTest, Constructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);

    EXPECT_EQ(test.size, obj.size());
    EXPECT_EQ(test.block_size, obj.blockSize());
    EXPECT_EQ(test.rank, obj.rankIndex());
    EXPECT_EQ(test.grid_size, obj.commGridSize());
    EXPECT_EQ(test.src_rank, obj.sourceRankIndex());

    EXPECT_EQ(test.global_tiles, obj.nrTiles());
    EXPECT_EQ(test.local_size, TestDistribution::testLocalSize(obj));
    EXPECT_EQ(test.local_tiles, obj.localNrTiles());
  }
}

TEST(DistributionTest, ConstructorLocalExceptions) {
  for (const auto& test : tests_constructor) {
    if (test.grid_size == comm::Size2D(1, 1)) {
      EXPECT_THROW(Distribution({-1, test.local_size.cols()}, test.block_size), std::invalid_argument);
      EXPECT_THROW(Distribution({test.local_size.rows(), -1}, test.block_size), std::invalid_argument);
      EXPECT_THROW(Distribution(test.local_size, {0, test.block_size.cols()}), std::invalid_argument);
      EXPECT_THROW(Distribution(test.local_size, {test.block_size.rows(), 0}), std::invalid_argument);
    }
  }
}

TEST(DistributionTest, ConstructorExceptions) {
  for (const auto& test : tests_constructor) {
    EXPECT_THROW(Distribution({-1, test.size.cols()}, test.block_size, test.grid_size, test.rank,
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution({test.size.rows(), -1}, test.block_size, test.grid_size, test.rank,
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, {0, test.block_size.cols()}, test.grid_size, test.rank,
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, {test.block_size.rows(), 0}, test.grid_size, test.rank,
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, {0, test.grid_size.cols()}, test.rank,
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, {test.grid_size.rows(), 0}, test.rank,
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size, {-1, test.rank.col()},
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size,
                              {test.grid_size.rows(), test.rank.col()}, test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size, {test.rank.row(), -1},
                              test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size,
                              {test.rank.row(), test.grid_size.cols()}, test.src_rank),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size, test.rank,
                              {-1, test.src_rank.col()}),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size, test.rank,
                              {test.grid_size.rows(), test.src_rank.col()}),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size, test.rank,
                              {test.src_rank.row(), -1}),
                 std::invalid_argument);
    EXPECT_THROW(Distribution(test.size, test.block_size, test.grid_size, test.rank,
                              {test.src_rank.row(), test.grid_size.cols()}),
                 std::invalid_argument);
  }
}

TEST(DistributionTest, ComparisonOperator) {
  for (const auto& test : tests_constructor) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    Distribution obj_eq(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);

    EXPECT_TRUE(obj == obj_eq);
    EXPECT_FALSE(obj != obj_eq);

    std::vector<Distribution> objs_ne;
    objs_ne.emplace_back(GlobalElementSize(test.size.rows() + 1, test.size.cols()), test.block_size,
                         test.grid_size, test.rank, test.src_rank);
    objs_ne.emplace_back(GlobalElementSize(test.size.rows(), test.size.cols() + 1), test.block_size,
                         test.grid_size, test.rank, test.src_rank);
    objs_ne.emplace_back(test.size, TileElementSize(test.block_size.rows() + 1, test.block_size.cols()),
                         test.grid_size, test.rank, test.src_rank);
    objs_ne.emplace_back(test.size, TileElementSize(test.block_size.rows(), test.block_size.cols() + 1),
                         test.grid_size, test.rank, test.src_rank);
    objs_ne.emplace_back(test.size, test.block_size,
                         comm::Size2D{test.grid_size.rows() + 1, test.grid_size.cols()}, test.rank,
                         test.src_rank);
    objs_ne.emplace_back(test.size, test.block_size,
                         comm::Size2D{test.grid_size.rows(), test.grid_size.cols() + 1}, test.rank,
                         test.src_rank);
    if (test.rank.row() < test.grid_size.rows() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size,
                           comm::Index2D(test.rank.row() + 1, test.rank.col()), test.src_rank);
    }
    if (test.rank.col() < test.grid_size.cols() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size,
                           comm::Index2D(test.rank.row(), test.rank.col() + 1), test.src_rank);
    }
    if (test.src_rank.row() < test.grid_size.rows() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size, test.rank,
                           comm::Index2D(test.src_rank.row() + 1, test.src_rank.col()));
    }
    if (test.src_rank.col() < test.grid_size.cols() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.grid_size, test.rank,
                           comm::Index2D(test.src_rank.row(), test.src_rank.col() + 1));
    }

    for (const auto& obj_ne : objs_ne) {
      EXPECT_TRUE(obj != obj_ne);
      EXPECT_FALSE(obj == obj_ne);
    }
  }
}

TEST(DistributionTest, CopyConstructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    EXPECT_EQ(obj0, obj);

    Distribution obj_copy(obj);
    EXPECT_EQ(obj0, obj);
    EXPECT_EQ(obj, obj_copy);
  }
}

TEST(DistributionTest, MoveConstructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    EXPECT_EQ(obj0, obj);

    Distribution obj_move(std::move(obj));
    EXPECT_EQ(Distribution(), obj);
    EXPECT_EQ(obj0, obj_move);
  }
}

TEST(DistributionTest, CopyAssignment) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    EXPECT_EQ(obj0, obj);

    Distribution obj_copy;
    obj_copy = obj;
    EXPECT_EQ(obj0, obj);
    EXPECT_EQ(obj, obj_copy);
  }
}

TEST(DistributionTest, MoveAssignment) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);
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
  // Valid indices
  GlobalElementIndex global_element;
  GlobalTileIndex global_tile;
  comm::Index2D rank_tile;
  std::array<SizeType, 2> local_tile;  // can be an invalid LocalTileIndex
  LocalTileIndex local_tile_next;
  TileElementIndex tile_element;
};

const std::vector<ParametersIndices> tests_indices = {
    // {size, block_size, rank, grid_size, src_rank, global_element, global_tile,
    // rank_tile, local_tile, local_tile_next, tile_element}
    {{121, 232}, {10, 25}, {0, 0}, {1, 1}, {0, 0}, {31, 231}, {3, 9}, {0, 0}, {3, 9}, {3, 9}, {1, 6}},
    {{133, 111}, {13, 25}, {1, 3}, {4, 5}, {3, 4}, {77, 102}, {5, 4}, {0, 3}, {-1, 0}, {1, 0}, {12, 2}},
    {{13, 130}, {25, 10}, {4, 0}, {5, 5}, {3, 0}, {0, 102}, {0, 10}, {3, 0}, {-1, 2}, {0, 2}, {0, 2}},
    {{134, 300}, {32, 64}, {2, 3}, {3, 5}, {2, 0}, {113, 229}, {3, 3}, {2, 3}, {1, 0}, {1, 0}, {17, 37}},
};

template <Coord rc>
void testIndex(const Distribution& obj, const ParametersIndices& test) {
  SizeType local_tile = rc == Coord::Row ? test.local_tile[0] : test.local_tile[1];

  EXPECT_EQ(test.global_element.get<rc>(),
            obj.globalElementFromGlobalTileAndTileElement<rc>(test.global_tile.get<rc>(),
                                                              test.tile_element.get<rc>()));
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
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);

    testIndex<Coord::Row>(obj, test);
    testIndex<Coord::Col>(obj, test);
  }
}

TEST(DistributionTest, Index2DConversions) {
  for (const auto& test : tests_indices) {
    Distribution obj(test.size, test.block_size, test.grid_size, test.rank, test.src_rank);

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
