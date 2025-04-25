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
#include <tuple>
#include <utility>
#include <vector>

#include <dlaf/matrix/distribution.h>

#include <gtest/gtest.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace testing;

struct ParametersConstructor {
  // Distribution settings
  GlobalElementSize size;
  GlobalElementSize block_size;
  TileElementSize tile_size;
  comm::Index2D rank;
  comm::Size2D grid_size;
  comm::Index2D src_rank;
  GlobalElementIndex offset;
  // Derived params
  GlobalTileSize global_tiles;
  LocalTileSize local_tiles;
  GlobalBlockSize global_blocks;
  LocalBlockSize local_blocks;
  LocalElementSize local_size;
};

// clang-format off
const std::vector<ParametersConstructor> tests_constructor = {
    // {size, block_size, tile_size, rank, grid_size, src_rank, offset, global_tiles, local_tiles, global_blocks, local_blocks, local_size}
    {{0, 0}, {13, 17}, {13, 17}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {13, 17}, {13, 17}, {2, 1}, {3, 2}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 128}, {12, 11}, {12, 11}, {1, 0}, {3, 1}, {0, 0}, {0, 0}, {0, 12}, {0, 12}, {0, 12}, {0, 12}, {0, 128}},
    {{25, 0}, {14, 7}, {14, 7}, {0, 1}, {3, 2}, {1, 1}, {0, 0}, {2, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {12, 16}, {4, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {10, 18}, {5, 9}, {2, 1}, {3, 2}, {0, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 128}, {12, 18}, {6, 3}, {1, 0}, {3, 1}, {0, 0}, {0, 0}, {0, 43}, {0, 43}, {0, 8}, {0, 8}, {0, 128}},
    {{1, 1}, {16, 16}, {16, 16}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
    {{1, 32}, {13, 21}, {13, 21}, {2, 1}, {3, 2}, {0, 0}, {0, 0}, {1, 2}, {0, 1}, {1, 2}, {0, 1}, {0, 11}},
    {{13, 16}, {13, 16}, {13, 16}, {5, 7}, {9, 8}, {2, 3}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 0}},
    {{523, 111}, {19, 11}, {19, 11}, {2, 5}, {9, 8}, {2, 3}, {0, 0}, {28, 11}, {4, 2}, {28, 11}, {4, 2}, {67, 12}},
    {{71, 3750}, {64, 128}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {0, 0}, {2, 30}, {0, 5}, {2, 30}, {0, 5}, {0, 550}},
    {{1020, 34}, {16, 32}, {16, 32}, {0, 0}, {1, 6}, {0, 0}, {0, 0}, {64, 2}, {64, 1}, {64, 2}, {64, 1}, {1020, 32}},
    {{1024, 1024}, {32, 32}, {32, 32}, {3, 2}, {6, 4}, {1, 1}, {0, 0}, {32, 32}, {5, 8}, {32, 32}, {5, 8}, {160, 256}},
    {{160, 192}, {32, 32}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {0, 0}, {5, 6}, {2, 2}, {5, 6}, {2, 2}, {64, 64}},
    {{1020, 34}, {16, 32}, {16, 32}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {64, 2}, {64, 2}, {64, 2}, {64, 2}, {1020, 34}},
    {{160, 192}, {32, 32}, {32, 32}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {5, 6}, {5, 6}, {5, 6}, {5, 6}, {160, 192}},
    {{1020, 34}, {16, 32}, {8, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {128, 9}, {128, 9}, {64, 2}, {64, 2}, {1020, 34}},
    {{161, 193}, {32, 32}, {8, 4}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {21, 49}, {21, 49}, {6, 7}, {6, 7}, {161, 193}},

    // offset != {0, 0}
    {{0, 0}, {13, 17}, {13, 17}, {0, 0}, {1, 1}, {0, 0}, {1, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {3, 3}, {3, 3}, {2, 1}, {3, 2}, {1, 1}, {4, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 0}, {13, 17}, {13, 17}, {2, 1}, {3, 2}, {0, 1}, {2, 1}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
    {{0, 128}, {12, 11}, {12, 11}, {1, 0}, {3, 1}, {0, 0}, {2, 3}, {0, 12}, {0, 12}, {0, 12}, {0, 12}, {0, 128}},
    {{25, 0}, {14, 7}, {14, 7}, {0, 1}, {3, 2}, {1, 1}, {3, 3}, {2, 0}, {0, 0}, {2, 0}, {0, 0}, {0, 0}},
    {{1, 1}, {16, 16}, {16, 16}, {0, 0}, {1, 1}, {0, 0}, {17, 17}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}},
    {{1, 32}, {13, 21}, {13, 21}, {2, 1}, {3, 2}, {0, 0}, {1, 1}, {1, 2}, {0, 1}, {1, 2}, {0, 1}, {0, 12}},
    {{1, 32}, {13, 21}, {13, 21}, {2, 1}, {3, 2}, {2, 1}, {1, 1}, {1, 2}, {1, 1}, {1, 2}, {1, 1}, {1, 20}},
    {{10, 15}, {5, 5}, {5, 5}, {1, 1}, {2, 2}, {1, 0}, {3, 7}, {3, 4}, {2, 2}, {3, 4}, {2, 2}, {5, 8}},
    {{13, 16}, {13, 16}, {13, 16}, {4, 5}, {9, 8}, {2, 3}, {32, 32}, {2, 1}, {1, 1}, {2, 1}, {1, 1}, {7, 16}},
    {{13, 16}, {13, 16}, {13, 16}, {5, 5}, {9, 8}, {2, 3}, {32, 32}, {2, 1}, {1, 1}, {2, 1}, {1, 1}, {6, 16}},
    {{13, 16}, {13, 16}, {13, 16}, {5, 7}, {9, 8}, {2, 3}, {32, 32}, {2, 1}, {1, 0}, {2, 1}, {1, 0}, {6, 0}},
    {{523, 111}, {19, 11}, {19, 11}, {2, 5}, {9, 8}, {2, 3}, {10, 10}, {29, 11}, {4, 2}, {29, 11}, {4, 2}, {66, 22}},
    {{71, 3750}, {64, 128}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {1, 1}, {2, 30}, {0, 5}, {2, 30}, {0, 5}, {0, 551}},
    {{71, 3750}, {64, 128}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {448, 0}, {2, 30}, {0, 5}, {2, 30}, {0, 5}, {0, 550}},
    {{71, 3750}, {64, 128}, {64, 128}, {1, 3}, {7, 6}, {3, 4}, {0, 768}, {2, 30}, {0, 5}, {2, 30}, {0, 5}, {0, 550}},
    {{1020, 34}, {16, 32}, {16, 32}, {0, 0}, {1, 6}, {0, 0}, {8, 8}, {65, 2}, {65, 1}, {65, 2}, {65, 1}, {1020, 24}},
    {{1024, 1024}, {32, 32}, {32, 32}, {3, 2}, {6, 4}, {1, 1}, {48, 48}, {33, 33}, {6, 9}, {33, 33}, {6, 9}, {192, 256}},
    {{160, 192}, {32, 32}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {16, 16}, {6, 7}, {2, 2}, {6, 7}, {2, 2}, {48, 48}},
    {{160, 192}, {32, 32}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {24, 8}, {6, 7}, {2, 2}, {6, 7}, {2, 2}, {40, 56}},
    {{160, 192}, {32, 32}, {32, 32}, {1, 1}, {4, 4}, {0, 0}, {24, 8}, {6, 7}, {2, 2}, {6, 7}, {2, 2}, {56, 64}},
    {{160, 192}, {32, 32}, {32, 32}, {0, 0}, {4, 4}, {3, 3}, {24, 8}, {6, 7}, {2, 2}, {6, 7}, {2, 2}, {56, 64}},
    {{71, 3750}, {64, 128}, {4, 16}, {1, 3}, {7, 6}, {3, 4}, {1, 1}, {18, 235}, {0, 35}, {2, 30}, {0, 5}, {0, 551}},
    {{71, 3750}, {64, 128}, {4, 16}, {1, 3}, {7, 6}, {3, 4}, {448, 0}, {18, 235}, {0, 35}, {2, 30}, {0, 5}, {0, 550}},
    {{71, 3750}, {64, 128}, {4, 16}, {1, 3}, {7, 6}, {3, 4}, {0, 768}, {18, 235}, {0, 35}, {2, 30}, {0, 5}, {0, 550}},
    {{1020, 34}, {16, 32}, {4, 8}, {0, 0}, {1, 6}, {0, 0}, {8, 8}, {255, 5}, {255, 3}, {65, 2}, {65, 1}, {1020, 24}},
    {{1024, 1024}, {32, 32}, {8, 32}, {3, 2}, {6, 4}, {1, 1}, {49, 48}, {129, 33}, {24, 9}, {33, 33}, {6, 9}, {192, 256}},
    {{160, 192}, {64, 128}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {16, 16}, {6, 7}, {2, 4}, {3, 2}, {1, 1}, {48, 112}},
    {{160, 192}, {64, 128}, {32, 32}, {0, 0}, {4, 4}, {0, 0}, {24, 8}, {6, 7}, {2, 4}, {3, 2}, {1, 1}, {40, 120}},
    {{160, 192}, {128, 64}, {32, 32}, {1, 1}, {4, 4}, {0, 0}, {24, 8}, {6, 7}, {2, 2}, {2, 4}, {1, 1}, {56, 64}},
    {{160, 192}, {128, 64}, {32, 32}, {0, 0}, {4, 4}, {3, 3}, {24, 8}, {6, 7}, {2, 2}, {2, 4}, {1, 1}, {56, 64}},
    {{36, 54}, {14, 39}, {7, 13}, {0, 1}, {3, 4}, {0, 3}, {11, 38}, {6, 6}, {2, 2}, {4, 3}, {2, 1}, {8, 14}},
    {{1020, 34}, {16, 32}, {16, 32}, {0, 0}, {1, 1}, {0, 0}, {65, 6}, {64, 2}, {64, 2}, {64, 2}, {64, 2}, {1020, 34}},
    {{160, 192}, {32, 32}, {32, 32}, {0, 0}, {1, 1}, {0, 0}, {37, 7}, {6, 7}, {6, 7}, {6, 7}, {6, 7}, {160, 192}},
    {{1020, 34}, {16, 32}, {8, 4}, {0, 0}, {1, 1}, {0, 0}, {3, 23}, {128, 10}, {128, 10}, {64, 2}, {64, 2}, {1020, 34}},
    {{161, 193}, {32, 32}, {8, 4}, {0, 0}, {1, 1}, {0, 0}, {18, 19}, {21, 49}, {21, 49}, {6, 7}, {6, 7}, {161, 193}},
};
// clang-format on

void check_constructor(const Distribution& obj, GlobalElementSize size, GlobalElementSize block_size,
                       TileElementSize tile_size, comm::Index2D rank, comm::Size2D grid_size,
                       comm::Index2D src_rank, GlobalElementIndex offset, GlobalTileSize global_tiles,
                       LocalTileSize local_tiles, GlobalBlockSize global_blocks,
                       LocalBlockSize local_blocks, LocalElementSize local_size) {
  EXPECT_EQ(size, obj.size());
  EXPECT_EQ(block_size, obj.block_size());
  EXPECT_EQ(tile_size, obj.tile_size());
  EXPECT_EQ(rank, obj.rank_index());
  EXPECT_EQ(grid_size, obj.grid_size());
  comm::Index2D expected_source_rank_index{static_cast<int>((src_rank.row() +
                                                             (offset.row() / block_size.rows())) %
                                                            grid_size.rows()),
                                           static_cast<int>((src_rank.col() +
                                                             (offset.col() / block_size.cols())) %
                                                            grid_size.cols())};
  EXPECT_EQ(expected_source_rank_index, obj.source_rank_index());

  GlobalElementIndex expected_offset{offset.row() % block_size.rows(), offset.col() % block_size.cols()};
  EXPECT_EQ(expected_offset, obj.offset());

  EXPECT_EQ(global_tiles, obj.nr_tiles());
  EXPECT_EQ(global_blocks, obj.nr_blocks());
  EXPECT_EQ(local_size, obj.local_size());
  EXPECT_EQ(local_tiles, obj.local_nr_tiles());
  EXPECT_EQ(local_blocks, obj.local_nr_blocks());
}
template <class Test>
void check_constructor(const Distribution& obj, const Test& test) {
  check_constructor(obj, test.size, test.block_size, test.tile_size, test.rank, test.grid_size,
                    test.src_rank, test.offset, test.global_tiles, test.local_tiles, test.global_blocks,
                    test.local_blocks, test.local_size);
}

bool block_is_equal_tile(const ParametersConstructor& test) noexcept {
  return test.tile_size.rows() == test.block_size.rows() &&
         test.tile_size.cols() == test.block_size.cols();
}

TEST(DistributionTest, DefaultConstructor) {
  Distribution obj;

  check_constructor(obj, GlobalElementSize(0, 0), GlobalElementSize(1, 1), TileElementSize(1, 1),
                    comm::Index2D(0, 0), comm::Size2D(1, 1), comm::Index2D(0, 0),
                    GlobalElementIndex(0, 0), GlobalTileSize(0, 0), LocalTileSize(0, 0),
                    GlobalBlockSize(0, 0), LocalBlockSize(0, 0), LocalElementSize(0, 0));
}

TEST(DistributionTest, ConstructorLocal) {
  for (const auto& test : tests_constructor) {
    if (test.grid_size == comm::Size2D(1, 1) && block_is_equal_tile(test)) {
      if (test.offset == GlobalElementIndex{0, 0}) {
        Distribution obj(test.local_size, test.tile_size);
        check_constructor(obj, test);
      }
      Distribution obj(test.local_size, test.tile_size, test.offset);
      check_constructor(obj, test);
    }
  }
}

TEST(DistributionTest, Constructor) {
  for (const auto& test : tests_constructor) {
    GlobalTileIndex tile_offset{test.offset.row() / test.tile_size.rows(),
                                test.offset.col() / test.tile_size.cols()};
    GlobalElementIndex tile_element_offset{test.offset.row() % test.tile_size.rows(),
                                           test.offset.col() % test.tile_size.cols()};

    if (block_is_equal_tile(test)) {
      if (test.offset == GlobalElementIndex{0, 0}) {
        Distribution obj1(test.size, test.tile_size, test.grid_size, test.rank, test.src_rank);
        check_constructor(obj1, test);
      }

      Distribution obj2(test.size, test.tile_size, test.grid_size, test.rank, test.src_rank,
                        test.offset);
      check_constructor(obj2, test);

      // An offset split into tile and element offsets should produce the same distribution
      Distribution obj3(test.size, test.tile_size, test.grid_size, test.rank, test.src_rank, tile_offset,
                        tile_element_offset);
      check_constructor(obj3, test);
    }

    if (test.offset == GlobalElementIndex{0, 0}) {
      Distribution obj1(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                        test.src_rank);
      check_constructor(obj1, test);
    }

    Distribution obj2(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                      test.src_rank, test.offset);
    check_constructor(obj2, test);

    // An offset split into tile and element offsets should produce the same distribution
    Distribution obj3(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                      test.src_rank, tile_offset, tile_element_offset);
    check_constructor(obj3, test);
  }
}

template <Coord rc>
std::tuple<SizeType, SizeType, SizeType> test_tile_size(const ParametersConstructor& test) {
  // Note if only a tile is present only size_first is correct
  const SizeType size_first =
      std::min(test.size.get<rc>(),
               test.tile_size.get<rc>() - test.offset.get<rc>() % test.tile_size.get<rc>());
  const SizeType size_middle = test.tile_size.get<rc>();
  SizeType size_last = (test.offset.get<rc>() + test.size.get<rc>()) % test.tile_size.get<rc>();
  if (size_last == 0)
    size_last = size_middle;
  return {size_first, size_middle, size_last};
}

TEST(DistributionTest, TileSizeOf) {
  for (const auto& test : tests_constructor) {
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);

    const auto [size_row_first, size_row_middle, size_row_last] = test_tile_size<Coord::Row>(test);
    const auto [size_col_first, size_col_middle, size_col_last] = test_tile_size<Coord::Col>(test);

    for (SizeType i = 0; i < obj.nr_tiles().rows(); ++i) {
      SizeType size_row = size_row_middle;
      if (i == 0)
        size_row = size_row_first;
      else if (i == obj.nr_tiles().rows() - 1)
        size_row = size_row_last;
      EXPECT_EQ(size_row, obj.global_tile_size_of<Coord::Row>(i));
      SizeType i_lc = obj.local_tile_from_global_tile<Coord::Row>(i);
      if (i_lc >= 0) {
        EXPECT_EQ(size_row, obj.local_tile_size_of<Coord::Row>(i_lc));
      }
    }

    for (SizeType j = 0; j < obj.nr_tiles().cols(); ++j) {
      SizeType size_col = size_col_middle;
      if (j == 0)
        size_col = size_col_first;
      else if (j == obj.nr_tiles().cols() - 1)
        size_col = size_col_last;
      EXPECT_EQ(size_col, obj.global_tile_size_of<Coord::Col>(j));
      SizeType j_lc = obj.local_tile_from_global_tile<Coord::Col>(j);
      if (j_lc >= 0) {
        EXPECT_EQ(size_col, obj.local_tile_size_of<Coord::Col>(j_lc));
      }
    }
  }
}

TEST(DistributionTest, TileSizeOf2D) {
  for (const auto& test : tests_constructor) {
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);

    const auto [size_row_first, size_row_middle, size_row_last] = test_tile_size<Coord::Row>(test);
    const auto [size_col_first, size_col_middle, size_col_last] = test_tile_size<Coord::Col>(test);

    for (SizeType i = 0; i < obj.nr_tiles().rows(); ++i) {
      SizeType size_row = size_row_middle;
      if (i == 0)
        size_row = size_row_first;
      else if (i == obj.nr_tiles().rows() - 1)
        size_row = size_row_last;

      for (SizeType j = 0; j < obj.nr_tiles().cols(); ++j) {
        SizeType size_col = size_col_middle;
        if (j == 0)
          size_col = size_col_first;
        else if (j == obj.nr_tiles().cols() - 1)
          size_col = size_col_last;
        EXPECT_EQ(TileElementSize(size_row, size_col), obj.tile_size_of(GlobalTileIndex(i, j)));
      }
    }
  }
}

TEST(DistributionTest, ComparisonOperator) {
  for (const auto& test : tests_constructor) {
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);
    std::vector<Distribution> objs_eq;
    objs_eq.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                         test.src_rank, test.offset);

    // If src_rank and offset are shifted accordingly, the resulted distribution is the same.
    comm::Index2D src_rank2{(test.src_rank.row() + test.grid_size.rows() - 1) % test.grid_size.rows(),
                            (test.src_rank.col() + test.grid_size.cols() - 3 % test.grid_size.cols()) %
                                test.grid_size.cols()};

    GlobalElementIndex offset2{test.offset.row() + test.block_size.rows(),
                               test.offset.col() + 3 * test.block_size.cols()};

    objs_eq.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                         src_rank2, offset2);

    GlobalTileIndex tile_offset2{test.block_size.rows() / test.tile_size.rows(),
                                 3 * test.block_size.cols() / test.tile_size.cols()};

    objs_eq.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                         src_rank2, tile_offset2, test.offset);

    for (const auto& obj_eq : objs_eq) {
      EXPECT_TRUE(obj == obj_eq);
      EXPECT_FALSE(obj != obj_eq);
    }

    std::vector<Distribution> objs_ne;
    objs_ne.emplace_back(GlobalElementSize(test.size.rows() + 1, test.size.cols()), test.block_size,
                         test.tile_size, test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(GlobalElementSize(test.size.rows(), test.size.cols() + 1), test.block_size,
                         test.tile_size, test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, TileElementSize(test.block_size.rows() + 1, test.block_size.cols()),
                         test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, TileElementSize(test.block_size.rows(), test.block_size.cols() + 1),
                         test.grid_size, test.rank, test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, test.block_size, test.tile_size,
                         comm::Size2D{test.grid_size.rows() + 1, test.grid_size.cols()}, test.rank,
                         test.src_rank, test.offset);
    objs_ne.emplace_back(test.size, test.block_size, test.tile_size,
                         comm::Size2D{test.grid_size.rows(), test.grid_size.cols() + 1}, test.rank,
                         test.src_rank, test.offset);
    if (test.rank.row() < test.grid_size.rows() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size,
                           comm::Index2D(test.rank.row() + 1, test.rank.col()), test.src_rank,
                           test.offset);
    }
    if (test.rank.col() < test.grid_size.cols() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size,
                           comm::Index2D(test.rank.row(), test.rank.col() + 1), test.src_rank,
                           test.offset);
    }
    if (test.src_rank.row() < test.grid_size.rows() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                           comm::Index2D(test.src_rank.row() + 1, test.src_rank.col()), test.offset);
    }
    if (test.src_rank.col() < test.grid_size.cols() - 1) {
      objs_ne.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                           comm::Index2D(test.src_rank.row(), test.src_rank.col() + 1), test.offset);
    }
    objs_ne.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                         test.src_rank, GlobalElementIndex(test.offset.row() + 1, test.offset.col()));
    objs_ne.emplace_back(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                         test.src_rank, GlobalElementIndex(test.offset.row(), test.offset.col() + 1));

    for (const auto& obj_ne : objs_ne) {
      EXPECT_TRUE(obj != obj_ne);
      EXPECT_FALSE(obj == obj_ne);
    }
  }
}

TEST(DistributionTest, CopyConstructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                      test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);
    EXPECT_EQ(obj0, obj);

    Distribution obj_copy(obj);
    EXPECT_EQ(obj0, obj);
    EXPECT_EQ(obj, obj_copy);
  }
}

TEST(DistributionTest, MoveConstructor) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                      test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);
    EXPECT_EQ(obj0, obj);

    Distribution obj_move(std::move(obj));
    EXPECT_EQ(Distribution(), obj);
    EXPECT_EQ(obj0, obj_move);
  }
}

TEST(DistributionTest, CopyAssignment) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                      test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);
    EXPECT_EQ(obj0, obj);

    Distribution obj_copy;
    obj_copy = obj;
    EXPECT_EQ(obj0, obj);
    EXPECT_EQ(obj, obj_copy);
  }
}

TEST(DistributionTest, MoveAssignment) {
  for (const auto& test : tests_constructor) {
    Distribution obj0(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                      test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);
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
void test_indices(const Distribution& obj, const ParametersIndices& test) {
  SizeType local_element = rc == Coord::Row ? test.local_element[0] : test.local_element[1];
  SizeType local_tile = rc == Coord::Row ? test.local_tile[0] : test.local_tile[1];

  EXPECT_EQ(test.global_element.get<rc>(), obj.global_element_from_global_tile_and_tile_element<rc>(
                                               test.global_tile.get<rc>(), test.tile_element.get<rc>()));
  EXPECT_EQ(test.rank_tile.get<rc>(), obj.rank_global_element<rc>(test.global_element.get<rc>()));
  EXPECT_EQ(test.rank_tile.get<rc>(), obj.rank_global_tile<rc>(test.global_tile.get<rc>()));

  EXPECT_EQ(test.global_tile.get<rc>(),
            obj.global_tile_from_global_element<rc>(test.global_element.get<rc>()));

  EXPECT_EQ(local_tile, obj.local_tile_from_global_element<rc>(test.global_element.get<rc>()));
  EXPECT_EQ(local_tile, obj.local_tile_from_global_tile<rc>(test.global_tile.get<rc>()));

  EXPECT_EQ(test.local_tile_next.get<rc>(),
            obj.next_local_tile_from_global_element<rc>(test.global_element.get<rc>()));
  EXPECT_EQ(test.local_tile_next.get<rc>(),
            obj.next_local_tile_from_global_tile<rc>(test.global_tile.get<rc>()));

  EXPECT_EQ(local_element, obj.local_element_from_global_element<rc>(test.global_element.get<rc>()));

  EXPECT_EQ(test.tile_element.get<rc>(),
            obj.tile_element_from_global_element<rc>(test.global_element.get<rc>()));

  if (local_tile >= 0) {
    EXPECT_EQ(test.global_element.get<rc>(), obj.global_element_from_local_element<rc>(local_element));
    EXPECT_EQ(test.global_element.get<rc>(), obj.global_element_from_local_tile_and_tile_element<rc>(
                                                 local_tile, test.tile_element.get<rc>()));

    EXPECT_EQ(test.global_tile.get<rc>(), obj.global_tile_from_local_tile<rc>(local_tile));

    EXPECT_EQ(local_element, obj.local_element_from_local_tile_and_tile_element<rc>(
                                 local_tile, test.tile_element.get<rc>()));

    EXPECT_EQ(local_tile, obj.local_tile_from_local_element<rc>(local_element));

    EXPECT_EQ(test.tile_element.get<rc>(), obj.tile_element_from_local_element<rc>(local_element));
  }
}

TEST(DistributionTest, IndexConversions) {
  for (const auto& test : tests_indices) {
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);

    test_indices<Coord::Row>(obj, test);
    test_indices<Coord::Col>(obj, test);
  }
}

TEST(DistributionTest, Index2DConversions) {
  for (const auto& test : tests_indices) {
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);

    EXPECT_EQ(test.global_element, obj.global_element_index(test.global_tile, test.tile_element));
    EXPECT_EQ(test.global_tile, obj.global_tile_index(test.global_element));
    EXPECT_EQ(test.rank_tile, obj.rank_global_tile(test.global_tile));
    EXPECT_EQ(test.tile_element, obj.tile_element_index(test.global_element));

    if (test.rank == test.rank_tile) {
      LocalTileIndex local_tile(test.local_tile);
      EXPECT_EQ(test.global_tile, obj.global_tile_index(local_tile));
      EXPECT_EQ(local_tile, obj.local_tile_index(test.global_tile));
    }
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

    EXPECT_EQ(sub_dist.block_size(), dist.block_size());
    EXPECT_EQ(sub_dist.tile_size(), dist.tile_size());
    EXPECT_EQ(sub_dist.rank_index(), dist.rank_index());
    EXPECT_EQ(sub_dist.grid_size(), dist.grid_size());

    EXPECT_LE(sub_dist.local_size().rows(), dist.local_size().rows());
    EXPECT_LE(sub_dist.local_size().cols(), dist.local_size().cols());
    EXPECT_LE(sub_dist.local_nr_tiles().rows(), dist.local_nr_tiles().rows());
    EXPECT_LE(sub_dist.local_nr_tiles().cols(), dist.local_nr_tiles().cols());
    EXPECT_LE(sub_dist.nr_tiles().rows(), dist.nr_tiles().rows());
    EXPECT_LE(sub_dist.nr_tiles().cols(), dist.nr_tiles().cols());

    if (!test.sub_size.isEmpty()) {
      EXPECT_EQ(sub_dist.global_tile_index(test.global_element), test.global_tile);
      EXPECT_EQ(sub_dist.rank_global_tile(sub_dist.global_tile_index(test.global_element)),
                test.rank_tile);

      EXPECT_EQ(
          sub_dist.local_tile_from_global_element<Coord::Row>(test.global_element.get<Coord::Row>()),
          test.local_tile[0]);
      EXPECT_EQ(
          sub_dist.local_tile_from_global_element<Coord::Col>(test.global_element.get<Coord::Col>()),
          test.local_tile[1]);
    }
  }
}

TEST(DistributionInternalTest, SingleTilePerBlockDistribution) {
  for (const auto& test : tests_constructor) {
    Distribution expected_obj(test.size, test.block_size,
                              {test.block_size.rows(), test.block_size.cols()}, test.grid_size,
                              test.rank, test.src_rank, test.offset);
    Distribution obj(test.size, test.block_size, test.tile_size, test.grid_size, test.rank,
                     test.src_rank, test.offset);
    EXPECT_EQ(expected_obj, dlaf::matrix::internal::get_single_tile_per_block_distribution(obj));
  }
}
