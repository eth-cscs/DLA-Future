//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/panel.h"

#include <vector>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/hpx_main.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/helpers.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_futures.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;

template <typename Type>
class WorkspaceLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(WorkspaceLocalTest, MatrixElementTypes);

TYPED_TEST(WorkspaceLocalTest, BasicColPanel) {
  using hpx::util::unwrapping;

  Distribution dist(LocalElementSize{2, 1}, TileElementSize{1, 1});

  constexpr auto VALUE = TypeUtilities<TypeParam>::element(26, 10);

  // init order matters!
  Matrix<TypeParam, Device::CPU> matrix(dist);
  matrix::util::set(matrix, [VALUE](auto&&) { return VALUE; });

  Panel<Coord::Col, TypeParam, Device::CPU> ws(dist);

  ws({0, 0}).then(unwrapping([](auto&& tile) {
    tile({0, 0}) = TypeUtilities<TypeParam>::element(13, 26);
  }));
  ws.read({0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtilities<TypeParam>::element(13, 26), tile({0, 0}));
  }));

  for (const auto& index : common::iterate_range2d(matrix.distribution().localNrTiles()))
    matrix.read(index).then(unwrapping([VALUE](auto&& tile) { EXPECT_EQ(VALUE, tile({0, 0})); }));

  ws.set_tile({1, 0}, matrix.read(LocalTileIndex{0, 0}));

  // EXPECT_DEATH(ws.set_tile({1, 0}, matrix.read(LocalTileIndex{0, 0})), "you cannot set it again");

  ws.read({0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtilities<TypeParam>::element(13, 26), tile({0, 0}));
  }));
  ws.read({1, 0}).then(unwrapping([VALUE](auto&& tile) { EXPECT_EQ(VALUE, tile({0, 0})); }));
}

TYPED_TEST(WorkspaceLocalTest, BasicRowPanel) {
  using hpx::util::unwrapping;

  Distribution dist(LocalElementSize{1, 2}, TileElementSize{1, 1});

  constexpr auto VALUE = TypeUtilities<TypeParam>::element(26, 10);

  // init order matters!
  Matrix<TypeParam, Device::CPU> matrix(dist);
  matrix::util::set(matrix, [VALUE](auto&&) { return VALUE; });

  Panel<Coord::Row, TypeParam, Device::CPU> ws(dist);

  ws({0, 0}).then(unwrapping([](auto&& tile) {
    tile({0, 0}) = TypeUtilities<TypeParam>::element(13, 26);
  }));
  ws.read({0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtilities<TypeParam>::element(13, 26), tile({0, 0}));
  }));

  for (const auto& index : common::iterate_range2d(matrix.distribution().localNrTiles()))
    matrix.read(index).then(unwrapping([VALUE](auto&& tile) { EXPECT_EQ(VALUE, tile({0, 0})); }));

  ws.set_tile({0, 1}, matrix.read(LocalTileIndex{0, 0}));

  // EXPECT_DEATH(ws.set_tile({0, 1}, matrix.read(LocalTileIndex{0, 0})), "you cannot set it again");

  ws.read({0, 0}).then(unwrapping([](auto&& tile) {
    EXPECT_EQ(TypeUtilities<TypeParam>::element(13, 26), tile({0, 0}));
  }));
  ws.read({0, 1}).then(unwrapping([VALUE](auto&& tile) { EXPECT_EQ(VALUE, tile({0, 0})); }));
}

TYPED_TEST(WorkspaceLocalTest, PopulateCol_RowWise) {
  using namespace dlaf;
  using hpx::util::unwrapping;

  constexpr comm::IndexT_MPI grid_rows = 2, grid_cols = 3;
  constexpr SizeType n = 5, nb = 1;

  static_assert(NUM_MPI_RANKS == grid_rows * grid_cols, "");

  dlaf::comm::Communicator world(MPI_COMM_WORLD);
  dlaf::comm::CommunicatorGrid grid(world, grid_rows, grid_cols, dlaf::common::Ordering::ColumnMajor);

  const GlobalElementSize matrix_size{n, n};
  const TileElementSize block_size{nb, nb};
  dlaf::Matrix<TypeParam, dlaf::Device::CPU> matrix(matrix_size, block_size, grid);

  const auto& dist = matrix.distribution();

  const GlobalTileSize global_offset{2, 2};

  const LocalTileSize at_offset{
      dist.template nextLocalTileFromGlobalTile<Coord::Row>(global_offset.rows()),
      dist.template nextLocalTileFromGlobalTile<Coord::Col>(global_offset.cols()),
  };

  Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);

  // TODO this part has to be better designed
  // const auto& ws_dist = ws.distribution();

  // EXPECT_EQ(at_localsize, ws_dist.localNrTiles());

  // EXPECT_EQ(at_localsize.rows(), ws_dist.localNrTiles().rows());
  // EXPECT_EQ(at_localsize.cols(), ws_dist.localNrTiles().cols());

  // EXPECT_EQ(at_localsize.rows(), ws_dist.nrTiles().rows());
  // EXPECT_EQ(at_localsize.cols(), ws_dist.nrTiles().cols());

  using TypeUtil = TypeUtilities<TypeParam>;
  const auto rank = dist.rankIndex();

  // Just access workspace non-transposed tiles
  // Each rank sets the tile with the index of the global row
  for (const auto i_w : ws_v) {
    const auto global_row = dist.template globalTileFromLocalTile<Coord::Row>(i_w.row());
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

    EXPECT_TRUE(rank_owner_row == rank.row());

    hpx::dataflow(unwrapping([global_row](auto&& tile) {
                    tile({0, 0}) = TypeUtil::element(global_row, 26);
                  }),
                  ws_v(i_w));
    hpx::dataflow(unwrapping([global_row](auto&& tile) {
                    EXPECT_EQ(TypeUtil::element(global_row, 26), tile({0, 0}));
                  }),
                  ws_v.read(i_w));
  }

  //  const auto rank_row_globaltile = std::mem_fn(&Distribution::rankGlobalTile<Coord::Row>);
  //  const auto rank_col_globaltile = std::mem_fn(&Distribution::rankGlobalTile<Coord::Col>);
  //
  //  // Access both non-transposed and transposed
  //  for (SizeType global_row = global_offset.rows(); global_row < dist.nrTiles().rows(); ++global_row) {
  //    const bool is_on_row = rank_row_globaltile(dist, global_row) == rank.row();
  //    const bool is_on_col = rank_col_globaltile(dist, global_row) == rank.col();
  //
  //    if (!is_on_row and !is_on_col)
  //      continue;
  //
  //    // check that at the rows are set according to previous phase
  //    if (is_on_row)
  //      hpx::dataflow(unwrapping([=](auto&& tile) {
  //                      EXPECT_EQ(TypeUtil::element(global_row, 26), tile({0, 0}));
  //                    }),
  //                    ws.read(global_row));
  //
  //    // then set every cell in the workspace with the negate of the row index
  //    hpx::dataflow(unwrapping([=](auto&& tile) { tile({0, 0}) = -global_row; }), ws(global_row));
  //  }
  //
  const comm::IndexT_MPI main_column = 0;

  // Just the main column of ranks sets back the non-tranposed workspace
  for (const auto& i_w : ws_v) {
    const auto global_row = dist.template globalTileFromLocalTile<Coord::Row>(i_w.row());
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

    EXPECT_TRUE(rank_owner_row == rank.row());

    if (rank.col() == main_column) {
      hpx::dataflow(unwrapping([global_row](auto&& tile) {
                      tile({0, 0}) = TypeUtil::element(-global_row, 26);
                    }),
                    ws_v(i_w));
    }
    else {  // others just checks that the non-tranposed is negated
      hpx::dataflow(unwrapping([=](auto&& tile) {
                      EXPECT_EQ(TypeUtil::element(global_row, 26), tile({0, 0}));
                    }),
                    ws_v.read(i_w));
    }
  }

  // TODO test for masked row (e.g. "v0")
  common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  share_panel(
      comm::row_wise{}, ws_v, [&](auto&&) { return std::make_pair(-1, main_column); }, serial_comm);

  for (const auto& i_w : ws_v) {
    const auto global_row = dist.template globalTileFromLocalTile<Coord::Row>(i_w.row());
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

    EXPECT_TRUE(rank_owner_row == rank.row());

    hpx::dataflow(unwrapping([=](auto&& tile) {
                    EXPECT_EQ(TypeUtil::element(-global_row, 26), tile({0, 0}));
                  }),
                  ws_v.read(i_w));
  }

  Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);
  auto whos_root = transpose(ws_v, ws_h);

  // Check that the row panel has the info (just in locally available ones)
  for (const auto& i_w : ws_h) {
    const auto global_row = dist.template globalTileFromLocalTile<Coord::Col>(i_w.col());
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

    if (rank_owner_row != rank.row())
      continue;

    hpx::dataflow(unwrapping([=](auto&& tile) {
                    EXPECT_EQ(TypeUtil::element(-global_row, 26), tile({0, 0}));
                  }),
                  ws_h.read(i_w));
  }

  share_panel(comm::col_wise{}, ws_h, whos_root, serial_comm);

  // Check that the row panel has the informations
  for (const auto& i_w : ws_h) {
    const auto global_row = dist.template globalTileFromLocalTile<Coord::Col>(i_w.col());

    hpx::dataflow(unwrapping([=](auto&& tile) {
                    EXPECT_EQ(TypeUtil::element(-global_row, 26), tile({0, 0}));
                  }),
                  ws_h.read(i_w));
  }

  // Just checking that nothing changed in the column panel
  for (const auto& i_w : ws_v) {
    const auto global_row = dist.template globalTileFromLocalTile<Coord::Row>(i_w.row());
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

    EXPECT_TRUE(rank_owner_row == rank.row());

    hpx::dataflow(unwrapping([=](auto&& tile) {
                    EXPECT_EQ(TypeUtil::element(-global_row, 26), tile({0, 0}));
                  }),
                  ws_v.read(i_w));
  }
}
