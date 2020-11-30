//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/helpers.h"
#include "dlaf/matrix/virtual_workspace.h"

#include <vector>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/hpx_main.hpp>

#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
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
class VirtualWorkspaceLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(VirtualWorkspaceLocalTest, MatrixElementTypes);

TYPED_TEST(VirtualWorkspaceLocalTest, Basic) {
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
  const LocalTileSize at_localsize{dist.localNrTiles().rows() - at_offset.rows(),
                                   dist.localNrTiles().cols() - at_offset.cols()};

  // distribution for local workspaces (useful for computations with trailing matrix At)
  const Distribution dist_col({at_localsize.rows() * nb, 1 * nb}, dist.blockSize());
  const Distribution dist_row({1 * nb, at_localsize.cols() * nb}, dist.blockSize());

  VirtualWorkspace<TypeParam, dlaf::Device::CPU> ws(dist, dist_col, dist_row, at_offset);

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
  for (SizeType index = at_offset.rows(); index < dist.localNrTiles().rows(); ++index) {
    const auto global_row = dist.template globalTileFromLocalTile<Coord::Row>(index);
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

    EXPECT_TRUE(rank_owner_row == rank.row());

    hpx::dataflow(unwrapping([global_row](auto&& tile) {
                    tile({0, 0}) = TypeUtil::element(global_row, 26);
                  }),
                  ws(global_row));
    hpx::dataflow(unwrapping([global_row](auto&& tile) {
                    EXPECT_EQ(TypeUtil::element(global_row, 26), tile({0, 0}));
                  }),
                  ws.read(global_row));
  }

  const auto rank_row_globaltile = std::mem_fn(&Distribution::rankGlobalTile<Coord::Row>);
  const auto rank_col_globaltile = std::mem_fn(&Distribution::rankGlobalTile<Coord::Col>);

  // Access both non-transposed and transposed
  for (SizeType global_row = global_offset.rows(); global_row < dist.nrTiles().rows(); ++global_row) {
    const bool is_on_row = rank_row_globaltile(dist, global_row) == rank.row();
    const bool is_on_col = rank_col_globaltile(dist, global_row) == rank.col();

    if (!is_on_row and !is_on_col)
      continue;

    // check that at the rows are set according to previous phase
    if (is_on_row)
      hpx::dataflow(unwrapping([=](auto&& tile) {
                      EXPECT_EQ(TypeUtil::element(global_row, 26), tile({0, 0}));
                    }),
                    ws.read(global_row));

    // then set every cell in the workspace with the negate of the row index
    hpx::dataflow(unwrapping([=](auto&& tile) { tile({0, 0}) = -global_row; }), ws(global_row));
  }

  const SizeType main_column = 0;

  // Just the main column of ranks sets back the non-tranposed workspace
  if (rank.col() == main_column) {
    for (SizeType index = at_offset.rows(); index < dist.localNrTiles().rows(); ++index) {
      const auto global_row = dist.template globalTileFromLocalTile<Coord::Row>(index);
      const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

      EXPECT_TRUE(rank_owner_row == rank.row());

      hpx::dataflow(unwrapping([global_row](auto&& tile) {
                      tile({0, 0}) = TypeUtil::element(global_row, 26);
                    }),
                    ws(global_row));
    }
  }
  else {  // others just checks that the non-tranposed is negated
    for (SizeType index = at_offset.rows(); index < dist.localNrTiles().rows(); ++index) {
      const auto global_row = dist.template globalTileFromLocalTile<Coord::Row>(index);
      const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(global_row);

      EXPECT_TRUE(rank_owner_row == rank.row());

      hpx::dataflow(unwrapping([](auto&& tile) {
                      EXPECT_LE(std::real(tile({0, 0})), 0);
                    }),
                    ws.read(global_row));
    }
  }

  // TODO test for masked row (e.g. "v0")

  common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  populate(comm::row_wise{}, ws, main_column, serial_comm);

  // Check that the row for everyone has been set correctly, but the other are unaltered
  for (SizeType global_row = global_offset.rows(); global_row < dist.nrTiles().rows(); ++global_row) {
    const bool is_on_row = rank_row_globaltile(dist, global_row) == rank.row();
    const bool is_on_col = rank_col_globaltile(dist, global_row) == rank.col();

    if (!is_on_row and !is_on_col)
      continue;

    // check that at the rows are set according to previous phase
    if (is_on_row)
      hpx::dataflow(unwrapping([global_row](auto&& tile) {
                      EXPECT_EQ(TypeUtil::element(global_row, 26), tile({0, 0}));
                    }),
                    ws.read(global_row));
    else
      hpx::dataflow(unwrapping([=](auto&& tile) {
                      EXPECT_EQ(TypeUtil::element(-global_row, 0), tile({0, 0}));
                    }),
                    ws.read(global_row));
  }

  populate(comm::col_wise{}, ws, serial_comm);

  // check all have the same information
  for (SizeType global_row = global_offset.rows(); global_row < dist.nrTiles().rows(); ++global_row) {
    const bool is_on_row = rank_row_globaltile(dist, global_row) == rank.row();
    const bool is_on_col = rank_col_globaltile(dist, global_row) == rank.col();

    if (!is_on_row and !is_on_col)
      continue;

    hpx::dataflow(unwrapping([global_row](auto&& tile) {
                    EXPECT_EQ(TypeUtil::element(global_row, 26), tile({0, 0}));
                  }),
                  ws.read(global_row));
  }
}
