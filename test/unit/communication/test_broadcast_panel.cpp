//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/broadcast_panel.h"

#include <gtest/gtest.h>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct PanelBcastTest : public ::testing::Test {
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(PanelBcastTest, MatrixElementTypes);

struct config_t {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalTileIndex offset;
};

std::vector<config_t> test_params{
    {{0, 0}, {3, 3}, {0, 0}},  // empty matrix
    {{26, 13}, {3, 3}, {1, 2}},
};

template <class TypeParam, Coord panel_axis>
void testBroadcast(comm::Executor& executor_mpi, const config_t& cfg, comm::CommunicatorGrid comm_grid) {
  using TypeUtil = TypeUtilities<TypeParam>;
  using hpx::util::unwrapping;

  constexpr Coord coord1D = orthogonal(panel_axis);

  Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();

  matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.get(coord1D), 26); });

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  // select the last available rank as root rank, i.e. it owns the panel to be broadcasted
  const comm::IndexT_MPI root = std::max(0, comm_grid.size().get(panel_axis) - 1);
  const auto rank = dist.rankIndex().get(panel_axis);

  // set all panels
  for (const auto i_w : panel.iteratorLocal())
    hpx::dataflow(unwrapping(
                      [rank](auto&& tile) { matrix::test::set(tile, TypeUtil::element(rank, 26)); }),
                  panel(i_w));

  // check that all panels have been set
  for (const auto i_w : panel.iteratorLocal())
    CHECK_TILE_EQ(TypeUtil::element(rank, 26), panel.read(i_w).get());

  // test it!
  constexpr Coord comm_dir = orthogonal(panel_axis);
  common::Pipeline<comm::Communicator> mpi_task_chain(comm_grid.subCommunicator(comm_dir));

  broadcast(executor_mpi, root, panel, mpi_task_chain);

  // check all panel are equal on all ranks
  for (const auto i_w : panel.iteratorLocal())
    CHECK_TILE_EQ(TypeUtil::element(root, 26), panel.read(i_w).get());
}

TYPED_TEST(PanelBcastTest, BroadcastCol) {
  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testBroadcast<TypeParam, Coord::Col>(executor_mpi, cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastRow) {
  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testBroadcast<TypeParam, Coord::Row>(executor_mpi, cfg, comm_grid);
}

std::vector<config_t> test_params_bcast_transpose{
    {{0, 0}, {1, 1}, {0, 0}},  // empty matrix
    {{9, 9}, {3, 3}, {3, 3}},  // empty panel (due to offset)
    {{10, 10}, {3, 3}, {1, 1}},
};

template <class TypeParam, Coord PANEL_SRC_AXIS>
void testBrodcastTranspose(comm::Executor& executor_mpi, const config_t& cfg,
                           comm::CommunicatorGrid comm_grid) {
  using TypeUtil = TypeUtilities<TypeParam>;
  using hpx::util::unwrapping;

  const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});
  const auto rank = dist.rankIndex().get(PANEL_SRC_AXIS);

  // It is important to keep the order of initialization to avoid deadlocks!
  constexpr Coord PANEL_DST_AXIS = orthogonal(PANEL_SRC_AXIS);
  Panel<PANEL_SRC_AXIS, TypeParam, dlaf::Device::CPU> panel_src(dist, cfg.offset);
  Panel<PANEL_DST_AXIS, TypeParam, dlaf::Device::CPU> panel_dst(dist, cfg.offset);

  for (const auto i_w : panel_src.iteratorLocal())
    hpx::dataflow(unwrapping(
                      [rank](auto&& tile) { matrix::test::set(tile, TypeUtil::element(rank, 26)); }),
                  panel_src(i_w));

  // test it!
  common::Pipeline<comm::Communicator> row_task_chain(comm_grid.rowCommunicator());
  common::Pipeline<comm::Communicator> col_task_chain(comm_grid.colCommunicator());

  // select a "random" source rank which will be the source for the data
  const comm::IndexT_MPI owner = comm_grid.size().get(PANEL_SRC_AXIS) / 2;

  broadcast(executor_mpi, owner, panel_src, panel_dst, row_task_chain, col_task_chain);

  // Note:
  // all source panels will have access to the same data available on the root rank,
  // while the destination panels will have access to the corresponding "transposed" tile, except
  // for the last global tile in the range.
  for (const auto idx : panel_src.iteratorLocal()) {
    constexpr auto CT = decltype(panel_src)::CoordType;
    const auto i = dist.template globalTileFromLocalTile<CT>(idx.get(CT));
    if (i == panel_src.rangeEnd() - 1) continue;
    CHECK_TILE_EQ(TypeUtil::element(owner, 26), panel_src.read(idx).get());
  }
}

TYPED_TEST(PanelBcastTest, BroadcastCol2Row) {
  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids())
    for (const auto& cfg : test_params_bcast_transpose)
      testBrodcastTranspose<TypeParam, Coord::Col>(executor_mpi, cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastRow2Col) {
  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids())
    for (const auto& cfg : test_params_bcast_transpose)
      testBrodcastTranspose<TypeParam, Coord::Row>(executor_mpi, cfg, comm_grid);
}
