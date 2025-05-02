//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>

#include <dlaf/common/range2d.h>
#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/panel.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;

using pika::execution::experimental::start_detached;
using pika::execution::experimental::then;
using pika::this_thread::experimental::sync_wait;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct PanelBcastTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(PanelBcastTest, MatrixElementTypes);

struct ParamsBcast {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalTileIndex offset;
};

std::vector<ParamsBcast> test_params{
    {{0, 0}, {3, 3}, {0, 0}},  // empty matrix
    {{26, 13}, {3, 3}, {1, 2}},
};

template <class TypeParam, Coord panel_axis, StoreTransposed Storage>
void testBroadcast(const ParamsBcast& cfg, comm::CommunicatorGrid& comm_grid) {
  using TypeUtil = TypeUtilities<TypeParam>;

  const matrix::Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<panel_axis, TypeParam, dlaf::Device::CPU, Storage> panel(dist, cfg.offset);

  // select the last available rank as root rank, i.e. it owns the panel to be broadcasted
  const comm::IndexT_MPI root = std::max(0, comm_grid.size().get(panel_axis) - 1);
  const auto rank = dist.rankIndex().get(panel_axis);

  // set all panels
  for (const auto i_w : panel.iteratorLocal()) {
    start_detached(panel.readwrite(i_w) |
                   then([rank](auto&& tile) { matrix::test::set(tile, TypeUtil::element(rank, 26)); }));
  }

  // check that all panels have been set
  for (const auto i_w : panel.iteratorLocal())
    CHECK_TILE_EQ(TypeUtil::element(rank, 26), sync_wait(panel.read(i_w)).get());

  // test it!
  constexpr Coord comm_dir = orthogonal(panel_axis);
  auto mpi_task_chain(comm_grid.communicator_pipeline<comm_dir>());

  broadcast(root, panel, mpi_task_chain);

  // check all panel are equal on all ranks
  for (const auto i_w : panel.iteratorLocal())
    CHECK_TILE_EQ(TypeUtil::element(root, 26), sync_wait(panel.read(i_w)).get());
}

TYPED_TEST(PanelBcastTest, BroadcastCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testBroadcast<TypeParam, Coord::Col, StoreTransposed::No>(cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testBroadcast<TypeParam, Coord::Row, StoreTransposed::No>(cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastColStoreTransposed) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testBroadcast<TypeParam, Coord::Col, StoreTransposed::Yes>(cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastRowStoreTransposed) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testBroadcast<TypeParam, Coord::Row, StoreTransposed::Yes>(cfg, comm_grid);
}

struct ParamsBcastTranspose {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalTileIndex offset;
  const GlobalTileIndex offsetT;
};

std::vector<ParamsBcastTranspose> test_params_bcast_transpose{
    {{0, 0}, {1, 1}, {0, 0}, {0, 0}},      // empty matrix
    {{10, 10}, {2, 2}, {5, 5}, {5, 5}},    // empty panel (due to offset)
    {{20, 20}, {2, 2}, {9, 9}, {10, 10}},  // just last tile (communicate without transpose)
    {{25, 25}, {5, 5}, {1, 1}, {1, 1}},
    {{25, 25}, {5, 5}, {1, 1}, {3, 3}},
};

template <class TypeParam, Coord AxisSrc, StoreTransposed storageT>
void testBroadcastTranspose(const ParamsBcastTranspose& cfg, comm::CommunicatorGrid& comm_grid) {
  using TypeUtil = TypeUtilities<TypeParam>;

  const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});
  const auto rank = dist.rankIndex().get(AxisSrc);

  // It is important to keep the order of initialization to avoid deadlocks!
  constexpr Coord AxisDst = orthogonal(AxisSrc);
  Panel<AxisSrc, TypeParam, dlaf::Device::CPU> panel_src(dist, cfg.offset);
  Panel<AxisDst, TypeParam, dlaf::Device::CPU, storageT> panel_dst(dist, cfg.offsetT);

  for (const auto i_w : panel_src.iteratorLocal()) {
    start_detached(panel_src.readwrite(i_w) |
                   then([rank](auto&& tile) { matrix::test::set(tile, TypeUtil::element(rank, 26)); }));
  }

  // test it!
  auto row_task_chain(comm_grid.row_communicator_pipeline());
  auto col_task_chain(comm_grid.col_communicator_pipeline());

  // select a "random" source rank which will be the source for the data
  const comm::IndexT_MPI owner = comm_grid.size().get(AxisSrc) / 2;

  broadcast(owner, panel_src, panel_dst, row_task_chain, col_task_chain);

  // Note:
  // all source panels will have access to the same data available on the root rank,
  // while the destination panels will have access to the corresponding "transposed" tile
  for (const auto idx : panel_src.iteratorLocal())
    CHECK_TILE_EQ(TypeUtil::element(owner, 26), sync_wait(panel_src.read(idx)).get());

  for (const auto idx : panel_dst.iteratorLocal()) {
    CHECK_TILE_EQ(TypeUtil::element(owner, 26), sync_wait(panel_dst.read(idx)).get());
  }
}

TYPED_TEST(PanelBcastTest, BroadcastCol2Row) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params_bcast_transpose)
      testBroadcastTranspose<TypeParam, Coord::Col, StoreTransposed::No>(cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastRow2Col) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params_bcast_transpose)
      testBroadcastTranspose<TypeParam, Coord::Row, StoreTransposed::No>(cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastCol2RowStoreTransposed) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params_bcast_transpose)
      testBroadcastTranspose<TypeParam, Coord::Col, StoreTransposed::Yes>(cfg, comm_grid);
}

TYPED_TEST(PanelBcastTest, BroadcastRow2ColStoreTransposed) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params_bcast_transpose)
      testBroadcastTranspose<TypeParam, Coord::Row, StoreTransposed::Yes>(cfg, comm_grid);
}
