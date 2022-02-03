//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/panel.h"

#include <vector>

#include <gtest/gtest.h>
#include <pika/unwrap.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_futures.h"
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
struct PanelTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(PanelTest, MatrixElementTypes);

struct config_t {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalTileIndex offset;
};

std::vector<config_t> test_params{
    {{0, 0}, {3, 3}, {0, 0}},  // empty matrix
    {{26, 13}, {3, 3}, {1, 2}},
};

TYPED_TEST(PanelTest, AssignToConstRef) {
  using namespace dlaf;
  using pika::unwrapping;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& cfg : test_params) {
      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> panel(dist);
      Panel<Coord::Col, const TypeParam, dlaf::Device::CPU>& ref = panel;

      std::vector<LocalTileIndex> exp_indices(panel.iteratorLocal().begin(),
                                              panel.iteratorLocal().end());
      std::vector<LocalTileIndex> ref_indices(ref.iteratorLocal().begin(), ref.iteratorLocal().end());

      EXPECT_EQ(exp_indices, ref_indices);

      for (const auto& idx : exp_indices) {
        const auto& exp_tile = panel.read(idx).get();
        auto get_element_ptr = [&exp_tile](const TileElementIndex& index) {
          return exp_tile.ptr(index);
        };
        CHECK_TILE_PTR(get_element_ptr, ref.read(idx).get());
      }
    }
  }
}

template <class TypeParam, Coord panel_axis>
void testIterator(const config_t& cfg, const comm::CommunicatorGrid& comm_grid) {
  const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  constexpr auto CT = decltype(panel)::CoordType;

  const auto offset_loc = dist.template nextLocalTileFromGlobalTile<CT>(cfg.offset.get<CT>());
  const auto exp_nrTiles = dist.localNrTiles().get<CT>();

  std::vector<LocalTileIndex> exp_indices;
  exp_indices.reserve(static_cast<size_t>(exp_nrTiles));
  for (auto index = offset_loc; index < exp_nrTiles; ++index)
    exp_indices.emplace_back(CT, index, 0);

  std::vector<LocalTileIndex> indices(panel.iteratorLocal().begin(), panel.iteratorLocal().end());

  EXPECT_EQ(exp_indices, indices);
}

TYPED_TEST(PanelTest, IteratorCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testIterator<TypeParam, Coord::Col>(cfg, comm_grid);
}

TYPED_TEST(PanelTest, IteratorRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testIterator<TypeParam, Coord::Row>(cfg, comm_grid);
}

template <class TypeParam, Coord panel_axis>
void testAccess(const config_t& cfg, const comm::CommunicatorGrid comm_grid) {
  using TypeUtil = TypeUtilities<TypeParam>;
  using pika::unwrapping;

  const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  constexpr Coord coord1D = decltype(panel)::CoordType;

  // rw-access
  for (const auto& idx : panel.iteratorLocal()) {
    panel(idx).then(unwrapping(
        [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(idx.get(coord1D), 26)); }));
  }

  // ro-access
  for (const auto& idx : panel.iteratorLocal())
    CHECK_MATRIX_EQ(TypeUtil::element(idx.get(coord1D), 26), panel.read(idx).get());
}

TYPED_TEST(PanelTest, AccessTileCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testAccess<TypeParam, Coord::Col>(cfg, comm_grid);
}

TYPED_TEST(PanelTest, AccessTileRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testAccess<TypeParam, Coord::Row>(cfg, comm_grid);
}

template <class TypeParam, Coord panel_axis>
void testExternalTile(const config_t& cfg, const comm::CommunicatorGrid comm_grid) {
  using TypeUtil = TypeUtilities<TypeParam>;
  using pika::unwrapping;

  constexpr Coord coord1D = orthogonal(panel_axis);

  Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();

  matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.get(coord1D), 26); });

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  // Note:
  // - Even indexed tiles in panel, odd indexed linked to the matrix first column
  // - Even indexed, i.e. the one using panel memory, are set to a different value
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 0)
      panel(idx).then(unwrapping(
          [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.get(coord1D), 13)); }));
    else
      panel.setTile(idx, matrix.read(idx));
  }

  // Check that the values are correct, both for internal and externally linked tiles
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 0)
      CHECK_TILE_EQ(TypeUtil::element(-idx.get(coord1D), 13), panel.read(idx).get());
    else
      CHECK_TILE_EQ(matrix.read(idx).get(), panel.read(idx).get());
  }

  // Reset external tiles links
  panel.reset();

  // Invert the "logic" of external tiles: even are linked to matrix, odd are in-panel
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 1)
      panel(idx).then(unwrapping(
          [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.get(coord1D), 5)); }));
    else
      panel.setTile(idx, matrix.read(idx));
  }

  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 1)
      CHECK_TILE_EQ(TypeUtil::element(-idx.get(coord1D), 5), panel.read(idx).get());
    else
      CHECK_TILE_EQ(matrix.read(idx).get(), panel.read(idx).get());
  }
}

TYPED_TEST(PanelTest, ExternalTilesCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testExternalTile<TypeParam, Coord::Col>(cfg, comm_grid);
}

TYPED_TEST(PanelTest, ExternalTilesRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testExternalTile<TypeParam, Coord::Row>(cfg, comm_grid);
}

template <class TypeParam, Coord panel_axis>
void testShrink(const config_t& cfg, const comm::CommunicatorGrid& comm_grid) {
  constexpr Coord coord1D = orthogonal(panel_axis);

  Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  auto setTile = [](const auto& tile, TypeParam value) noexcept {
    tile::internal::laset(lapack::MatrixType::General, value, value, tile);
  };

  auto setAndCheck = [=, &matrix, &panel](std::string msg, SizeType head_loc, SizeType tail_loc) {
    const auto message = ::testing::Message()
                         << msg << " head_loc:" << head_loc << " tail_loc:" << tail_loc;

    EXPECT_EQ(tail_loc - head_loc,
              std::distance(panel.iteratorLocal().begin(), panel.iteratorLocal().end()));

    SizeType counter = 0;
    for (SizeType k = head_loc; k < tail_loc; ++k) {
      const LocalTileIndex idx(coord1D, k);
      pika::dataflow(pika::unwrapping(setTile), panel(idx), counter++);
      const auto& tile = panel.read(idx).get();
      SCOPED_TRACE(message);
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }

    counter = 0;
    for (const auto& idx : panel.iteratorLocal()) {
      SCOPED_TRACE(message);
      CHECK_TILE_EQ(fixedValueTile(counter++), panel.read(idx).get());
      const auto& tile = panel.read(idx).get();
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }
  };

  // Shrink from head
  for (SizeType head = cfg.offset.get<coord1D>(); head <= dist.nrTiles().get(coord1D); ++head) {
    panel.setRangeStart(GlobalTileIndex(coord1D, head));

    const auto head_loc = dist.template nextLocalTileFromGlobalTile<coord1D>(head);
    const auto tail_loc = dist.localNrTiles().get(coord1D);

    setAndCheck("head", head_loc, tail_loc);

    panel.reset();
  }

  // Shrink from tail
  for (SizeType tail = dist.nrTiles().get(coord1D); cfg.offset.get<coord1D>() <= tail; --tail) {
    panel.setRangeStart(cfg.offset);
    panel.setRangeEnd(GlobalTileIndex(coord1D, tail));

    const auto head_loc = dist.template nextLocalTileFromGlobalTile<coord1D>(cfg.offset.get<coord1D>());
    const auto tail_loc = dist.template nextLocalTileFromGlobalTile<coord1D>(tail);

    setAndCheck("tail", head_loc, tail_loc);

    panel.reset();
  }

  // Shrink from both ends
  for (SizeType head = cfg.offset.get<coord1D>(), tail = dist.nrTiles().get(coord1D); head <= tail;
       ++head, --tail) {
    panel.setRange(GlobalTileIndex(coord1D, head), GlobalTileIndex(coord1D, tail));

    const auto head_loc = dist.template nextLocalTileFromGlobalTile<coord1D>(head);
    const auto tail_loc = dist.template nextLocalTileFromGlobalTile<coord1D>(tail);

    setAndCheck("both ends", head_loc, tail_loc);

    panel.reset();
  }
}

TYPED_TEST(PanelTest, ShrinkCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testShrink<TypeParam, Coord::Col>(cfg, comm_grid);
}

TYPED_TEST(PanelTest, ShrinkRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testShrink<TypeParam, Coord::Row>(cfg, comm_grid);
}

// For a col panel dim is the panel width.
// For a row panel dim is the panel height.
template <Coord panel_axis, class T, Device D>
void checkPanelTileSize(SizeType dim, Panel<panel_axis, T, D>& panel) {
  constexpr auto coord = std::decay_t<decltype(panel)>::CoordType;
  const Distribution& dist = panel.parentDistribution();
  for (SizeType i = panel.rangeStartLocal(); i < panel.rangeEndLocal(); ++i) {
    // Define the correct tile_size
    auto dim_perp = dist.blockSize().get<coord>();
    if (dist.globalTileFromLocalTile<coord>(i) == dist.nrTiles().get<coord>() - 1)
      dim_perp = dist.size().get<coord>() % dist.blockSize().get<coord>();
    const auto tile_size = [](auto dim, auto dim_perp) {
      return TileElementSize(panel_axis, dim, dim_perp);
    }(dim, dim_perp);

    EXPECT_EQ(tile_size, panel(LocalTileIndex{coord, i}).get().size());
    EXPECT_EQ(tile_size, panel.read(LocalTileIndex{coord, i}).get().size());
  }
}

TYPED_TEST(PanelTest, SetWidth) {
  for (auto& comm_grid : this->commGrids()) {
    const config_t cfg = {{26, 13}, {4, 5}, {0, 0}};

    Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});
    Panel<Coord::Col, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);

    const auto default_dim = cfg.blocksz.cols();

    checkPanelTileSize(default_dim, panel);
    // Check twice as size shouldn't change
    checkPanelTileSize(default_dim, panel);
    for (const auto dim : {default_dim / 2, default_dim}) {
      panel.reset();
      panel.setWidth(dim);
      checkPanelTileSize(dim, panel);
      // Check twice as size shouldn't change
      checkPanelTileSize(dim, panel);
    }
    panel.reset();
    checkPanelTileSize(default_dim, panel);
    // Check twice as size shouldn't change
    checkPanelTileSize(default_dim, panel);
  }
}

TYPED_TEST(PanelTest, SetHeight) {
  for (auto& comm_grid : this->commGrids()) {
    config_t cfg = {{26, 13}, {4, 5}, {0, 0}};

    Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});
    Panel<Coord::Row, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);

    const auto default_dim = cfg.blocksz.rows();

    checkPanelTileSize(default_dim, panel);
    // Check twice as size shouldn't change
    checkPanelTileSize(default_dim, panel);
    for (const auto dim : {default_dim / 2, default_dim}) {
      panel.reset();
      panel.setHeight(dim);
      checkPanelTileSize(dim, panel);
      // Check twice as size shouldn't change
      checkPanelTileSize(dim, panel);
    }
    panel.reset();
    checkPanelTileSize(default_dim, panel);
    // Check twice as size shouldn't change
    checkPanelTileSize(default_dim, panel);
  }
}
