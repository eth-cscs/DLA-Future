//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/panel.h"

#include <vector>

#include <gtest/gtest.h>
#include <hpx/future.hpp>
#include <hpx/include/parallel_executors.hpp>

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
struct PanelTest : public ::testing::Test {
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(PanelTest, MatrixElementTypes);

struct config_t {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalElementIndex offset;
};

std::vector<config_t> test_params{
    {{0, 0}, {3, 3}, {0, 0}},  // empty matrix
    {{26, 13}, {3, 3}, {1, 2}},
};

TYPED_TEST(PanelTest, AssignToConstRef) {
  using namespace dlaf;
  using hpx::util::unwrapping;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& cfg : test_params) {
      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> panel(dist);
      Panel<Coord::Col, const TypeParam, dlaf::Device::CPU>& ref = panel;

      std::vector<LocalTileIndex> exp_indices(panel.iterator().begin(), panel.iterator().end());
      std::vector<LocalTileIndex> ref_indices(ref.iterator().begin(), ref.iterator().end());

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

  const LocalTileSize at_offset(dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
                                dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()));

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, at_offset);
  constexpr Coord coord1D = decltype(panel)::CoordType;

  const auto exp_nrTiles = dist.localNrTiles().get<coord1D>();

  std::vector<LocalTileIndex> exp_indices;
  exp_indices.reserve(static_cast<size_t>(exp_nrTiles));
  for (auto index = at_offset.get<coord1D>(); index < exp_nrTiles; ++index)
    exp_indices.emplace_back(coord1D, index, 0);

  std::vector<LocalTileIndex> indices(panel.iterator().begin(), panel.iterator().end());

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
  using hpx::util::unwrapping;

  const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

  const LocalTileSize at_offset(dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
                                dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()));

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, at_offset);
  constexpr Coord coord1D = decltype(panel)::CoordType;

  // rw-access
  for (const auto& idx : panel.iterator()) {
    panel(idx).then(unwrapping(
        [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(idx.get(coord1D), 26)); }));
  }

  // ro-access
  for (const auto& idx : panel.iterator())
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
  using hpx::util::unwrapping;

  constexpr Coord coord1D = orthogonal(panel_axis);

  Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();

  matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.get(coord1D), 26); });

  const LocalTileSize at_offset{
      dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
      dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
  };

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, at_offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  // Note:
  // - Even indexed tiles in panel, odd indexed linked to the matrix first column
  // - Even indexed, i.e. the one using panle memory, are set to a different value
  for (const auto& idx : panel.iterator()) {
    if (idx.row() % 2 == 0)
      panel(idx).then(unwrapping(
          [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.get(coord1D), 13)); }));
    else
      panel.setTile(idx, matrix.read(idx));
  }

  // Check that the values are correct, both for internal and externally linked tiles
  for (const auto& idx : panel.iterator()) {
    if (idx.row() % 2 == 0)
      CHECK_TILE_EQ(TypeUtil::element(-idx.get(coord1D), 13), panel.read(idx).get());
    else
      CHECK_TILE_EQ(matrix.read(idx).get(), panel.read(idx).get());
  }

  // Reset external tiles links
  panel.reset();

  // Invert the "logic" of external tiles: even are linked to matrix, odd are in-panel
  for (const auto& idx : panel.iterator()) {
    if (idx.row() % 2 == 1)
      panel(idx).then(unwrapping(
          [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.get(coord1D), 5)); }));
    else
      panel.setTile(idx, matrix.read(idx));
  }

  for (const auto& idx : panel.iterator()) {
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
  using TypeUtil = TypeUtilities<TypeParam>;

  constexpr Coord coord1D = orthogonal(panel_axis);

  Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();

  // setup the panel
  const LocalTileSize at_offset{
      dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
      dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
  };

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, at_offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  for (SizeType i = at_offset.get(coord1D); i < dist.localNrTiles().get(coord1D); ++i)
    panel(LocalTileIndex(coord1D, i)).get()({0, 0}) = i;

  // Shrink from head
  for (SizeType head = at_offset.get(coord1D); head <= dist.localNrTiles().get(coord1D); ++head) {
    panel.setRangeStart(LocalTileSize(coord1D, head));

    for (SizeType k = head; k < dist.localNrTiles().get(coord1D); ++k) {
      const LocalTileIndex idx(coord1D, k);
      auto& tile = panel.read(idx).get();
      EXPECT_EQ(tile({0, 0}), TypeUtil::element(k, 0));
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }

    for (const auto& idx : panel.iterator()) {
      auto& tile = panel.read(idx).get();
      EXPECT_EQ(tile({0, 0}), TypeUtil::element(idx.get(coord1D), 0));
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }
    EXPECT_EQ(dist.localNrTiles().get(coord1D) - head,
              std::distance(panel.iterator().begin(), panel.iterator().end()));
  }

  // Shrink from tail
  panel.setRangeStart(at_offset);

  for (SizeType tail = dist.localNrTiles().get(coord1D); at_offset.get(coord1D) <= tail; --tail) {
    panel.setRangeEnd(LocalTileSize(coord1D, tail));

    for (SizeType k = at_offset.get(coord1D); k < tail; ++k) {
      const LocalTileIndex idx(coord1D, k);
      auto& tile = panel.read(idx).get();
      EXPECT_EQ(tile({0, 0}), TypeUtil::element(k, 0));
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }

    for (const auto& idx : panel.iterator()) {
      auto& tile = panel.read(idx).get();
      EXPECT_EQ(tile({0, 0}), TypeUtil::element(idx.get(coord1D), 0));
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }
    EXPECT_EQ(tail - at_offset.get(coord1D),
              std::distance(panel.iterator().begin(), panel.iterator().end()));
  }

  // Shrink from both ends
  panel.setRangeEnd(dist.localNrTiles());

  for (SizeType head = at_offset.get(coord1D), tail = dist.localNrTiles().get(coord1D); head <= tail;
       ++head, --tail) {
    panel.setRange(LocalTileSize(coord1D, head), LocalTileSize(coord1D, tail));

    for (SizeType k = head; k < tail; ++k) {
      const LocalTileIndex idx(coord1D, k);
      auto& tile = panel.read(idx).get();
      EXPECT_EQ(tile({0, 0}), TypeUtil::element(k, 0));
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }

    for (const auto& idx : panel.iterator()) {
      auto& tile = panel.read(idx).get();
      EXPECT_EQ(tile({0, 0}), TypeUtil::element(idx.get(coord1D), 0));
      EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
    }
    EXPECT_EQ(tail - head, std::distance(panel.iterator().begin(), panel.iterator().end()));
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
