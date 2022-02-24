//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/index2d.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/panel.h"

#include <vector>

#include <gtest/gtest.h>
#include <pika/unwrap.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/sender/transform.h"
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

// Helper for checking if current rank, along specific Axis, locally stores just an incomplete tile,
// i.e. a tile with a size < blocksize
template <Coord Axis>
bool doesThisRankOwnsJustIncomplete(const matrix::Distribution& dist) {
  // an empty matrix does not fall in this specific edge-case
  if (dist.nrTiles().isEmpty())
    return false;

  // look at the last tile along panel_axis dimension, and see if its size is full or not
  const GlobalTileIndex last_tile_axis(Axis, dist.nrTiles().get(Axis) - 1);
  const bool is_last_tile =
      dist.rankIndex().get(Axis) == dist.template rankGlobalTile<Axis>(last_tile_axis.get(Axis));
  return is_last_tile && dist.localNrTiles().get(Axis) == 1 &&
         dist.tileSize(last_tile_axis).get(Axis) != dist.blockSize().get(Axis);
}

struct config_t {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalTileIndex offset = {0, 0};
};

std::vector<config_t> test_params{
    {{0, 0}, {3, 3}, {0, 0}},  // empty matrix
    {{8, 5}, {3, 3}, {0, 0}},
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

  using PanelType = Panel<panel_axis, TypeParam, dlaf::Device::CPU>;
  PanelType panel(dist, cfg.offset);
  constexpr Coord coord1D = decltype(panel)::CoordType;

  // rw-access
  for (const auto& idx : panel.iteratorLocal()) {
    panel(idx).then(unwrapping(
        [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(idx.get(coord1D), 26)); }));
  }

  // ro-access
  for (const auto& idx : panel.iteratorLocal())
    CHECK_MATRIX_EQ(TypeUtil::element(idx.get(coord1D), 26), panel.read(idx).get());

  // Repeat the same test with sender adaptors

  // Sender adaptors
  // rw-access
  for (const auto& idx : panel.iteratorLocal()) {
    dlaf::internal::transformDetach(
        dlaf::internal::Policy<dlaf::Backend::MC>(),
        [idx](typename PanelType::TileType&& tile) {
          matrix::test::set(tile, TypeUtil::element(idx.get(coord1D), 42));
        },
        panel.readwrite_sender(idx));
  }

  // ro-access
  for (const auto& idx : panel.iteratorLocal())
    CHECK_MATRIX_EQ(TypeUtil::element(idx.get(coord1D), 42),
                    pika::execution::experimental::sync_wait(panel.read_sender(idx)).get());
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

  using MatrixType = Matrix<TypeParam, dlaf::Device::CPU>;
  MatrixType matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();

  matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.get(coord1D), 26); });

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  // if locally there are just incomplete tiles, skip the test (not worth it)
  if (doesThisRankOwnsJustIncomplete<panel_axis>(dist))
    return;

  // if there is no local tiles...cannot test external tiles
  if (dist.localNrTiles().isEmpty())
    return;

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

template <class TypeParam, Coord panel_axis>
void testExternalTileWithSenders(const config_t& cfg, const comm::CommunicatorGrid comm_grid) {
  using TypeUtil = TypeUtilities<TypeParam>;
  using dlaf::internal::transformDetach;
  using dlaf::internal::Policy;
  using pika::execution::experimental::sync_wait;
  using pika::unwrapping;

  constexpr Coord coord1D = orthogonal(panel_axis);

  using MatrixType = Matrix<TypeParam, dlaf::Device::CPU>;
  using TileType = typename MatrixType::TileType;
  MatrixType matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();
  const Policy<Backend::MC> policy{};

  matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.get(coord1D), 26); });

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  // Note:
  // - Even indexed tiles in panel, odd indexed linked to the matrix first column
  // - Even indexed, i.e. the one using panel memory, are set to a different value
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 0)
      panel.readwrite_sender(idx) | transformDetach(policy, [idx](TileType&& tile) {
        matrix::test::set(tile, TypeUtil::element(-idx.get(coord1D), 13));
      });
    else
      panel.setTile(idx, matrix.read(idx));
  }

  // Check that the values are correct, both for internal and externally linked tiles
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 0)
      CHECK_TILE_EQ(TypeUtil::element(-idx.get(coord1D), 13), sync_wait(panel.read_sender(idx)).get());
    else
      CHECK_TILE_EQ(sync_wait(matrix.read_sender(idx)).get(), sync_wait(panel.read_sender(idx)).get());
  }

  // Reset external tiles links
  panel.reset();

  // Invert the "logic" of external tiles: even are linked to matrix, odd are in-panel
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 1)
      panel.readwrite_sender(idx) | transformDetach(policy, [idx](TileType&& tile) {
        matrix::test::set(tile, TypeUtil::element(-idx.get(coord1D), 5));
      });
    else
      panel.setTile(idx, matrix.read(idx));
  }

  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.row() % 2 == 1)
      CHECK_TILE_EQ(TypeUtil::element(-idx.get(coord1D), 5), sync_wait(panel.read_sender(idx)).get());
    else
      CHECK_TILE_EQ(sync_wait(matrix.read_sender(idx)).get(), sync_wait(panel.read_sender(idx)).get());
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

TYPED_TEST(PanelTest, ExternalTilesColWithSenders) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testExternalTileWithSenders<TypeParam, Coord::Col>(cfg, comm_grid);
}

TYPED_TEST(PanelTest, ExternalTilesRowWithSenders) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& cfg : test_params)
      testExternalTileWithSenders<TypeParam, Coord::Row>(cfg, comm_grid);
}

template <class TypeParam, Coord panel_axis>
void testShrink(const config_t& cfg, const comm::CommunicatorGrid& comm_grid) {
  constexpr Coord coord1D = orthogonal(panel_axis);

  Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
  const auto& dist = matrix.distribution();

  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);
  static_assert(coord1D == decltype(panel)::CoordType, "coord types mismatch");

  // if locally there are just incomplete tiles, skip the test (not worth it)
  if (doesThisRankOwnsJustIncomplete<panel_axis>(dist))
    return;

  // if there is no local tiles...there is nothing to check
  if (dist.localNrTiles().isEmpty())
    return;

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
      dlaf::internal::transformLiftDetach(dlaf::internal::Policy<Backend::MC>(), setTile,
                                          panel.readwrite_sender(idx), counter++);
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

template <class T, Coord Axis>
void testOffsetTileUnaligned(const GlobalElementSize size, const TileElementSize blocksize,
                             const comm::CommunicatorGrid& comm_grid) {
  const Distribution dist(size, blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<Axis, T, dlaf::Device::CPU> panel(dist);
  constexpr auto Coord1D = Panel<Axis, T, Device::CPU>::CoordType;

  const SizeType size_axis = std::min(blocksize.get<Axis>(), size.get<Axis>());
  const GlobalElementSize panel_size(Coord1D, size.get<Coord1D>(), size_axis);

  // use each row of the matrix as offset
  for (SizeType offset_index = 0; offset_index < panel_size.get(Coord1D); ++offset_index) {
    const GlobalElementIndex offset_e(Coord1D, offset_index);
    const SizeType offset = dist.globalTileFromGlobalElement<Coord1D>(offset_e.get<Coord1D>());

    panel.setRangeStart(offset_e);

    for (const LocalTileIndex& i : panel.iteratorLocal()) {
      const TileElementSize expected_size = [&]() {
        // Note:  globalTile used with GlobalElementIndex is preferred over the one with LocalTileIndex,
        //        because the former one does not implicitly target anything local, that would otherwise
        //        be problematic in case of ranks that do not have any part of the matrix locally.
        const GlobalTileIndex i_global(Coord1D, dist.globalTileFromLocalTile<Coord1D>(i.get<Coord1D>()));

        const TileElementSize full_size = dist.tileSize(i_global);

        // just first global tile may have offset, others are full size, compatibly with matrix size
        if (i_global.get<Coord1D>() != offset)
          return full_size;

        // by computing the offseted size with repsect to the acutal tile size, it also checks the
        // edge case where a panel has a single tile, both offseted and "incomplete"
        const SizeType sub_offset = dist.tileElementFromGlobalElement<Coord1D>(offset_e.get<Coord1D>());
        return TileElementSize(Coord1D, full_size.get<Coord1D>() - sub_offset, size_axis);
      }();

      EXPECT_EQ(expected_size, panel.read(i).get().size());
      EXPECT_EQ(expected_size, panel(i).get().size());
    }

    panel.reset();
  }
}

TYPED_TEST(PanelTest, OffsetTileUnalignedRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testOffsetTileUnaligned<TypeParam, Coord::Row>(size, blocksize, comm_grid);
}

TYPED_TEST(PanelTest, OffsetTileUnalignedCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testOffsetTileUnaligned<TypeParam, Coord::Col>(size, blocksize, comm_grid);
}
