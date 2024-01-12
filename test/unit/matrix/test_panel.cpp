//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <functional>
#include <type_traits>
#include <vector>

#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/sender/transform.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_senders.h>
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
struct PanelTest : public TestWithCommGrids {};

template <class T>
using PanelStoreTransposedTest = PanelTest<T>;

TYPED_TEST_SUITE(PanelTest, MatrixElementTypes);
TYPED_TEST_SUITE(PanelStoreTransposedTest, MatrixElementTypes);

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
    // square blocksize
    {{8, 5}, {3, 3}, {0, 0}},
    {{26, 13}, {3, 3}, {1, 2}},
    // non-square blocksize
    {{26, 13}, {5, 4}, {1, 2}},
    {{13, 26}, {5, 4}, {1, 2}},
};

template <Coord Axis, class T, Device D, StoreTransposed Storage>
void testStaticAPI() {
  using panel_t = Panel<Axis, T, D, Storage>;

  static_assert(panel_t::device == D, "wrong device");
  static_assert(panel_t::coord == orthogonal(Axis), "wrong coord");

  // MatrixLike Traits
  using ncT = std::remove_const_t<T>;
  static_assert(std::is_same_v<ncT, typename panel_t::ElementType>, "wrong ElementType");
  static_assert(std::is_same_v<Tile<ncT, D>, typename panel_t::TileType>, "wrong TileType");
  static_assert(std::is_same_v<Tile<const T, D>, typename panel_t::ConstTileType>,
                "wrong ConstTileType");
}

TYPED_TEST(PanelTest, StaticAPI) {
  testStaticAPI<Coord::Row, TypeParam, Device::CPU, StoreTransposed::No>();
  testStaticAPI<Coord::Col, TypeParam, Device::CPU, StoreTransposed::No>();

  testStaticAPI<Coord::Row, TypeParam, Device::GPU, StoreTransposed::No>();
  testStaticAPI<Coord::Col, TypeParam, Device::GPU, StoreTransposed::No>();
}

TYPED_TEST(PanelTest, StaticAPIConst) {
  testStaticAPI<Coord::Row, const TypeParam, Device::CPU, StoreTransposed::No>();
  testStaticAPI<Coord::Col, const TypeParam, Device::CPU, StoreTransposed::No>();

  testStaticAPI<Coord::Row, const TypeParam, Device::GPU, StoreTransposed::No>();
  testStaticAPI<Coord::Col, const TypeParam, Device::GPU, StoreTransposed::No>();
}

TYPED_TEST(PanelStoreTransposedTest, StaticAPI) {
  testStaticAPI<Coord::Row, TypeParam, Device::CPU, StoreTransposed::Yes>();
  testStaticAPI<Coord::Col, TypeParam, Device::CPU, StoreTransposed::Yes>();

  testStaticAPI<Coord::Row, TypeParam, Device::GPU, StoreTransposed::Yes>();
  testStaticAPI<Coord::Col, TypeParam, Device::GPU, StoreTransposed::Yes>();
}

TYPED_TEST(PanelStoreTransposedTest, StaticAPIConst) {
  testStaticAPI<Coord::Row, const TypeParam, Device::CPU, StoreTransposed::Yes>();
  testStaticAPI<Coord::Col, const TypeParam, Device::CPU, StoreTransposed::Yes>();

  testStaticAPI<Coord::Row, const TypeParam, Device::GPU, StoreTransposed::Yes>();
  testStaticAPI<Coord::Col, const TypeParam, Device::GPU, StoreTransposed::Yes>();
}

template <Coord Axis, class T, StoreTransposed Storage>
void testAssignToConstRef(const GlobalElementSize size, const TileElementSize blocksize,
                          comm::CommunicatorGrid& comm_grid) {
  const Distribution dist(size, blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<Axis, T, dlaf::Device::CPU, Storage> panel(dist);
  Panel<Axis, const T, dlaf::Device::CPU, Storage>& ref = panel;

  std::vector<LocalTileIndex> exp_indices(panel.iteratorLocal().begin(), panel.iteratorLocal().end());
  std::vector<LocalTileIndex> ref_indices(ref.iteratorLocal().begin(), ref.iteratorLocal().end());

  EXPECT_EQ(exp_indices, ref_indices);

  for (const auto& idx : exp_indices) {
    auto exp_tile_f = sync_wait(panel.read(idx));
    const auto& exp_tile = exp_tile_f.get();
    auto get_element_ptr = [&exp_tile](const TileElementIndex& index) { return exp_tile.ptr(index); };
    CHECK_TILE_PTR(get_element_ptr, sync_wait(ref.read(idx)).get());
  }
}

TYPED_TEST(PanelTest, AssignToConstRefCol) {
  using namespace dlaf;

  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, _] : test_params)
      testAssignToConstRef<Coord::Col, TypeParam, StoreTransposed::No>(size, blocksize, comm_grid);
}

TYPED_TEST(PanelTest, AssignToConstRefRow) {
  using namespace dlaf;

  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, _] : test_params)
      testAssignToConstRef<Coord::Row, TypeParam, StoreTransposed::No>(size, blocksize, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, AssignToConstRefCol) {
  using namespace dlaf;

  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, _] : test_params)
      testAssignToConstRef<Coord::Col, TypeParam, StoreTransposed::Yes>(size, blocksize, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, AssignToConstRefRow) {
  using namespace dlaf;

  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, _] : test_params)
      testAssignToConstRef<Coord::Row, TypeParam, StoreTransposed::Yes>(size, blocksize, comm_grid);
}

template <Coord Axis, class T, StoreTransposed Storage>
void testIterator(const GlobalElementSize size, const TileElementSize blocksize,
                  const GlobalTileIndex offset, comm::CommunicatorGrid& comm_grid) {
  const Distribution dist(size, blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<Axis, T, dlaf::Device::CPU, Storage> panel(dist, offset);
  constexpr Coord coord = decltype(panel)::coord;

  const auto offset_loc = dist.template nextLocalTileFromGlobalTile<coord>(offset.get<coord>());
  const auto exp_nrTiles = dist.localNrTiles().get<coord>();

  std::vector<LocalTileIndex> exp_indices;
  exp_indices.reserve(static_cast<size_t>(exp_nrTiles));
  for (auto index = offset_loc; index < exp_nrTiles; ++index)
    exp_indices.emplace_back(coord, index, 0);

  std::vector<LocalTileIndex> indices(panel.iteratorLocal().begin(), panel.iteratorLocal().end());

  EXPECT_EQ(exp_indices, indices);
}

TYPED_TEST(PanelTest, IteratorCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testIterator<Coord::Col, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelTest, IteratorRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testIterator<Coord::Row, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, IteratorCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testIterator<Coord::Col, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, IteratorRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testIterator<Coord::Row, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

template <Coord Axis, class T, StoreTransposed Storage>
void testAccess(const GlobalElementSize size, const TileElementSize blocksize,
                const GlobalTileIndex offset, comm::CommunicatorGrid& comm_grid) {
  using TypeUtil = TypeUtilities<T>;

  const Distribution dist(size, blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<Axis, T, dlaf::Device::CPU, Storage> panel(dist, offset);
  constexpr Coord coord = decltype(panel)::coord;

  // rw-access
  for (const auto& idx : panel.iteratorLocal()) {
    start_detached(panel.readwrite(idx) | then([idx](const auto& tile) {
                     matrix::test::set(tile, TypeUtil::element(idx.get(coord), 26));
                   }));
  }

  // ro-access
  for (const auto& idx : panel.iteratorLocal())
    CHECK_TILE_EQ(TypeUtil::element(idx.get(coord), 26), sync_wait(panel.read(idx)).get());

  // Repeat the same test with sender adaptors

  // Sender adaptors
  // rw-access
  for (const auto& idx : panel.iteratorLocal()) {
    dlaf::internal::transformDetach(
        dlaf::internal::Policy<dlaf::Backend::MC>(),
        [idx](const auto& tile) { matrix::test::set(tile, TypeUtil::element(idx.get(coord), 42)); },
        panel.readwrite(idx));
  }

  // ro-access
  for (const auto& idx : panel.iteratorLocal())
    CHECK_TILE_EQ(TypeUtil::element(idx.get(coord), 42), sync_wait(panel.read(idx)).get());
}

TYPED_TEST(PanelTest, AccessTileCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testAccess<Coord::Col, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelTest, AccessTileRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testAccess<Coord::Row, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, AccessTileCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testAccess<Coord::Col, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, AccessTileRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testAccess<Coord::Row, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

template <Coord Axis, class T, StoreTransposed Storage>
void testExternalTile(const GlobalElementSize size, const TileElementSize blocksize,
                      const GlobalTileIndex offset, comm::CommunicatorGrid& comm_grid) {
  using TypeUtil = TypeUtilities<T>;
  using dlaf::internal::Policy;
  using dlaf::internal::transformDetach;

  namespace tt = pika::this_thread::experimental;

  constexpr Coord coord = orthogonal(Axis);

  const comm::Index2D src_rank_idx(0, 0);
  const matrix::Distribution dist(size, blocksize, comm_grid.size(), comm_grid.rank(), src_rank_idx);

  // Note:
  // dist_t represents the full transposition of the distribution dist, which is useful when dealing with
  // StoreTransposed panels.
  // Indeed, a StoreTransposed panel represents a "classic" panel for a fully  transposed matrix (both
  // shape and rank distribution). So, passing the straight dist for panel construciton, we can find
  // work with shape-compatible if we construct the support matrix with dist_t.
  // It is important to highlight that, due to the transposition, a StoreTransposed::Yes Column panel,
  // will "match" the transposed matrix along the orthogonal axis, i.e. Row of the matrix, and viceversa.
  //
  // For this reason we have to conditionally (depending on StoreTransposed value) use:
  // - dist_t
  // - correctIndex helper
  [[maybe_unused]] const matrix::Distribution dist_t(
      common::transposed(size), common::transposed(blocksize), common::transposed(comm_grid.size()),
      common::transposed(comm_grid.rank()), common::transposed(src_rank_idx));

  const auto correctIndex = [](LocalTileIndex panel_ij) {
    if constexpr (StoreTransposed::Yes == Storage)
      panel_ij.transpose();
    return panel_ij;
  };

  Matrix<T, dlaf::Device::CPU> matrix(StoreTransposed::No == Storage ? dist : dist_t);
  Panel<Axis, T, dlaf::Device::CPU, Storage> panel(dist, offset);

  matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.get(coord), 26); });

  // if locally there are just incomplete tiles, skip the test (not worth it)
  if (doesThisRankOwnsJustIncomplete<Axis>(dist))
    return;

  // if there are no local tiles...cannot test external tiles
  if (dist.localNrTiles().isEmpty())
    return;

  // Note:
  // - Even indexed tiles in panel, odd indexed linked to the matrix first row/column
  // - Even indexed, i.e. the one using panel memory, are set to a different value
  for (auto idx : panel.iteratorLocal()) {
    if (idx.template get<coord>() % 2 == 0)
      panel.readwrite(idx) | transformDetach(Policy<Backend::MC>{}, [idx](const auto& tile) {
        matrix::test::set(tile, TypeUtil::element(-idx.get(coord), 13));
      });
    else
      panel.setTile(idx, matrix.read(correctIndex(idx)));
  }

  // Check that the values are correct, both for internal and externally linked tiles
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.template get<coord>() % 2 == 0)
      CHECK_TILE_EQ(TypeUtil::element(-idx.get(coord), 13), sync_wait(panel.read(idx)).get());
    else
      CHECK_TILE_EQ(sync_wait(matrix.read(correctIndex(idx))).get(), sync_wait(panel.read(idx)).get());
  }

  // Reset external tiles links
  panel.reset();

  // Invert the "logic" of external tiles: even are linked to matrix, odd are in-panel
  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.template get<coord>() % 2 == 0)
      panel.readwrite(idx) | transformDetach(Policy<Backend::MC>{}, [idx](const auto& tile) {
        matrix::test::set(tile, TypeUtil::element(-idx.get(coord), 5));
      });
    else
      panel.setTile(idx, matrix.read(correctIndex(idx)));
  }

  for (const auto& idx : panel.iteratorLocal()) {
    if (idx.template get<coord>() % 2 == 0)
      CHECK_TILE_EQ(TypeUtil::element(-idx.get(coord), 5), tt::sync_wait(panel.read(idx)).get());
    else
      CHECK_TILE_EQ(tt::sync_wait(matrix.read(correctIndex(idx))).get(),
                    tt::sync_wait(panel.read(idx)).get());
  }
}

TYPED_TEST(PanelTest, ExternalTilesCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testExternalTile<Coord::Col, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelTest, ExternalTilesRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testExternalTile<Coord::Row, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, ExternalTilesCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testExternalTile<Coord::Col, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, ExternalTilesRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testExternalTile<Coord::Row, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

template <Coord Axis, class T, StoreTransposed Storage>
void testShrink(const GlobalElementSize size, const TileElementSize blocksize,
                const GlobalTileIndex offset, comm::CommunicatorGrid& comm_grid) {
  constexpr Coord coord = orthogonal(Axis);

  Matrix<T, dlaf::Device::CPU> matrix(size, blocksize, comm_grid);
  const auto& dist = matrix.distribution();
  const SizeType bs = dist.blockSize().get(coord);

  Panel<Axis, T, dlaf::Device::CPU, Storage> panel(dist, offset);
  static_assert(coord == decltype(panel)::coord, "coord types mismatch");
  EXPECT_EQ(offset.get<coord>() * bs, panel.offsetElement());

  // if locally there are just incomplete tiles, skip the test (not worth it)
  if (doesThisRankOwnsJustIncomplete<Axis>(dist))
    return;

  // if there is no local tiles...there is nothing to check
  if (dist.localNrTiles().isEmpty())
    return;

  auto setTile = [](const auto& tile, T value) noexcept {
    tile::internal::laset(blas::Uplo::General, value, value, tile);
  };

  // Note:
  // The only difference between StoreTransposed::Yes and StoreTransposed::No is that in the former
  // case tiles are expected to have a transposed shape.
  auto setAndCheck = [=, &matrix, &panel](std::string msg, SizeType head_loc, SizeType tail_loc) {
    const auto message = ::testing::Message()
                         << msg << " head_loc:" << head_loc << " tail_loc:" << tail_loc;

    EXPECT_EQ(tail_loc - head_loc,
              std::distance(panel.iteratorLocal().begin(), panel.iteratorLocal().end()));

    SizeType counter = 0;
    for (SizeType k = head_loc; k < tail_loc; ++k) {
      const LocalTileIndex idx(coord, k);
      dlaf::internal::transformLiftDetach(dlaf::internal::Policy<Backend::MC>(), setTile,
                                          panel.readwrite(idx), counter++);

      // Getting the sender from the panel and getting a reference to the tile
      // are separated because combining them into one operation would lead to
      // the tile being a dangling reference since the tile wrapper sent by
      // panel.read(idx) is released at the end of the expression.
      //
      // Also note that sync_wait is not called inside EXPECT_EQ because it may
      // yield and change worker thread. SCOPED_TRACE uses thread locals and
      // does not support being created on one thread and destroyed on another
      // and will segfault if that happens.
      auto panel_tile_f = sync_wait(panel.read(idx));
      const auto& panel_tile = panel_tile_f.get();
      auto matrix_tile_f = sync_wait(matrix.read(idx));
      const auto& matrix_tile = matrix_tile_f.get();

      auto tile_size = matrix_tile.size();
      if constexpr (StoreTransposed::Yes == Storage)
        tile_size.transpose();

      SCOPED_TRACE(message);
      EXPECT_EQ(tile_size, panel_tile.size());
    }

    counter = 0;
    for (const auto& idx : panel.iteratorLocal()) {
      // See comment in previous for loop. This section has the same concerns
      // regarding dangling references and yielding with SCOPED_TRACE.
      auto panel_tile_f = sync_wait(panel.read(idx));
      const auto& panel_tile = panel_tile_f.get();
      auto matrix_tile_f = sync_wait(matrix.read(idx));
      const auto& matrix_tile = matrix_tile_f.get();

      auto tile_size = matrix_tile.size();
      if constexpr (StoreTransposed::Yes == Storage)
        tile_size.transpose();

      SCOPED_TRACE(message);
      EXPECT_EQ(tile_size, panel_tile.size());

      CHECK_TILE_EQ(fixedValueTile(counter++), panel_tile);
    }
  };

  // Shrink from head
  for (SizeType head = offset.get<coord>(); head <= dist.nrTiles().get(coord); ++head) {
    panel.setRangeStart(GlobalTileIndex(coord, head));
    EXPECT_EQ(head * bs, panel.offsetElement());

    const auto head_loc = dist.template nextLocalTileFromGlobalTile<coord>(head);
    const auto tail_loc = dist.localNrTiles().get(coord);

    setAndCheck("head", head_loc, tail_loc);

    panel.reset();
  }

  // Shrink from tail
  for (SizeType tail = dist.nrTiles().get(coord); offset.get<coord>() <= tail; --tail) {
    panel.setRangeStart(offset);
    panel.setRangeEnd(GlobalTileIndex(coord, tail));
    EXPECT_EQ(offset.get<coord>() * bs, panel.offsetElement());

    const auto head_loc = dist.template nextLocalTileFromGlobalTile<coord>(offset.get<coord>());
    const auto tail_loc = dist.template nextLocalTileFromGlobalTile<coord>(tail);

    setAndCheck("tail", head_loc, tail_loc);

    panel.reset();
  }

  // Shrink from both ends
  for (SizeType head = offset.get<coord>(), tail = dist.nrTiles().get(coord); head <= tail;
       ++head, --tail) {
    panel.setRange(GlobalTileIndex(coord, head), GlobalTileIndex(coord, tail));
    EXPECT_EQ(head * bs, panel.offsetElement());

    const auto head_loc = dist.template nextLocalTileFromGlobalTile<coord>(head);
    const auto tail_loc = dist.template nextLocalTileFromGlobalTile<coord>(tail);

    setAndCheck("both ends", head_loc, tail_loc);

    panel.reset();
  }
}

TYPED_TEST(PanelTest, ShrinkCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testShrink<Coord::Col, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelTest, ShrinkRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testShrink<Coord::Row, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, ShrinkCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testShrink<Coord::Col, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, ShrinkRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testShrink<Coord::Row, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
}

template <Coord CoordMutable, Coord Axis, class T, Device D, StoreTransposed Storage>
void checkPanelTileSize(SizeType dim_mutable, Panel<Axis, T, D, Storage>& panel) {
  constexpr Coord CoordFixed = orthogonal(CoordMutable);
  const Distribution& dist = panel.parentDistribution();

  for (const auto ij : panel.iteratorLocal()) {
    const TileElementSize full_tile_size = dist.tileSize(dist.globalTileIndex(ij));
    // Note:
    // Get the fixed coordinate value, but in case it is StoreTransposed::Yes, the expected full tile
    // size is tranposed wrt the matrix related one.
    const SizeType dim_fixed =
        (StoreTransposed::No == Storage ? full_tile_size : common::transposed(full_tile_size))
            .template get<CoordFixed>();
    const TileElementSize tile_size(CoordMutable, dim_mutable, dim_fixed);

    EXPECT_EQ(tile_size, sync_wait(panel.readwrite(ij)).size());
    EXPECT_EQ(tile_size, sync_wait(panel.read(ij)).get().size());
  }
}

template <Coord Axis, class T, StoreTransposed Storage>
void testSetMutable(const GlobalElementSize size, const TileElementSize blocksize,
                    const GlobalTileIndex offset, comm::CommunicatorGrid& comm_grid) {
  const Distribution dist(size, blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});

  constexpr Coord CoordFixed = StoreTransposed::No == Storage ? orthogonal(Axis) : Axis;
  constexpr Coord CoordMutable = orthogonal(CoordFixed);

  using PanelType = Panel<Axis, T, dlaf::Device::CPU, Storage>;

  PanelType panel(dist, offset);

  // Note:
  // These next two helpers hides in the test the difference in accessing information about the
  // mutable dimension for the two different Panel axis.
  auto getMutableDim = []() {
    using Func = SizeType (PanelType::*)(void) const;
    if constexpr (Coord::Col == Axis)
      return std::mem_fn(Func(&PanelType::getWidth));
    else
      return std::mem_fn(Func(&PanelType::getHeight));
  }();

  auto setMutableDim = []() {
    using Func = void (PanelType::*)(SizeType);
    if constexpr (Coord::Col == Axis)
      return std::mem_fn(Func(&PanelType::setWidth));
    else
      return std::mem_fn(Func(&PanelType::setHeight));
  }();

  const SizeType default_dim =
      (StoreTransposed::No == Storage ? blocksize : common::transposed(blocksize))
          .template get<CoordMutable>();

  EXPECT_EQ(default_dim, getMutableDim(panel));
  checkPanelTileSize<CoordMutable>(default_dim, panel);
  // Check twice as size shouldn't change
  checkPanelTileSize<CoordMutable>(default_dim, panel);

  for (const SizeType dim : {default_dim / 2, default_dim}) {
    panel.reset();
    setMutableDim(panel, dim);
    EXPECT_EQ(dim, getMutableDim(panel));
    checkPanelTileSize<CoordMutable>(dim, panel);
    // Check twice as size shouldn't change
    checkPanelTileSize<CoordMutable>(dim, panel);
  }

  panel.reset();
  EXPECT_EQ(default_dim, getMutableDim(panel));
  checkPanelTileSize<CoordMutable>(default_dim, panel);
  // Check twice as size shouldn't change
  checkPanelTileSize<CoordMutable>(default_dim, panel);
}

TYPED_TEST(PanelTest, SetMutableDim) {
  for (auto& comm_grid : this->commGrids()) {
    const auto [size, blocksize, offset] = config_t{{26, 13}, {4, 5}, {0, 0}};
    testSetMutable<Coord::Col, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
    testSetMutable<Coord::Row, TypeParam, StoreTransposed::No>(size, blocksize, offset, comm_grid);
  }
}

TYPED_TEST(PanelStoreTransposedTest, SetMutableDim) {
  for (auto& comm_grid : this->commGrids()) {
    const auto [size, blocksize, offset] = config_t{{26, 13}, {4, 5}, {0, 0}};
    testSetMutable<Coord::Col, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
    testSetMutable<Coord::Row, TypeParam, StoreTransposed::Yes>(size, blocksize, offset, comm_grid);
  }
}

template <Coord Axis, class T, StoreTransposed Storage>
void testOffsetTileUnaligned(const GlobalElementSize size, const TileElementSize blocksize,
                             comm::CommunicatorGrid& comm_grid) {
  const Distribution dist(size, blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});

  Panel<Axis, T, dlaf::Device::CPU, Storage> panel(dist);
  constexpr Coord coord = decltype(panel)::coord;

  const SizeType size_axis = std::min(blocksize.get<Axis>(), size.get<Axis>());
  const GlobalElementSize panel_size(coord, size.get<coord>(), size_axis);

  // use each row/col of the matrix as offset
  for (SizeType offset_index = 0; offset_index < panel_size.get(coord); ++offset_index) {
    const GlobalElementIndex offset_e(coord, offset_index);
    const SizeType offset = dist.globalTileFromGlobalElement<coord>(offset_e.get<coord>());

    panel.setRangeStart(offset_e);
    EXPECT_EQ(offset_e.get<coord>(), panel.offsetElement());

    for (const LocalTileIndex& i : panel.iteratorLocal()) {
      TileElementSize expected_tile_size = [&]() {
        // Get the index of the first column
        const GlobalTileIndex i_global(coord, dist.globalTileFromLocalTile<coord>(i.get<coord>()));

        const TileElementSize full_size = dist.tileSize(i_global);

        // just first global tile may have offset, others are full size, compatibly with matrix size
        if (i_global.get<coord>() != offset)
          return full_size;

        // by computing the offseted size with repsect to the acutal tile size, it also checks the
        // edge case where a panel has a single tile, both offseted and "incomplete"
        const SizeType sub_offset = dist.tileElementFromGlobalElement<coord>(offset_e.get<coord>());

        return TileElementSize(coord, full_size.get<coord>() - sub_offset, size_axis);
      }();

      if constexpr (StoreTransposed::Yes == Storage)
        expected_tile_size.transpose();

      EXPECT_EQ(expected_tile_size, sync_wait(panel.read(i)).get().size());
      EXPECT_EQ(expected_tile_size, sync_wait(panel.readwrite(i)).size());
    }

    panel.reset();
  }
}

TYPED_TEST(PanelTest, OffsetTileUnalignedRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testOffsetTileUnaligned<Coord::Row, TypeParam, StoreTransposed::No>(size, blocksize, comm_grid);
}

TYPED_TEST(PanelTest, OffsetTileUnalignedCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, _] : test_params)
      testOffsetTileUnaligned<Coord::Col, TypeParam, StoreTransposed::No>(size, blocksize, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, OffsetTileUnalignedRow) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, offset] : test_params)
      testOffsetTileUnaligned<Coord::Row, TypeParam, StoreTransposed::Yes>(size, blocksize, comm_grid);
}

TYPED_TEST(PanelStoreTransposedTest, OffsetTileUnalignedCol) {
  for (auto& comm_grid : this->commGrids())
    for (const auto& [size, blocksize, _] : test_params)
      testOffsetTileUnaligned<Coord::Col, TypeParam, StoreTransposed::Yes>(size, blocksize, comm_grid);
}

struct ConfigSub {
  const TileElementSize block_size;
  const GlobalElementIndex sub_origin;
  const GlobalElementSize sub_size;
};

const std::vector<ConfigSub> sub_configs{
    {{3, 4}, {4, 2}, {12, 20}},
    {{3, 5}, {7, 1}, {12, 11}},
};

TYPED_TEST(PanelTest, MatrixWithOffset) {
  namespace ex = pika::execution::experimental;
  namespace tt = pika::this_thread::experimental;
  namespace di = dlaf::internal;

  using dlaf::matrix::internal::MatrixRef;

  for (const auto& [block_size, sub_origin, sub_size] : sub_configs) {
    const LocalElementSize full_size{sub_origin.row() + sub_size.rows(),
                                     sub_origin.col() + sub_size.cols()};

    Matrix<TypeParam, Device::CPU> mat(full_size, block_size);
    MatrixRef<TypeParam, Device::CPU> mat_sub(mat, {sub_origin, sub_size});

    Panel<Coord::Col, TypeParam, Device::CPU, StoreTransposed::No> panel(mat_sub.distribution());

    const LocalTileSize sub_size_lc = mat_sub.distribution().local_nr_tiles();
    for (SizeType j_lc = 0; j_lc < sub_size_lc.cols(); ++j_lc) {
      if (j_lc == 0 || j_lc == sub_size_lc.cols() - 1) {
        panel.setWidth(mat_sub.distribution().template tile_size_of<Coord::Col>(j_lc));
      }

      for (SizeType i_lc = 0; i_lc < sub_size_lc.rows(); ++i_lc) {
        const LocalTileIndex ij_lc(i_lc, j_lc);
        tt::sync_wait(ex::when_all(panel.read(ij_lc), mat_sub.read(ij_lc)) |
                      di::transform(di::Policy<Backend::MC>(), [](auto& tile_p, auto& tile_m) {
                        EXPECT_EQ(tile_p.size(), tile_m.size());
                      }));
      }

      panel.reset();
    }
  }
}
