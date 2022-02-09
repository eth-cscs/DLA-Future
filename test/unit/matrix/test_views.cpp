//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/views.h"

#include <sstream>

#include <gtest/gtest.h>
#include <gtest/internal/gtest-internal.h>

#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"

using namespace dlaf;
using namespace dlaf::test;

using matrix::SubPanelView;
using matrix::SubMatrixView;

using matrix::Distribution;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

GlobalElementSize globalSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

namespace test::SubMatrixView {
struct config_t {
  LocalElementSize size;
  TileElementSize blocksize;
  GlobalElementIndex offset;

  friend std::ostream& operator<<(std::ostream& os, const config_t& config) {
    return os << config.size << " " << config.blocksize << " " << config.offset;
  }
};

const std::vector<config_t> configs{
    // complete-tiles
    {{8, 8}, {4, 4}, {0, 0}},    // no offset
    {{12, 12}, {4, 4}, {0, 0}},  // no offset
    {{8, 8}, {4, 4}, {6, 4}},    // offset
    {{12, 12}, {4, 4}, {2, 2}},  // symmetric-offset
    {{12, 12}, {4, 4}, {4, 2}},  // asymmetric-offset
    // incomplete tiles
    {{13, 13}, {4, 4}, {4, 2}},  // symmetric offset
    {{13, 13}, {4, 4}, {4, 2}},  // asymmetric offset
                                 // TODO non-square tiles
                                 // TODO non-square matrix
};
}

void testMatrixOffset(const Distribution& dist, const GlobalElementIndex& offset_e) {
  const SubMatrixView view(dist, offset_e);

  const GlobalTileIndex offset = dist.globalTileIndex(offset_e);
  EXPECT_EQ(view.offset(), offset);
}

void testMatrixRange(const Distribution& dist, const GlobalElementIndex& offset_e) {
  const SubMatrixView view(dist, offset_e);

  const LocalTileIndex offset_local = {
      dist.template nextLocalTileFromGlobalElement<Coord::Row>(offset_e.row()),
      dist.template nextLocalTileFromGlobalElement<Coord::Col>(offset_e.col()),
  };

  std::list<LocalTileIndex> indices_in_range;
  for (SizeType j = offset_local.col(); j < dist.localNrTiles().cols(); ++j)
    for (SizeType i = offset_local.row(); i < dist.localNrTiles().rows(); ++i)
      indices_in_range.emplace_back(LocalTileIndex(i, j));

  std::list<LocalTileIndex> indices_from_view;
  for (const auto ij : view.iteratorLocal())
    indices_from_view.emplace_back(ij);

  EXPECT_EQ(indices_in_range, indices_from_view);
}

void testMatrixSpecs(const Distribution& dist, const GlobalElementIndex offset_e) {
  const SubMatrixView view(dist, offset_e);

  const LocalTileIndex offset_local{
      dist.template nextLocalTileFromGlobalElement<Coord::Row>(offset_e.row()),
      dist.template nextLocalTileFromGlobalElement<Coord::Col>(offset_e.col()),
  };

  for (SizeType j = offset_local.col(); j < dist.localNrTiles().cols(); ++j) {
    for (SizeType i = offset_local.row(); i < dist.localNrTiles().rows(); ++i) {
      const LocalTileIndex ij_local(i, j);
      const GlobalTileIndex ij(dist.globalTileIndex(ij_local));

      SizeType sub_i = 0, sub_j = 0;
      if (ij.row() == view.offset().row())
        sub_i += dist.tileElementIndex(offset_e).row();
      if (ij.col() == view.offset().col())
        sub_j += dist.tileElementIndex(offset_e).col();

      const TileElementIndex sub_offset(sub_i, sub_j);
      const TileElementSize sub_size = dist.tileSize(ij) - TileElementSize(sub_i, sub_j);

      const matrix::SubTileSpec spec = view(ij_local);

      EXPECT_EQ(spec.size, sub_size);
      EXPECT_EQ(spec.origin, sub_offset);
    }
  }
}

using MatrixViewTest = ::testing::TestWithParam<::test::SubMatrixView::config_t>;

INSTANTIATE_TEST_SUITE_P(AllConfigs, MatrixViewTest,
                         ::testing::ValuesIn(::test::SubMatrixView::configs));

TEST_P(MatrixViewTest, OffsetLocal) {
  const auto& [size, blocksize, offset_e] = GetParam();
  const Distribution dist(size, blocksize);
  testMatrixOffset(dist, offset_e);
}

TEST_P(MatrixViewTest, OffsetDistributed) {
  const auto& [size, blocksize, offset_e] = GetParam();
  for (const auto& comm_grid : comm_grids) {
    const Distribution dist(globalSize(size), blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});
    testMatrixOffset(dist, offset_e);
  }
}

TEST_P(MatrixViewTest, RangeLocal) {
  const auto& [size, blocksize, offset_e] = GetParam();
  const Distribution dist(size, blocksize);
  testMatrixRange(dist, offset_e);
}

TEST_P(MatrixViewTest, RangeDistributed) {
  const auto& [size, blocksize, offset_e] = GetParam();
  for (const auto& comm_grid : comm_grids) {
    const Distribution dist(globalSize(size), blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});
    testMatrixRange(dist, offset_e);
  }
}

TEST_P(MatrixViewTest, SpecsLocal) {
  const auto& [size, blocksize, offset_e] = GetParam();
  const Distribution dist(size, blocksize);
  testMatrixSpecs(dist, offset_e);
}

TEST_P(MatrixViewTest, SpecsDistributed) {
  const auto& [size, blocksize, offset_e] = GetParam();
  for (const auto& comm_grid : comm_grids) {
    const Distribution dist(globalSize(size), blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});
    testMatrixSpecs(dist, offset_e);
  }
}

namespace test::SubPanelView {
struct config_t {
  LocalElementSize size;
  TileElementSize blocksize;
  GlobalElementIndex offset;
  SizeType width;

  friend std::ostream& operator<<(std::ostream& os, const config_t& config) {
    return os << config.size << " " << config.blocksize << " " << config.offset << " " << config.width;
  }
};

const std::vector<config_t> configs{
    {{12, 12}, {4, 4}, {2, 2}, 2},
    {{12, 12}, {6, 6}, {2, 2}, 4},
};
}

void testPanelOffset(const Distribution& dist, const GlobalElementIndex& offset_e,
                     const SizeType width) {
  const SubPanelView panel_view(dist, offset_e, width);

  const GlobalTileIndex offset = dist.globalTileIndex(offset_e);
  EXPECT_EQ(panel_view.offset(), offset);
}

void testPanelRange(const Distribution& dist, const GlobalElementIndex& offset_e, const SizeType width) {
  const SubPanelView panel_view(dist, offset_e, width);

  const LocalTileIndex offset_local = {
      dist.template nextLocalTileFromGlobalElement<Coord::Row>(offset_e.row()),
      dist.template nextLocalTileFromGlobalElement<Coord::Col>(offset_e.col()),
  };

  std::list<LocalTileIndex> indices_in_range;
  for (SizeType i = offset_local.row(); i < dist.localNrTiles().rows(); ++i)
    indices_in_range.emplace_back(LocalTileIndex(i, offset_local.col()));

  std::list<LocalTileIndex> indices_from_view;
  for (const auto ij : panel_view.iteratorLocal())
    indices_from_view.emplace_back(ij);

  EXPECT_EQ(indices_in_range, indices_from_view);
}

void testPanelSpecs(const Distribution& dist, const GlobalElementIndex offset_e, const SizeType width) {
  const SubPanelView panel_view(dist, offset_e, width);

  if (dist.rankIndex().col() != dist.rankGlobalElement<Coord::Col>(offset_e.col()))
    return;

  const LocalTileIndex offset_local{
      dist.template nextLocalTileFromGlobalElement<Coord::Row>(offset_e.row()),
      dist.template localTileFromGlobalElement<Coord::Col>(offset_e.col()),
  };

  const SizeType j = offset_local.col();
  for (SizeType i = offset_local.row(); i < dist.localNrTiles().rows(); ++i) {
    const LocalTileIndex ij_local(i, j);
    const GlobalTileIndex ij(dist.globalTileIndex(ij_local));

    SizeType sub_i = 0, sub_j = 0;
    if (ij.row() == panel_view.offset().row())
      sub_i += dist.tileElementIndex(offset_e).row();
    sub_j += dist.tileElementIndex(offset_e).col();

    const TileElementIndex sub_offset(sub_i, sub_j);
    const TileElementSize sub_size = dist.tileSize(ij) - TileElementSize(sub_i, sub_j);

    const matrix::SubTileSpec spec = panel_view(ij_local);

    EXPECT_EQ(spec.size, sub_size);
    EXPECT_EQ(spec.origin, sub_offset);
  }
}

using PanelViewTest = ::testing::TestWithParam<::test::SubPanelView::config_t>;

INSTANTIATE_TEST_SUITE_P(AllConfigs, PanelViewTest, ::testing::ValuesIn(::test::SubPanelView::configs));

TEST_P(PanelViewTest, OffsetLocal) {
  const auto& [size, blocksize, offset_e, width] = GetParam();
  const Distribution dist(size, blocksize);
  testPanelOffset(dist, offset_e, width);
}

TEST_P(PanelViewTest, OffsetDistributed) {
  const auto& [size, blocksize, offset_e, width] = GetParam();
  for (const auto& comm_grid : comm_grids) {
    const Distribution dist(globalSize(size), blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});
    testPanelOffset(dist, offset_e, width);
  }
}

TEST_P(PanelViewTest, RangeLocal) {
  const auto& [size, blocksize, offset_e, width] = GetParam();
  const Distribution dist(size, blocksize);
  testPanelRange(dist, offset_e, width);
}

TEST_P(PanelViewTest, RangeDistributed) {
  const auto& [size, blocksize, offset_e, width] = GetParam();
  for (const auto& comm_grid : comm_grids) {
    const Distribution dist(globalSize(size), blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});
    testPanelRange(dist, offset_e, width);
  }
}

TEST_P(PanelViewTest, SpecsLocal) {
  const auto& [size, blocksize, offset_e, width] = GetParam();
  const Distribution dist(size, blocksize);
  testPanelSpecs(dist, offset_e, width);
}

TEST_P(PanelViewTest, SpecsDistributed) {
  const auto& [size, blocksize, offset_e, width] = GetParam();
  for (const auto& comm_grid : comm_grids) {
    const Distribution dist(globalSize(size), blocksize, comm_grid.size(), comm_grid.rank(), {0, 0});
    testPanelSpecs(dist, offset_e, width);
  }
}
