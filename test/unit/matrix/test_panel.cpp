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

using test_params_t = std::tuple<GlobalElementSize, TileElementSize, GlobalElementIndex>;

std::vector<test_params_t> test_params{
    // test_params_t({5, 10}, {3, 3}, {1, 2}),
    test_params_t({26, 13}, {3, 3}, {1, 2}),
};

struct config_t {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalElementIndex offset;
};

config_t configure(const test_params_t& params) {
  return {std::get<0>(params), std::get<1>(params), std::get<2>(params)};
}

TYPED_TEST(PanelTest, AssignToConstRef) {
  using namespace dlaf;
  using hpx::util::unwrapping;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

      // TODO a good idea would be to set a matrix, than set externals, then change matrix
      // and check that the ref points correctly
      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> panel(dist, {0, 0});
      Panel<Coord::Col, const TypeParam, dlaf::Device::CPU>& ref = panel;

      std::vector<LocalTileIndex> exp_indices;
      for (const auto& idx : panel.iterator()) {
        exp_indices.push_back(idx);
      }

      std::vector<LocalTileIndex> ref_indices;
      for (const auto& idx : ref.iterator()) {
        ref_indices.push_back(idx);
      }

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

TYPED_TEST(PanelTest, IteratorCol) {
  using namespace dlaf;
  using hpx::util::unwrapping;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

      // setup the panel
      const LocalTileSize at_offset(dist.template nextLocalTileFromGlobalTile<Coord::Row>(
                                        cfg.offset.row()),
                                    dist.template nextLocalTileFromGlobalTile<Coord::Col>(
                                        cfg.offset.col()));

      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);

      const auto exp_nrTiles = dist.localNrTiles().rows();

      std::vector<LocalTileIndex> exp_indices;
      exp_indices.reserve(static_cast<size_t>(exp_nrTiles));
      for (auto index = at_offset.rows(); index < exp_nrTiles; ++index) {
        exp_indices.emplace_back(index, 0);
      }

      std::vector<LocalTileIndex> indices;
      indices.reserve(static_cast<size_t>(exp_nrTiles));
      for (const auto& idx : ws_v.iterator()) {
        indices.push_back(idx);
      }

      EXPECT_EQ(exp_indices, indices);
    }
  }
}

TYPED_TEST(PanelTest, IteratorRow) {
  using namespace dlaf;
  using hpx::util::unwrapping;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

      // setup the panel
      const LocalTileSize at_offset(dist.template nextLocalTileFromGlobalTile<Coord::Row>(
                                        cfg.offset.row()),
                                    dist.template nextLocalTileFromGlobalTile<Coord::Col>(
                                        cfg.offset.col()));

      Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);

      const auto exp_nrTiles = dist.localNrTiles().cols();

      std::vector<LocalTileIndex> exp_indices;
      exp_indices.reserve(static_cast<size_t>(exp_nrTiles));
      for (auto index = at_offset.cols(); index < exp_nrTiles; ++index) {
        exp_indices.emplace_back(0, index);
      }

      std::vector<LocalTileIndex> indices;
      indices.reserve(static_cast<size_t>(exp_nrTiles));
      for (const auto& idx : ws_h.iterator()) {
        indices.push_back(idx);
      }

      EXPECT_EQ(exp_indices, indices);
    }
  }
}

TYPED_TEST(PanelTest, AccessCol) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

      // setup the panel
      const LocalTileSize at_offset(dist.template nextLocalTileFromGlobalTile<Coord::Row>(
                                        cfg.offset.row()),
                                    dist.template nextLocalTileFromGlobalTile<Coord::Col>(
                                        cfg.offset.col()));

      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);

      // rw-access
      for (const auto& idx : ws_v.iterator()) {
        ws_v(idx).then(unwrapping(
            [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(idx.row(), 26)); }));
      }

      // ro-access
      for (const auto& idx : ws_v.iterator()) {
        ws_v.read(idx).then(
            unwrapping([idx](auto&& tile) { CHECK_MATRIX_EQ(TypeUtil::element(idx.row(), 26), tile); }));
      }
    }
  }
}

TYPED_TEST(PanelTest, AccessRow) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});

      // setup the panel
      const LocalTileSize at_offset(dist.template nextLocalTileFromGlobalTile<Coord::Row>(
                                        cfg.offset.row()),
                                    dist.template nextLocalTileFromGlobalTile<Coord::Col>(
                                        cfg.offset.col()));

      Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);

      // rw-access
      for (const auto& idx : ws_h.iterator()) {
        ws_h(idx).then(unwrapping(
            [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(idx.col(), 26)); }));
      }

      // ro-access
      for (const auto& idx : ws_h.iterator()) {
        ws_h.read(idx).then(
            unwrapping([idx](auto&& tile) { CHECK_MATRIX_EQ(TypeUtil::element(idx.col(), 26), tile); }));
      }
    }
  }
}

TYPED_TEST(PanelTest, ExternalTilesCol) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
      const auto& dist = matrix.distribution();

      matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.row(), 26); });

      // setup the panel
      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);

      // even in panel, odd linked to matrix first column
      for (const auto& idx : ws_v.iterator()) {
        if (idx.row() % 2 == 0) {
          ws_v(idx).then(unwrapping(
              [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.row(), 13)); }));
        }
        else {
          ws_v.setTile(idx, matrix.read(idx));
        }
      }

      for (const auto& idx : ws_v.iterator()) {
        if (idx.row() % 2 == 0) {
          CHECK_TILE_EQ(TypeUtil::element(-idx.row(), 13), ws_v.read(idx).get());
        }
        else {
          CHECK_TILE_EQ(matrix.read(idx).get(), ws_v.read(idx).get());
        }
      }

      ws_v.reset();

      for (const auto& idx : ws_v.iterator()) {
        if (idx.row() % 2 == 1) {
          ws_v(idx).then(unwrapping(
              [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.row(), 13)); }));
        }
        else {
          ws_v.setTile(idx, matrix.read(idx));
        }
      }

      for (const auto& idx : ws_v.iterator()) {
        if (idx.row() % 2 == 1) {
          CHECK_TILE_EQ(TypeUtil::element(-idx.row(), 13), ws_v.read(idx).get());
        }
        else {
          CHECK_TILE_EQ(matrix.read(idx).get(), ws_v.read(idx).get());
        }
      }
    }
  }
}

TYPED_TEST(PanelTest, ExternalTilesRow) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
      const auto& dist = matrix.distribution();

      matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.row(), 26); });

      // setup the panel
      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);

      // even in panel, odd linked to matrix first row
      for (const auto& idx : ws_h.iterator()) {
        if (idx.col() % 2 == 0) {
          ws_h(idx).then(unwrapping(
              [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.col(), 13)); }));
        }
        else {
          ws_h.setTile(idx, matrix.read(idx));
        }
      }

      for (const auto& idx : ws_h.iterator()) {
        if (idx.col() % 2 == 0) {
          CHECK_TILE_EQ(TypeUtil::element(-idx.col(), 13), ws_h.read(idx).get());
        }
        else {
          CHECK_TILE_EQ(matrix.read(idx).get(), ws_h.read(idx).get());
        }
      }

      ws_h.reset();

      for (const auto& idx : ws_h.iterator()) {
        if (idx.col() % 2 == 1) {
          ws_h(idx).then(unwrapping(
              [idx](auto&& tile) { matrix::test::set(tile, TypeUtil::element(-idx.col(), 13)); }));
        }
        else {
          ws_h.setTile(idx, matrix.read(idx));
        }
      }

      for (const auto& idx : ws_h.iterator()) {
        if (idx.col() % 2 == 1) {
          CHECK_TILE_EQ(TypeUtil::element(-idx.col(), 13), ws_h.read(idx).get());
        }
        else {
          CHECK_TILE_EQ(matrix.read(idx).get(), ws_h.read(idx).get());
        }
      }
    }
  }
}

TYPED_TEST(PanelTest, OffsetCol) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
      const auto& dist = matrix.distribution();

      // setup the panel
      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);

      for (SizeType i = at_offset.rows(); i < matrix.distribution().localNrTiles().rows(); ++i)
        ws_v({i, 0}).get()({0, 0}) = i;

      for (SizeType i = at_offset.rows(); i < matrix.distribution().localNrTiles().rows(); ++i) {
        ws_v.setOffset({i, 0});

        for (SizeType k = i; k < matrix.distribution().localNrTiles().rows(); ++k) {
          const LocalTileIndex idx(k, 0);
          auto& tile = ws_v.read(idx).get();
          EXPECT_EQ(tile({0, 0}), TypeUtil::element(k, 0));
          EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
        }

        for (const auto& idx : ws_v.iterator()) {
          auto& tile = ws_v.read(idx).get();
          EXPECT_EQ(tile({0, 0}), TypeUtil::element(idx.row(), 0));
          EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
        }
      }
    }
  }
}

TYPED_TEST(PanelTest, OffsetRow) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
      const auto& dist = matrix.distribution();

      // setup the panel
      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);

      for (SizeType j = at_offset.cols(); j < matrix.distribution().localNrTiles().cols(); ++j)
        ws_h({0, j}).get()({0, 0}) = j;

      for (SizeType j = at_offset.cols(); j < matrix.distribution().localNrTiles().cols(); ++j) {
        ws_h.setOffset({0, j});

        for (SizeType k = j; k < matrix.distribution().localNrTiles().cols(); ++k) {
          const LocalTileIndex idx(0, k);
          auto& tile = ws_h.read(idx).get();
          EXPECT_EQ(tile({0, 0}), TypeUtil::element(k, 0));
          EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
        }

        for (const auto& idx : ws_h.iterator()) {
          auto& tile = ws_h.read(idx).get();
          EXPECT_EQ(tile({0, 0}), TypeUtil::element(idx.col(), 0));
          EXPECT_EQ(tile.size(), matrix.read(idx).get().size());
        }
      }
    }
  }
}

TYPED_TEST(PanelTest, BroadcastCol) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
      const auto& dist = matrix.distribution();

      matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.row(), 26); });

      // setup the panel
      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);

      // select the last available rank as root rank, i.e. it owns the panel to be broadcasted
      const comm::IndexT_MPI root_col = std::max(0, comm_grid.size().cols() - 1);
      const auto rank_col = dist.rankIndex().col();

      // set all panels
      for (const auto i_w : ws_v.iterator())
        hpx::dataflow(unwrapping([rank_col](auto&& tile) {
                        matrix::test::set(tile, TypeUtil::element(rank_col, 26));
                      }),
                      ws_v(i_w));

      // check that all panels have been set
      for (const auto i_w : ws_v.iterator())
        hpx::dataflow(unwrapping([rank_col](auto&& tile) {
                        CHECK_TILE_EQ(TypeUtil::element(rank_col, 26), tile);
                      }),
                      ws_v.read(i_w));

      // test it!
      common::Pipeline<comm::Communicator> mpi_row_task_chain(comm_grid.rowCommunicator());

      broadcast(executor_mpi, root_col, ws_v, mpi_row_task_chain, comm_grid.size());

      // check all panel are equal on all ranks
      for (const auto i_w : ws_v.iterator())
        hpx::dataflow(unwrapping([root_col](auto&& tile) {
                        CHECK_TILE_EQ(TypeUtil::element(root_col, 26), tile);
                      }),
                      ws_v.read(i_w));
    }
  }
}

TYPED_TEST(PanelTest, BroadcastRow) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids()) {
    for (const auto& params : test_params) {
      const auto cfg = configure(params);

      Matrix<TypeParam, dlaf::Device::CPU> matrix(cfg.sz, cfg.blocksz, comm_grid);
      const auto& dist = matrix.distribution();

      matrix::test::set(matrix, [](const auto& index) { return TypeUtil::element(index.row(), 26); });

      // setup the panel
      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);

      // select the last available rank as root rank, i.e. it owns the panel to be broadcasted
      const comm::IndexT_MPI root_row = std::max(0, comm_grid.size().rows() - 1);
      const auto rank_row = dist.rankIndex().row();

      // set all panels
      for (const auto i_w : ws_h.iterator())
        hpx::dataflow(unwrapping([rank_row](auto&& tile) {
                        matrix::test::set(tile, TypeUtil::element(rank_row, 26));
                      }),
                      ws_h(i_w));

      // check that all panels have been set
      for (const auto i_w : ws_h.iterator())
        hpx::dataflow(unwrapping([rank_row](auto&& tile) {
                        CHECK_TILE_EQ(TypeUtil::element(rank_row, 26), tile);
                      }),
                      ws_h.read(i_w));

      // test it!
      common::Pipeline<comm::Communicator> mpi_col_task_chain(comm_grid.colCommunicator());

      broadcast(executor_mpi, root_row, ws_h, mpi_col_task_chain, comm_grid.size());

      // check all panel are equal on all ranks
      for (const auto i_w : ws_h.iterator())
        hpx::dataflow(unwrapping([root_row](auto&& tile) {
                        CHECK_TILE_EQ(TypeUtil::element(root_row, 26), tile);
                      }),
                      ws_h.read(i_w));
    }
  }
}

std::vector<test_params_t> test_params_bcast_transpose{
    test_params_t({10, 10}, {3, 3}, {1, 1}),
};

config_t configure_bcast_transpose(const test_params_t& params, const CommunicatorGrid&) {
  return {std::get<0>(params), std::get<1>(params), std::get<2>(params)};
}

TYPED_TEST(PanelTest, BroadcastCol2Row) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids()) {
    for (const auto& params : test_params_bcast_transpose) {
      const auto cfg = configure_bcast_transpose(params, comm_grid);

      // TODO use config size
      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});
      const auto rank_col = dist.rankIndex().col();

      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      // TODO It is important to keep the order of initialization to avoid deadlocks!
      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);
      Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);

      for (const auto i_w : ws_v.iterator())
        hpx::dataflow(unwrapping([rank_col](auto&& tile) {
                        matrix::test::set(tile, TypeUtil::element(rank_col, 26));
                      }),
                      ws_v(i_w));

      // test it!
      common::Pipeline<comm::Communicator> row_task_chain(comm_grid.rowCommunicator());
      common::Pipeline<comm::Communicator> col_task_chain(comm_grid.colCommunicator());

      // select a "random" col which will be the source for the data
      const comm::IndexT_MPI owner = comm_grid.size().cols() / 2;

      broadcast(executor_mpi, owner, ws_v, ws_h, row_task_chain, col_task_chain, comm_grid.size());

      // check that all destination row panels got the value from the right rank
      for (const auto i_w : ws_h.iterator()) {
        hpx::dataflow(unwrapping(
                          [owner](auto&& tile) { CHECK_TILE_EQ(TypeUtil::element(owner, 26), tile); }),
                      ws_h.read(i_w));
      }
    }
  }
}

TYPED_TEST(PanelTest, BroadcastRow2Col) {
  using namespace dlaf;
  using hpx::util::unwrapping;
  using TypeUtil = TypeUtilities<TypeParam>;

  comm::Executor executor_mpi;

  for (auto comm_grid : this->commGrids()) {
    for (const auto& params : test_params_bcast_transpose) {
      const auto cfg = configure_bcast_transpose(params, comm_grid);

      // TODO use config size
      const Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});
      const auto rank_row = dist.rankIndex().row();

      const LocalTileSize at_offset{
          dist.template nextLocalTileFromGlobalTile<Coord::Row>(cfg.offset.row()),
          dist.template nextLocalTileFromGlobalTile<Coord::Col>(cfg.offset.col()),
      };

      // TODO It is important to keep the order of initialization to avoid deadlocks!
      Panel<Coord::Row, TypeParam, dlaf::Device::CPU> ws_h(dist, at_offset);
      Panel<Coord::Col, TypeParam, dlaf::Device::CPU> ws_v(dist, at_offset);

      // each row panel is initialized with a value identifying the row of the rank
      for (const auto i_w : ws_h.iterator())
        hpx::dataflow(unwrapping([rank_row](auto&& tile) {
                        matrix::test::set(tile, TypeUtil::element(rank_row, 26));
                      }),
                      ws_h(i_w));

      // test it!
      common::Pipeline<comm::Communicator> col_task_chain(comm_grid.colCommunicator());
      common::Pipeline<comm::Communicator> row_task_chain(comm_grid.rowCommunicator());

      // select a "random" row which will be the source for the data
      const comm::IndexT_MPI owner = comm_grid.size().rows() / 2;

      broadcast(executor_mpi, owner, ws_h, ws_v, row_task_chain, col_task_chain, comm_grid.size());

      // check that all destination column panels got the value from the right rank
      for (const auto i_w : ws_v.iterator()) {
        hpx::dataflow(unwrapping(
                          [owner](auto&& tile) { CHECK_TILE_EQ(TypeUtil::element(owner, 26), tile); }),
                      ws_v.read(i_w));
      }
    }
  }
}
