//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/future.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/helpers.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/panel_workspace.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

template <class T, Device device>
struct VirtualWorkspace;

template <class T, Device device>
struct VirtualWorkspace<const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  struct WorkspaceTileIndex {
    comm::IndexT_MPI row_source;
    Coord where;
    LocalTileIndex index;
  };

  VirtualWorkspace(Distribution matrix, Distribution dist_col, Distribution dist_row,
                   LocalTileSize offset)
      : offset_(offset), dist_matrix_(matrix), column_(dist_col), row_(dist_row) {
    matrix::util::set(row_, [](auto&&) { return 0; });
  }

  LocalTileSize offset() const {
    return offset_;
  }

  Distribution distribution_main() const {
    return dist_matrix_;
  }

  LocalTileSize colNrTiles() const {
    return {(dist_matrix_.localNrTiles() - offset_).rows(), 1};
  }

  // TODO it is dangerous: using it with iterate_range2d, it returns
  // a local index that used directly, returns a tile from the column,
  // not from the row
  LocalTileSize rowNrTiles() const {
    return {1, (dist_matrix_.localNrTiles() - offset_).cols()};
  }

  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    return column_.read(index);
  }

  hpx::shared_future<ConstTileT> read(const GlobalTileIndex& index) {
    return read(index.row());
  }

  hpx::shared_future<ConstTileT> read(const SizeType global_i) {
    const auto ws_index = workspace_index(global_i);
    switch (ws_index.where) {
      case Coord::Col:
        return column_.read(ws_index.index);
      case Coord::Row:
        return row_.read(ws_index.index);
      default:
        return {};
    }
  }

protected:
  WorkspaceTileIndex workspace_index(const SizeType global_i) {
    const auto& rank = dist_matrix_.rankIndex();

    const auto owner_rank_row = dist_matrix_.rankGlobalTile<Coord::Row>(global_i);

    const bool is_owner = owner_rank_row == rank.row();

    SizeType local_i;
    LocalTileIndex local_index;
    if (is_owner) {
      local_i = dist_matrix_.template localTileFromGlobalTile<Coord::Row>(global_i);
      local_index = {local_i - offset_.rows(), 0};
    }
    else {
      DLAF_ASSERT_MODERATE(dist_matrix_.rankGlobalTile<Coord::Col>(global_i) == rank.col(),
                           dist_matrix_.rankGlobalTile<Coord::Col>(global_i), rank.col());
      local_i = dist_matrix_.template localTileFromGlobalTile<Coord::Col>(global_i);
      local_index = {0, local_i - offset_.cols()};
    }

    DLAF_ASSERT_HEAVY(local_index.isValid(), local_index);

    return {owner_rank_row, is_owner ? Coord::Col : Coord::Row, local_index};
  }

  const LocalTileSize offset_;
  const matrix::Distribution dist_matrix_;

  PanelWorkspace<Coord::Col, T, device> column_;
  Matrix<T, device> row_;
};

template <class T, Device device>
struct VirtualWorkspace : public VirtualWorkspace<const T, device> {
protected:
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  using VirtualWorkspace<const T, device>::column_;
  using VirtualWorkspace<const T, device>::row_;
  using VirtualWorkspace<const T, device>::workspace_index;

public:
  using VirtualWorkspace<const T, device>::VirtualWorkspace;

  void set_tile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> fut) {
    column_.set_tile(index, std::move(fut));
  }

  hpx::future<TileT> operator()(const LocalTileIndex index) {
    return column_(index);
  }

  hpx::future<TileT> operator()(const GlobalTileIndex index) {
    return *this(index.row());
  }

  hpx::future<TileT> operator()(const SizeType global_i) {
    const auto ws_index = workspace_index(global_i);
    switch (ws_index.where) {
      case Coord::Col:
        return column_(ws_index.index);
      case Coord::Row:
        return row_(ws_index.index);
      default:
        return {};
    }
  }
};

/// Given:
/// - a workspace and
/// - the rank column where the workspace has the source values
template <class T, class SerialComm>
void populate(comm::row_wise rowwise, VirtualWorkspace<T, Device::CPU>& workspace,
              comm::IndexT_MPI source_col, SerialComm&& serial_comm) {
  using common::iterate_range2d;
  using namespace comm;
  using namespace comm::sync;

  const auto& dist = workspace.distribution_main();
  const auto offset = workspace.offset();

  for (const auto& index_row : iterate_range2d(workspace.colNrTiles())) {
    const auto global_row =
        dist.template globalTileFromLocalTile<Coord::Row>((index_row + offset).row());
    if (source_col == dist.rankIndex().col())
      hpx::dataflow(broadcast_send(rowwise), workspace.read(global_row), serial_comm());
    else
      hpx::dataflow(broadcast_recv(rowwise, source_col), workspace(global_row), serial_comm());
  }
}

/// Given the workspace
template <class T, class SerialComm>
void populate(comm::col_wise colwise, VirtualWorkspace<T, Device::CPU>& workspace,
              SerialComm&& serial_comm) {
  using common::iterate_range2d;
  using namespace comm;
  using namespace comm::sync;

  const auto& dist = workspace.distribution_main();
  const auto offset = workspace.offset();

  for (const auto& index_col : iterate_range2d(workspace.rowNrTiles())) {
    const auto global_row =
        dist.template globalTileFromLocalTile<Coord::Col>((index_col + offset).col());

    const auto source_row = dist.template rankGlobalTile<Coord::Row>(global_row);

    if (source_row == dist.rankIndex().row())
      hpx::dataflow(broadcast_send(colwise), workspace.read(global_row), serial_comm());
    else
      hpx::dataflow(broadcast_recv(colwise, source_row), workspace(global_row), serial_comm());
  }
}
}
}
