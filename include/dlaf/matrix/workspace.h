//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/helpers.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {

namespace internal {
template <Coord panel_type>
matrix::Distribution compute_size(const matrix::Distribution& dist_matrix, const LocalTileSize start) {
  const auto mb = dist_matrix.blockSize().rows();
  const auto nb = dist_matrix.blockSize().cols();

  const auto panel_size = [&]() -> LocalElementSize {
    switch (panel_type) {
      case Coord::Col: {
        const auto mat_size = dist_matrix.localNrTiles().rows();
        const auto i_tile = start.rows();
        return {(mat_size - i_tile) * mb, nb};
      }
      case Coord::Row: {
        const auto mat_size = dist_matrix.localNrTiles().cols();
        const auto i_tile = start.cols();
        return {mb, (mat_size - i_tile) * nb};
      }
    }
  }();

  return {panel_size, dist_matrix.blockSize()};
}

}

namespace matrix {

template <Coord panel_type, class T, Device device>
struct Workspace;

// TODO it works just for tile layout
// TODO the matrix always occupies memory entirely
template <Coord panel_type, class T, Device device>
struct Workspace<panel_type, const T, device> : protected internal::MatrixBase {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  virtual ~Workspace() {
    reset();
  }

  // Note:
  // these 2 next methods allows to use iterate_range2d
  auto localNrTiles() const {
    return this->distribution().localNrTiles();
  }

  auto offset() const {
    switch (panel_type) {
      case Coord::Row:
        return LocalTileIndex{0, offset_.cols()};
      case Coord::Col:
        return LocalTileIndex{offset_.rows(), 0};
    }
  }

  auto rankIndex() const {
    return dist_matrix_.rankIndex();
  }

  auto distribution_matrix() const {
    return dist_matrix_;
  }

  void set_tile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> new_tile_fut) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    DLAF_ASSERT(!is_masked(index), "already set to external", index);
    external_[panel_index(index)] = std::move(new_tile_fut);
  }

  // Note:
  // TODO LocalTileIndex must take into account the offset, so to keep association with SoR of the
  // related matrix.
  //
  // You can access tiles by:
  // - read(LocalTileIndex)           LocalTileIndex in the panel
  // - TODO read(SizeType)                 Linear local index (like an array)
  // - TODO read(GlobalTileIndex)     still thinking how it should work
  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    return is_masked(index) ? external_[panel_index(index)] : internal_.read(full_index(index));
  }

  // it is possible to reset the mask for external tiles, so that memory can be easily re-used
  void reset() {
    external_.clear();
  }

protected:
  Workspace(matrix::Distribution dist_matrix, LocalTileSize offset)  // TODO migrate to Global
      : MatrixBase(dlaf::internal::compute_size<panel_type>(dist_matrix, offset)),
        dist_matrix_(dist_matrix), offset_(offset), internal_(distribution()) {
    // TODO assert not distributed

    switch (panel_type) {
      case Coord::Row:
        DLAF_ASSERT(nrTiles().rows() == 1, nrTiles());
        external_.resize(nrTiles().cols());
        break;
      case Coord::Col:
        DLAF_ASSERT(nrTiles().cols() == 1, nrTiles());
        external_.resize(nrTiles().rows());
        break;
    }
    DLAF_ASSERT_MODERATE(internal_.distribution().localNrTiles().linear_size(),
                         external_.size() == external_.size(), internal_.nrTiles(), external_.size());
  }

  SizeType panel_index(const LocalTileIndex& index) const {
    switch (panel_type) {
      case Coord::Row:
        return index.col() - offset_.cols();
      case Coord::Col:
        return index.row() - offset_.rows();
    }
  }

  LocalTileIndex full_index(LocalTileIndex index) const {
    DLAF_ASSERT_MODERATE(index.row() == 0 || index.col() == 0, index);
    index = [&]() -> LocalTileIndex {
      switch (panel_type) {
        case Coord::Row:
          return {0, index.col() - offset_.cols()};
        case Coord::Col:
          return {index.row() - offset_.rows(), 0};
      }
    }();
    DLAF_ASSERT(index.isIn(localNrTiles()), index);
    return index;
  }

  bool is_masked(const LocalTileIndex linear_index) const {
    return external_[panel_index(linear_index)].valid();
  }

  Distribution dist_matrix_;
  LocalTileSize offset_;

  ///> Stores the shared_future
  common::internal::vector<hpx::shared_future<ConstTileT>> external_;
  ///> non-distributed matrix for internally allocated tiles
  Matrix<T, device> internal_;
};

template <Coord panel_type, class T, Device device>
struct Workspace : public Workspace<panel_type, const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  Workspace(matrix::Distribution distribution, LocalTileSize start = {0, 0})
      : Workspace<panel_type, const T, device>(std::move(distribution), std::move(start)) {}

  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(index.isIn(this->dist_matrix_.localNrTiles()), index, this->dist_matrix_.localNrTiles());
    DLAF_ASSERT(!is_masked(index), "read-only access on external tiles", index);
    return internal_(this->full_index(index));
  }

protected:
  using BaseT = Workspace<panel_type, const T, device>;
  using BaseT::is_masked;

  using BaseT::internal_;
};

template <class T, Device device, class BcastDir, Coord panel_type, class PredicateOwner>
void share_panel(BcastDir direction, Workspace<panel_type, T, device>& ws, PredicateOwner&& whos_root,
                 common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using namespace comm::sync;

  const auto rank = [&]() {
    if (std::is_same<decltype(direction), comm::row_wise>::value)
      return ws.rankIndex().col();
    else if (std::is_same<decltype(direction), comm::col_wise>::value)
      return ws.rankIndex().row();
  }();

  for (const auto& index : iterate_range2d(ws.offset(), ws.localNrTiles())) {
    const comm::IndexT_MPI source_rank = whos_root(index).second;
    if (rank == source_rank)
      hpx::dataflow(broadcast_send(direction), ws.read(index), serial_comm());
    else
      hpx::dataflow(broadcast_recv(direction, source_rank), ws(index), serial_comm());
  }
}

template <class T, Device device>
auto transpose(Workspace<Coord::Col, T, device>& ws_col, Workspace<Coord::Row, T, device>& ws_row) {
  DLAF_ASSERT(ws_col.distribution_matrix() == ws_row.distribution_matrix(),
              "they must refer to the same matrix");
  // DLAF_ASSERT(ws_col.offset() == ws_row.offset(), ws_col.offset(), ws_row.offset()); // TODO do I have
  // to check for offset?!

  auto whos_root = [dist = ws_col.distribution_matrix()](const LocalTileIndex index) {
    const auto k = dist.template globalTileFromLocalTile<Coord::Col>(index.col());
    const auto rank_owner = dist.template rankGlobalTile<Coord::Row>(k);
    return std::make_pair(k, rank_owner);
  };

  const auto& dist = ws_col.distribution_matrix();
  for (const auto& index : iterate_range2d(ws_row.offset(), ws_row.localNrTiles())) {
    SizeType k;
    comm::IndexT_MPI rank_owner;

    std::tie(k, rank_owner) = whos_root(index);

    if (dist.rankIndex().row() != rank_owner)
      continue;

    const auto i = dist.template localTileFromGlobalTile<Coord::Row>(k);
    ws_row.set_tile(index, ws_col.read({i, 0}));
  }

  return whos_root;
}
}
}
