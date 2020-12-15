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
namespace matrix {

template <Coord panel_type, class T, Device device>
struct Panel;

// TODO it works just for tile layout
// TODO the matrix always occupies memory entirely
template <Coord panel_type, class T, Device device>
struct Panel<panel_type, const T, device> : protected Matrix<T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;
  using BaseT = Matrix<T, device>;

  virtual ~Panel() {
    reset();
  }

  auto begin() const {
    return range_.begin();
  }

  auto end() const {
    return range_.end();
  }

  auto rankIndex() const {
    return dist_matrix_.rankIndex();
  }

  auto distribution_matrix() const {
    return dist_matrix_;
  }

  void set_tile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> new_tile_fut) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    DLAF_ASSERT(internal_.count(panel_index(index)) == 0, "internal tile have been already used", index);
    DLAF_ASSERT(!is_masked(index), "already set to external", index);

    external_[panel_index(index)] = std::move(new_tile_fut);
  }

  // index w.r.t. the matrix coordinates system, not in the workspace (so it takes into account the offset)
  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    if (is_masked(index)) {
      return external_[panel_index(index)];
    }
    else {
      internal_.insert(panel_index(index));
      return BaseT::read(full_index(index));
    }
  }

  // it is possible to reset masks, so that memory can be easily re-used
  void reset() {
    external_.clear();
    internal_.clear();
  }

protected:
  static Distribution compute_size(const Distribution& dist_matrix, const LocalTileSize start) {
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

  // TODO think about passing a reference to the matrix instead of the distribution (useful for tilesize)
  Panel(matrix::Distribution dist_matrix,
            LocalTileSize offset)  // TODO migrate to index? don't know...
      : BaseT(compute_size(dist_matrix, offset)), dist_matrix_(dist_matrix),
        offset_(Coord::Col == panel_type ? offset.rows() : offset.cols()),
        range_(iterate_range2d(Coord::Col == panel_type ? LocalTileIndex{offset_, 0}
                                                        : LocalTileIndex{0, offset_},
                               BaseT::distribution().localNrTiles())) {
    util::set(*((Matrix<T, device>*) this),
              [](auto&&) { return 0; });  // TODO remove this and enable util::set

    switch (panel_type) {
      case Coord::Row:
        DLAF_ASSERT(BaseT::nrTiles().rows() == 1, BaseT::nrTiles());
        external_.resize(BaseT::nrTiles().cols());
        break;
      case Coord::Col:
        DLAF_ASSERT(BaseT::nrTiles().cols() == 1, BaseT::nrTiles());
        external_.resize(BaseT::nrTiles().rows());
        break;
    }

    DLAF_ASSERT_MODERATE(BaseT::distribution().localNrTiles().linear_size() == external_.size(),
                         BaseT::distribution().localNrTiles().linear_size(), external_.size());
  }

  SizeType panel_index(const LocalTileIndex& index) const {
    switch (panel_type) {
      case Coord::Row:
        return index.col() - offset_;
      case Coord::Col:
        return index.row() - offset_;
    }
  }

  LocalTileIndex full_index(LocalTileIndex index) const {
    DLAF_ASSERT_MODERATE(index.row() == 0 || index.col() == 0, index);

    index = [&]() -> LocalTileIndex {
      switch (panel_type) {
        case Coord::Row:
          return {0, index.col() - offset_};
        case Coord::Col:
          return {index.row() - offset_, 0};
      }
    }();

    DLAF_ASSERT(index.isIn(BaseT::distribution().localNrTiles()), index);
    return index;
  }

  bool is_masked(const LocalTileIndex linear_index) const {
    return external_[panel_index(linear_index)].valid();
  }

  using iter2d_t = decltype(iterate_range2d(LocalTileIndex{0, 0}, LocalTileSize{0, 0}));

  Distribution dist_matrix_;
  SizeType offset_;
  iter2d_t range_;

  ///> Stores the shared_future
  common::internal::vector<hpx::shared_future<ConstTileT>> external_;

  std::set<SizeType> internal_;
};

template <Coord panel_type, class T, Device device>
struct Panel : public Panel<panel_type, const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  Panel(matrix::Distribution distribution, LocalTileSize start = {0, 0})
      : Panel<panel_type, const T, device>(std::move(distribution), std::move(start)) {}

  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(index.isIn(BaseT::dist_matrix_.localNrTiles()), index,
                BaseT::dist_matrix_.localNrTiles());
    DLAF_ASSERT(!is_masked(index), "read-only access on external tiles", index);

    BaseT::internal_.insert(BaseT::panel_index(index));
    return BaseT::operator()(BaseT::full_index(index));
  }

protected:
  using BaseT = Panel<panel_type, const T, device>;
  using BaseT::is_masked;
};

template <class T, Device device, class BcastDir, Coord panel_type, class PredicateOwner>
void share_panel(BcastDir direction, Panel<panel_type, T, device>& ws, PredicateOwner&& whos_root,
                 common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using namespace comm::sync;

  const auto rank = [&]() {
    if (std::is_same<decltype(direction), comm::row_wise>::value)
      return ws.rankIndex().col();
    else if (std::is_same<decltype(direction), comm::col_wise>::value)
      return ws.rankIndex().row();
  }();

  for (const auto& index : ws) {
    const comm::IndexT_MPI source_rank = whos_root(index).second;
    if (rank == source_rank)
      hpx::dataflow(broadcast_send(direction), ws.read(index), serial_comm());
    else
      hpx::dataflow(broadcast_recv(direction, source_rank), ws(index), serial_comm());
  }
}

template <class T, Device device>
auto transpose(Panel<Coord::Col, T, device>& ws_col, Panel<Coord::Row, T, device>& ws_row) {
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
  for (const auto& index : ws_row) {
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
