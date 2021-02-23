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
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

template <Coord panel_type, class T, Device device>
struct Panel;

/// Panel with read-only access to its tiles
///
/// A row or column panel, strictly related to a given Matrix (from the coords point of view)
/// not really useful, it is just the base for the RW version
template <Coord dir, class T, Device device>
struct Panel<dir, const T, device> : protected Matrix<T, device> {
  // TODO it works just for tile layout
  // TODO the matrix always occupies memory entirely
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;
  using BaseT = Matrix<T, device>;

  virtual ~Panel() noexcept {
    reset();
  }

  Panel(Panel&&) = default;

  auto begin() const noexcept {
    return range_.begin();
  }

  auto end() const noexcept {
    return range_.end();
  }

  auto rankIndex() const noexcept {
    return dist_matrix_.rankIndex();
  }

  auto distribution_matrix() const noexcept {
    return dist_matrix_;
  }

  void set_tile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> new_tile_fut) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    DLAF_ASSERT(internal_.count(panel_index(index)) == 0, "internal tile have been already used", index);
    DLAF_ASSERT(!is_external(index), "already set to external", index);

    external_[panel_index(index)] = std::move(new_tile_fut);
  }

  // index w.r.t. the matrix coordinates system, not in the workspace (so it takes into account the offset)
  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    // DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());

    const SizeType internal_linear_idx = panel_index(index);
    if (is_external(index)) {
      return external_[internal_linear_idx];
    }
    else {
      internal_.insert(internal_linear_idx);
      return BaseT::read(full_index(index));
    }
  }

  // Set the panel to a new offset with respect to the matrix
  void set_offset(LocalTileIndex offset) noexcept {
    DLAF_ASSERT(offset.get(component(dir)) >= bias_, offset, bias_);

    offset_ = offset.get(component(dir)) - bias_;

    range_ = iterate_range2d(LocalTileIndex(component(dir), offset_ + bias_),
                             LocalTileSize(component(dir),
                                           dist_matrix_.localNrTiles().get(component(dir)) -
                                               (offset_ + bias_),
                                           1));
  }

  // it is possible to reset masks, so that memory can be easily re-used
  void reset() noexcept {
    for (auto& e : external_)
      e = {};
    internal_.clear();
  }

protected:
  /// Create the internal matrix used for storing tiles
  ///
  /// It allocates just the memory needed for the part of matrix that it works with,
  /// i.e. starting from `start`, so skippiing the first start tiles
  static Matrix<T, device> setup_matrix(const Distribution& dist_matrix, const LocalTileIndex start) {
    const auto mb = dist_matrix.blockSize().rows();
    const auto nb = dist_matrix.blockSize().cols();

    const auto panel_size = [&]() -> LocalElementSize {
      const auto mat_size = dist_matrix.localSize().get(component(dir));
      const auto i_tile = start.get(component(dir));

      switch (dir) {
        case Coord::Col:
          return {mat_size - i_tile * mb, nb};
        case Coord::Row:
          return {mb, mat_size - i_tile * nb};
      }
    }();

    Distribution dist{panel_size, dist_matrix.blockSize()};
    auto layout = tileLayout(dist);
    return {std::move(dist), layout};
  }

  // TODO think about passing a reference to the matrix instead of the distribution (useful for tilesize)
  Panel(matrix::Distribution dist_matrix, LocalTileIndex offset)
      : BaseT(setup_matrix(dist_matrix, offset)), dist_matrix_(dist_matrix),
        bias_(offset.get(component(dir))), offset_(0),
        range_(iterate_range2d(LocalTileIndex(component(dir), bias_ + offset_),
                               LocalTileSize(component(dir),
                                             dist_matrix_.localNrTiles().get(component(dir)) -
                                                 (bias_ + offset_),
                                             1))) {
    DLAF_ASSERT_MODERATE(BaseT::nrTiles().get(dir) == 1, BaseT::nrTiles());

    external_.resize(BaseT::nrTiles().get(component(dir)));

    DLAF_ASSERT_HEAVY(BaseT::distribution().localNrTiles().linear_size() == external_.size(),
                      BaseT::distribution().localNrTiles().linear_size(), external_.size());
  }

  /// Given a matrix index, compute the internal linear index
  SizeType panel_index(const LocalTileIndex& index) const {
    // TODO check that index is more than offset
    return index.get(component(dir)) - bias_;
  }

  LocalTileIndex full_index(LocalTileIndex index) const {
    // DLAF_ASSERT_MODERATE(index.row() == 0 || index.col() == 0, index);

    index = LocalTileIndex(component(dir), panel_index(index));

    DLAF_ASSERT_MODERATE(index.isIn(BaseT::distribution().localNrTiles()), index,
                         BaseT::distribution().localNrTiles());
    return index;
  }

  /// Given a matrix index, check if the corresponding tile in the panel is external or not
  bool is_external(const LocalTileIndex idx_matrix) const {
    return external_[panel_index(idx_matrix)].valid();
  }

  using iter2d_t = decltype(iterate_range2d(LocalTileIndex{0, 0}, LocalTileSize{0, 0}));

  Distribution dist_matrix_;
  SizeType bias_;
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

  Panel(matrix::Distribution distribution, LocalTileIndex start)
      : Panel<panel_type, const T, device>(std::move(distribution), std::move(start)) {}

  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    // DLAF_ASSERT(index.isIn(BaseT::dist_matrix_.localNrTiles()), index,
    //            BaseT::dist_matrix_.localNrTiles());
    DLAF_ASSERT(!is_external(index), "read-only access on external tiles", index);

    BaseT::internal_.insert(BaseT::panel_index(index));
    return BaseT::operator()(BaseT::full_index(index));
  }

protected:
  using BaseT = Panel<panel_type, const T, device>;
  using BaseT::is_external;
};

namespace internal {

// helper function that identifies the owner of a transposed coordinate,
// it returns both the component of the rank in the transposed dimension and
// its global cross coordinate (i.e. row == col in the global frame of reference)
template <Coord dst_coord>
std::pair<SizeType, comm::IndexT_MPI> transposed_owner(const Distribution& dist,
                                                       const LocalTileIndex idx) {
  const auto idx_cross = dist.template globalTileFromLocalTile<dst_coord>(idx.get(dst_coord));
  const auto rank_owner = dist.template rankGlobalTile<transposed(dst_coord)>(idx_cross);
  return std::make_pair(idx_cross, rank_owner);
}
}

template <class T, Device device, Coord panel_type>
void broadcast(hpx::execution::parallel_executor ex, comm::IndexT_MPI rank_root,
               Panel<panel_type, T, device>& ws, common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using namespace comm::sync::broadcast;
  using hpx::dataflow;

  constexpr auto comm_dir = transposed(panel_type);

  // TODO
  // if (grid_size.get(component(comm_dir)) < 1)
  //  return;

  const auto rank = ws.rankIndex().get(component(comm_dir));

  for (const auto& index : ws) {
    if (rank == rank_root)
      dataflow(ex, comm::send_tile_o, serial_comm(), comm_dir, ws.read(index));
    else
      dataflow(ex, comm::recv_tile_o, serial_comm(), comm_dir, ws(index), rank_root);
  }
}

/// Broadcast
///
/// This communication pattern enables access to the tile in the column panel which shares
/// the same index in the row panel (with transposed coordinate)
///
/// For each tile in the row panel the rank owning the corresponding tile in the column panel
/// is identified, then each tile is either:
///
/// - linked as external tile to the corresponding one in the column panel, if current rank owns it
/// - received from the owning rank, which broadcasts the tile from the row panel along the column
template <class T, Device device, Coord from_dir, Coord to_dir>
void broadcast(hpx::execution::parallel_executor ex, comm::IndexT_MPI rank_root,
               Panel<from_dir, T, device>& ws_from, Panel<to_dir, T, device>& ws_to,
               common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  static_assert(from_dir == transposed(to_dir), "this method broadcasts and transposes coordinates");

  using namespace dlaf::comm::sync::broadcast;
  using hpx::dataflow;

  DLAF_ASSERT(ws_from.distribution_matrix() == ws_to.distribution_matrix(),
              "they must refer to the same matrix");

  // TODO add check about sizes?!
  // DLAF_ASSERT(ws_from.localNrTiles() >= ws_to.localNrTiles(),
  //            "sizes", ws_from.localNrTiles(), ws_to.localNrTiles());

  // TODO do I have to check for offset?!

  constexpr Coord from_coord = component(from_dir);
  constexpr Coord to_coord = component(to_dir);

  // communicate each tile orthogonally to the direction of the destination panel
  constexpr Coord comm_dir = transposed(to_dir);

  // share the main panel, so that it can be used as source for orthogonal communications
  broadcast(ex, rank_root, ws_from, serial_comm);

  const auto& dist = ws_from.distribution_matrix();
  for (const auto& idx_dst : ws_to) {
    SizeType idx_cross;
    comm::IndexT_MPI owner;

    std::tie(idx_cross, owner) = internal::transposed_owner<to_coord>(dist, idx_dst);

    if (dist.rankIndex().get(from_coord) == owner) {
      const auto idx_src = dist.template localTileFromGlobalTile<from_coord>(idx_cross);
      ws_to.set_tile(idx_dst, ws_from.read({from_coord, idx_src}));
      // TODO if (grid_size.get(component(comm_dir)) > 1)
      dataflow(ex, comm::send_tile_o, serial_comm(), comm_dir, ws_to.read(idx_dst));
    }
    else {
      // TODO if (grid_size.get(component(comm_dir)) > 1)
      dataflow(ex, comm::recv_tile_o, serial_comm(), comm_dir, ws_to(idx_dst), owner);
    }
  }
}

}
}
