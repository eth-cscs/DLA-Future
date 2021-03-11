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

/// Panel (1D array of tiles)
///
/// 1D array of tiles, i.e. a Row or Column panel strictly related to a given dlaf::Matrix (from the
/// coords point of view)
template <Coord axis, class T, Device device>
struct Panel;

template <Coord axis, class T, Device device>
struct Panel<axis, const T, device> : protected Matrix<T, device> {
  // Note:
  // This specialization acts as base for the RW version of the panel,
  // moreover allows the casting between references (i.e. Panel<const T>& = Panel<T>)

  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;
  using BaseT = Matrix<T, device>;

  Panel(Panel&&) = default;

  virtual ~Panel() noexcept {
    reset();
  }

  // Note:
  // begin() and end() allows to loop over the tiles of the panel with a for range loop.
  //
  // for (auto& idx_tile : panel) {
  //   panel.read(idx_tile);
  // }

  /// Return an IteratorRange2D pointing at the first tile part of the panel
  auto begin() const noexcept {
    return range_.begin();
  }

  /// Return an IteratorRange2D pointing just after the last tile in the panel
  auto end() const noexcept {
    return range_.end();
  }

  /// Return the rank which this (local) panel belongs to
  auto rankIndex() const noexcept {
    return dist_matrix_.rankIndex();
  }

  /// Return the Distribution of the parent matrix
  auto parent_distribution() const noexcept {
    return dist_matrix_;
  }

  /// Set a specific index to point to the specified external tile
  ///
  /// It is possible to set to an external tile on an index if, since last reset() or from
  /// the creation of the panel, the specific index:
  /// - has not been accessed, neither on read or read/write
  /// - has not been already set to an external tile
  ///
  /// @pre @p index must be a valid index for the current panel size
  void set_tile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> new_tile_fut) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    DLAF_ASSERT(internal_.count(linear_index(index)) == 0, "internal tile have been already used",
                index);
    DLAF_ASSERT(!is_external(index), "already set to external", index);

    external_[linear_index(index)] = std::move(new_tile_fut);
  }

  /// Access a Tile of the panel in read-only mode
  ///
  /// This method is very similar to the one available in dlaf::Matrix.
  ///
  /// @p index is in the coordinate system of the matrix which this panel is related to
  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    DLAF_ASSERT_HEAVY(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());

    const SizeType internal_linear_idx = linear_index(index);
    if (is_external(index)) {
      return external_[internal_linear_idx];
    }
    else {
      internal_.insert(internal_linear_idx);
      return BaseT::read(full_index(index));
    }
  }

  /// Set the panel to a new offset (with respect to the "parent" matrix)
  ///
  /// @pre offset cannot be less than the offset has been specifed on construction
  void set_offset(LocalTileSize offset, bool reset_values = false) noexcept {
    DLAF_ASSERT(offset.get(component(axis)) >= bias_, offset, bias_);

    offset_ = offset.get(component(axis)) - bias_;

    const LocalTileIndex panel_start(component(axis), offset_ + bias_);
    const LocalTileSize panel_size(component(axis),
                                   dist_matrix_.localNrTiles().get(component(axis)) - (offset_ + bias_),
                                   1);
    range_ = iterate_range2d(panel_start, panel_size);

    // TODO It would be enough to set to zero just the part of matrix used (considering offset)
    if (reset_values)
      util::set(*(static_cast<Matrix<T, device>*>(this)), [](auto&&) { return 0; });
  }

  /// Reset the internal usage status of the panel.
  ///
  /// In particular:
  /// - usage status of each tile is reset
  /// - external tiles references are dropped and internal ones are set back
  void reset() noexcept {
    for (auto& e : external_)
      e = {};
    internal_.clear();
  }

protected:
  using iter2d_t = decltype(iterate_range2d(LocalTileSize{0, 0}));

  static LocalElementSize compute_panel_size(LocalElementSize size, TileElementSize blocksize,
                                             LocalTileSize start) {
    const auto mb = blocksize.rows();
    const auto nb = blocksize.cols();

    const auto mat_size = size.get(component(axis));
    const auto i_tile = start.get(component(axis));

    switch (axis) {
      case Coord::Col:
        return {mat_size - i_tile * mb, nb};
      case Coord::Row:
        return {mb, mat_size - i_tile * nb};
    }
  }

  /// Create the internal matrix, with tile layout, used for storing tiles
  ///
  /// It allocates just the memory needed for the part of matrix used, so
  /// starting from @p start
  static Matrix<T, device> setup_matrix(const Distribution& dist_matrix, const LocalTileSize start) {
    const auto panel_size = compute_panel_size(dist_matrix.localSize(), dist_matrix.blockSize(), start);

    Distribution dist{panel_size, dist_matrix.blockSize()};
    auto layout = tileLayout(dist);
    return {std::move(dist), layout};
  }

  /// Create a Panel related to the Matrix passed as parameter.
  ///
  /// The Panel is strictly related to its parent dlaf::Matrix.
  /// In particular, it will create a Row or Column with the same size of its parent matrix (local),
  /// considering the specified offset from the top left origin.
  ///
  /// e.g. a 4x5 matrix with an offset 2x1 will have either:
  /// - a Panel<Col> 2x1
  /// - or a Panel<Row> 4x1
  Panel(matrix::Distribution dist_matrix, LocalTileSize offset)
      : BaseT(setup_matrix(dist_matrix, offset)), dist_matrix_(dist_matrix),
        bias_(offset.get(component(axis))), range_(iterate_range2d(LocalTileSize{0, 0})) {
    DLAF_ASSERT_HEAVY(BaseT::nrTiles().get(axis) == 1, BaseT::nrTiles());

    set_offset(offset);

    external_.resize(BaseT::nrTiles().get(component(axis)));

    DLAF_ASSERT_HEAVY(BaseT::distribution().localNrTiles().linear_size() == external_.size(),
                      BaseT::distribution().localNrTiles().linear_size(), external_.size());
  }

  /// Given a matrix index, compute the internal linear index
  SizeType linear_index(const LocalTileIndex& index) const noexcept {
    const auto idx = index.get(component(axis));

    DLAF_ASSERT_MODERATE(idx >= offset_, idx, offset_);

    return idx - bias_;
  }

  /// Given a matrix index, compute the projected internal index
  ///
  /// It is similar to what linear_index does, so it takes into account the @p offset_,
  /// but it computes a 2D index instead of a linear one.
  /// The 2D index is the projection of the given index, i.e. in a Panel<Col> the Col for index
  /// will always be 0 (and relatively for a Panel<Row>)
  LocalTileIndex full_index(LocalTileIndex index) const {
    index = LocalTileIndex(component(axis), linear_index(index));

    DLAF_ASSERT_HEAVY(index.isIn(BaseT::distribution().localNrTiles()), index,
                      BaseT::distribution().localNrTiles());

    return index;
  }

  /// Given a matrix index, check if the corresponding tile in the panel is external or not
  bool is_external(const LocalTileIndex idx_matrix) const noexcept {
    return external_[linear_index(idx_matrix)].valid();
  }

  ///> Parent matrix which this panel is related to
  Distribution dist_matrix_;

  ///> It represents from where it is necessary to allocate memory
  SizeType bias_;
  ///> It represents from where the panel gives access to tiles
  SizeType offset_;
  ///> Interanlly store the range of tiles accessible (changes according to @p offset_)
  iter2d_t range_;

  ///> Container for references to external tiles
  common::internal::vector<hpx::shared_future<ConstTileT>> external_;
  ///> Keep track of usage status of internal tiles (accessed or not)
  std::set<SizeType> internal_;
};

template <Coord axis, class T, Device device>
struct Panel : public Panel<axis, const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  explicit Panel(matrix::Distribution distribution, LocalTileSize start = {0, 0})
      : Panel<axis, const T, device>(std::move(distribution), std::move(start)) {}

  /// Access tile at specified index in readwrite mode
  ///
  /// It is possible to access just internal tiles in RW mode.
  ///
  /// @pre index must point to a tile which is internally managed by the panel
  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(!is_external(index), "read-only access on external tiles", index);

    BaseT::internal_.insert(BaseT::linear_index(index));
    return BaseT::operator()(BaseT::full_index(index));
  }

protected:
  using BaseT = Panel<axis, const T, device>;
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
  const auto rank_owner = dist.template rankGlobalTile<orthogonal(dst_coord)>(idx_cross);
  return std::make_pair(idx_cross, rank_owner);
}

}

/// Broadcast a panel in the direction orthogonal to its axis
template <class T, Device device, Coord axis>
void broadcast(hpx::execution::parallel_executor ex, comm::IndexT_MPI rank_root,
               Panel<axis, T, device>& ws, common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using namespace comm::sync::broadcast;
  using hpx::dataflow;

  constexpr auto comm_dir = orthogonal(axis);

  // TODO
  // if (grid_size.get(component(comm_dir)) < 1)
  //  return;

  const auto rank = ws.rankIndex().get(component(comm_dir));

  for (const auto& index : ws) {
    if (rank == rank_root)
      dataflow(ex, comm::sendTile_o, serial_comm(), comm_dir, ws.read(index));
    else
      dataflow(ex, comm::recvTile_o, serial_comm(), comm_dir, ws(index), rank_root);
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
template <class T, Device device, Coord axis_from, Coord axis_to>
void broadcast(hpx::execution::parallel_executor ex, comm::IndexT_MPI rank_root,
               Panel<axis_from, T, device>& ws_from, Panel<axis_to, T, device>& ws_to,
               common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  static_assert(axis_from == orthogonal(axis_to), "this method broadcasts and transposes coordinates");

  using namespace dlaf::comm::sync::broadcast;
  using hpx::dataflow;

  DLAF_ASSERT(ws_from.parent_distribution() == ws_to.parent_distribution(),
              "they must refer to the same matrix");

  // TODO add check about sizes?!
  // DLAF_ASSERT(ws_from.localNrTiles() >= ws_to.localNrTiles(),
  //            "sizes", ws_from.localNrTiles(), ws_to.localNrTiles());

  // TODO do I have to check for offset?!

  constexpr Coord coord_from = component(axis_from);
  constexpr Coord coord_to = component(axis_to);

  // communicate each tile orthogonally to the direction of the destination panel
  constexpr Coord comm_dir = orthogonal(axis_to);

  // share the main panel, so that it can be used as source for orthogonal communications
  broadcast(ex, rank_root, ws_from, serial_comm);

  const auto& dist = ws_from.parent_distribution();
  for (const auto& idx_dst : ws_to) {
    SizeType idx_cross;
    comm::IndexT_MPI owner;

    std::tie(idx_cross, owner) = internal::transposed_owner<coord_to>(dist, idx_dst);

    if (dist.rankIndex().get(coord_from) == owner) {
      const auto idx_src = dist.template localTileFromGlobalTile<coord_from>(idx_cross);
      ws_to.set_tile(idx_dst, ws_from.read({coord_from, idx_src}));
      // TODO if (grid_size.get(component(comm_dir)) > 1)
      dataflow(ex, comm::sendTile_o, serial_comm(), comm_dir, ws_to.read(idx_dst));
    }
    else {
      // TODO if (grid_size.get(component(comm_dir)) > 1)
      dataflow(ex, comm::recvTile_o, serial_comm(), comm_dir, ws_to(idx_dst), owner);
    }
  }
}

}
}
