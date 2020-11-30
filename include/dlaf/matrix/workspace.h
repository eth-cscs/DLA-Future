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
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

template <Coord panel_type, class T, Device device>
struct Workspace;

// TODO it works just for tile layout
// TODO the matrix always occupies memory entirely
template <Coord panel_type, class T, Device device>
struct Workspace<panel_type, const T, device> : public internal::MatrixBase {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  virtual ~Workspace() {
    reset();
  }

  void set_tile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> new_tile_fut) {
    DLAF_ASSERT(!is_masked(index), "you cannot set it again");
    external_[index.row()] = std::move(new_tile_fut);
  }

  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    return is_masked(index) ? external(index) : internal_.read(index);
  }

  void reset() {
    external_.clear();
  }

protected:
  static Distribution compute_size(const Distribution& dist_matrix,
      const LocalTileSize start = {0, 0}) {
    //DLAF_ASSERT(start.isIn(dist_matrix.nrTiles()), start, dist_matrix.nrTiles());

    const bool is_colpanel = panel_type == Coord::Col;

    const Coord index_type = is_colpanel ? Coord::Row : Coord::Col;
    const auto mat_size = is_colpanel ? dist_matrix.localNrTiles().rows() : dist_matrix.localNrTiles().cols();
    const auto i_tile = is_colpanel ? start.rows() : start.cols();

    const auto i_local = dist_matrix.template nextLocalTileFromGlobalTile<index_type>(i_tile);
    const auto panel_length = mat_size - i_local;

    const auto mb = dist_matrix.blockSize().rows();
    const auto nb = dist_matrix.blockSize().cols();

    const auto panel_size = is_colpanel ? LocalElementSize{panel_length * mb, nb} : LocalElementSize{mb, panel_length * nb};

    return Distribution(panel_size,
                        dist_matrix.blockSize());  // TODO transpose size for row panel
  }

  // Note
  // TODO Row/Col, Matrix Distribution
  // If with workspace we identify just a panel of tiles, we can simply
  // have row/col workspace and internally compute sizes from the matrix
  // distribution to which they apply to
  //
  // TODO sometimes we don't want to have a full matrix panel (e.g. when
  // working with submatrix, like with trailing matrix)
  Workspace(matrix::Distribution dist_matrix, LocalTileSize offset) // TODO migrate to Global
      : MatrixBase(compute_size(std::move(dist_matrix), offset)), internal_(distribution()) {
    // TODO assert not distributed
    DLAF_ASSERT(nrTiles().cols() == 1 || nrTiles().rows() == 1, nrTiles());

    external_.resize(nrTiles().rows());
  }

  hpx::shared_future<ConstTileT> external(const LocalTileIndex& index) const {
    DLAF_ASSERT(index.isIn(distribution().localNrTiles()), index);
    return external_[index.row()];
  }

  bool is_masked(const LocalTileIndex& index) const {
    return external(index).valid();
  }

  ///> Stores the shared_future
  common::internal::vector<hpx::shared_future<ConstTileT>> external_;
  ///> non-distributed matrix for internally allocated tiles
  Matrix<T, device> internal_;
};

template <Coord panel_type, class T, Device device>
struct Workspace : public Workspace<panel_type, const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  Workspace(matrix::Distribution distribution, LocalTileSize start) : Workspace<panel_type, const T, device>(std::move(distribution), std::move(start)) {
    util::set(internal_, [](auto&&) { return 0; });
  }

  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(!is_masked(index), "read-only access", index);
    return internal_(index);
  }

protected:
  using BaseT = Workspace<panel_type, const T, device>;
  using BaseT::external;
  using BaseT::is_masked;

  using BaseT::internal_;
};
}
}
