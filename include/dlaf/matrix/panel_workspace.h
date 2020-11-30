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

#include "dlaf/common/vector.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

template <Coord shape, class T, Device device>
struct PanelWorkspace;

// TODO it works just for tile layout
// TODO the matrix always occupies memory entirely
template <class T, Device device>
struct PanelWorkspace<Coord::Col, const T, device> : public internal::MatrixBase {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  PanelWorkspace(matrix::Distribution distribution) : MatrixBase(distribution), internal_(distribution) {
    DLAF_ASSERT(nrTiles().cols() == 1, nrTiles());
    // TODO assert not distributed
    external_.resize(nrTiles().rows());
    util::set(internal_, [](auto&&) { return 0; });
  }

  virtual ~PanelWorkspace() {
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
  hpx::shared_future<ConstTileT> external(const LocalTileIndex& index) const {
    DLAF_ASSERT(index.isIn(internal_.distribution().localNrTiles()), index);
    return external_[index.row()];
  }

  bool is_masked(const LocalTileIndex& index) const {
    return external(index).valid();
  }

  Matrix<T, device> internal_;
  common::internal::vector<hpx::shared_future<ConstTileT>> external_;
};

template <class T, Device device>
struct PanelWorkspace<Coord::Col, T, device> : public PanelWorkspace<Coord::Col, const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  using PanelWorkspace<Coord::Col, const T, device>::PanelWorkspace;

  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(!is_masked(index), "read-only access", index);
    return internal_(index);
  }

protected:
  using PanelWorkspace<Coord::Col, const T, device>::is_masked;
  using PanelWorkspace<Coord::Col, const T, device>::internal_;
};
}
}
