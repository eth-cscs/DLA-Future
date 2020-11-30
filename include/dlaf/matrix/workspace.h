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

#include "dlaf/common/vector.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace matrix {

template <class T, Device device>
struct Workspace;

// TODO it works just for tile layout
// TODO the matrix always occupies memory entirely
template <class T, Device device>
struct Workspace<const T, device> : internal::MatrixBase {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  Workspace(matrix::Distribution distribution) : MatrixBase(distribution) {
    // TODO assert not distributed
    DLAF_ASSERT(nrTiles().cols() == 1 || nrTiles().rows() == 1, nrTiles());

    external_.resize(nrTiles().rows());
  }

  virtual ~Workspace() {
    reset();
  }

  void set_tile(const LocalTileIndex& index, hpx::shared_future<ConstTileT> new_tile_fut) {
    DLAF_ASSERT(!is_masked(index), "you cannot set it again");
    external_[index.row()] = std::move(new_tile_fut);
  }

  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    if (is_masked(index))
      return external(index);
  }

  void reset() {
    external_.clear();
  }

protected:
  hpx::shared_future<ConstTileT> external(const LocalTileIndex& index) const {
    DLAF_ASSERT(index.isIn(distribution().localNrTiles()), index);
    return external_[index.row()];
  }

  bool is_masked(const LocalTileIndex& index) const {
    return external(index).valid();
  }

  ///> Stores the shared_future
  common::internal::vector<hpx::shared_future<ConstTileT>> external_;
};

template <class T, Device device>
struct Workspace : public Workspace<const T, device> {
  using TileT = Tile<T, device>;
  using ConstTileT = Tile<const T, device>;

  Workspace(matrix::Distribution distribution)
      : Workspace<const T, device>(distribution), internal_(distribution) {
    util::set(internal_, [](auto&&) { return 0; });
  }

  hpx::future<TileT> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(!is_masked(index), "read-only access", index);
    return internal_(index);
  }

  hpx::shared_future<ConstTileT> read(const LocalTileIndex& index) {
    return is_masked(index) ? external(index) : internal_.read(index);
  }

protected:
  using Workspace<const T, device>::external;
  using Workspace<const T, device>::is_masked;

  ///> non-distributed matrix for internally allocated tiles
  Matrix<T, device> internal_;
};
}
}
