//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
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
template <Coord axis, class T, Device D>
struct Panel;

template <Coord axis, class T, Device D>
struct Panel<axis, const T, D> {
  // Note:
  // This specialization acts as base for the RW version of the panel,
  // moreover allows the casting between references (i.e. Panel<const T>& = Panel<T>)

  constexpr static Coord CoordType = axis == Coord::Col ? Coord::Row : Coord::Col;

  using TileType = Tile<T, D>;
  using ConstTileType = Tile<const T, D>;
  using BaseT = Matrix<T, D>;

  Panel(Panel&&) = default;

  /// On destruction, reset the panel
  ///
  /// Resetting the panel implies removing external dependencies
  virtual ~Panel() noexcept {
    reset();
  }

  /// Return an IterableRange2D with a range over all tiles of the panel (considering the offset)
  auto iteratorLocal() const noexcept {
    return common::iterate_range2d(LocalTileIndex(CoordType, rangeStartLocal()),
                                   LocalTileIndex(CoordType, rangeEndLocal(), 1));
  }

  /// Return the rank which this (local) panel belongs to
  auto rankIndex() const noexcept {
    return dist_matrix_.rankIndex();
  }

  /// Return the Distribution of the parent matrix
  auto parentDistribution() const noexcept {
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
  void setTile(const LocalTileIndex& index, hpx::shared_future<ConstTileType> new_tile_fut) {
    DLAF_ASSERT(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());
    DLAF_ASSERT(internal_.count(linearIndex(index)) == 0, "internal tile have been already used", index);
    DLAF_ASSERT(!isExternal(index), "already set to external", index);

    external_[linearIndex(index)] = std::move(new_tile_fut);
  }

  /// Access a Tile of the panel in read-only mode
  ///
  /// This method is very similar to the one available in dlaf::Matrix.
  ///
  /// @p index is in the coordinate system of the matrix which this panel is related to
  hpx::shared_future<ConstTileType> read(const LocalTileIndex& index) {
    DLAF_ASSERT_HEAVY(index.isIn(dist_matrix_.localNrTiles()), index, dist_matrix_.localNrTiles());

    const SizeType internal_linear_idx = linearIndex(index);
    if (isExternal(index)) {
      return external_[internal_linear_idx];
    }
    else {
      internal_.insert(internal_linear_idx);
      return data_.read(fullIndex(index));
    }
  }

  /// Set the panel to enable access to the range of tiles [start, end)
  ///
  /// With respect to the parent matrix.
  ///
  /// @pre this can be called as first operation after construction or after reset()
  /// @pre (just the index relevant for the axis of the panel)
  /// @pre start <= end
  /// @pre panel offset on construction <= start
  /// @pre panel offset on construction <= end
  /// @pre start <= 1 past the panel last tile
  /// @pre end <= 1 past the panel last tile
  void setRange(GlobalTileSize start_idx, GlobalTileSize end_idx) noexcept {
    start_ = start_idx.get(CoordType);
    start_local_ = dist_matrix_.template nextLocalTileFromGlobalTile<CoordType>(start_);

    end_ = end_idx.get(CoordType);
    end_local_ = dist_matrix_.template nextLocalTileFromGlobalTile<CoordType>(end_);

    DLAF_ASSERT(rangeStart() <= rangeEnd(), rangeStart(), rangeEnd());
    DLAF_ASSERT(rangeStartLocal() >= bias_, start_idx, bias_);
    DLAF_ASSERT(rangeEnd() <= dist_matrix_.nrTiles().get(CoordType), end_idx,
                dist_matrix_.nrTiles().get(CoordType));
  }

  /// Change the start boundary of the range of tiles to which the panel allows access to
  ///
  /// With respect to the parent matrix.
  ///
  /// @pre this can be called as first operation after construction or after reset()
  /// @pre (just the index relevant for the axis of the panel)
  /// @pre start <= current end range of the panel
  /// @pre panel offset on construction <= start
  void setRangeStart(GlobalTileSize start_idx) noexcept {
    start_ = start_idx.get(CoordType);
    start_local_ = dist_matrix_.template nextLocalTileFromGlobalTile<CoordType>(start_);

    DLAF_ASSERT(rangeStartLocal() >= bias_ && rangeStart() <= rangeEnd(), rangeStart(), rangeEnd(),
                bias_);
  }

  /// Change the end boundary of the range of tiles to which the panel allows access to
  ///
  /// With respect to the parent matrix.
  ///
  /// @pre this can be called as first operation after construction or after reset()
  /// @pre (just the index relevant for the axis of the panel)
  /// @pre current end range of the panel <= end
  /// @pre end <= 1 past the panel last tile
  void setRangeEnd(GlobalTileSize end_idx) noexcept {
    end_ = end_idx.get(CoordType);
    end_local_ = dist_matrix_.template nextLocalTileFromGlobalTile<CoordType>(end_);

    DLAF_ASSERT(rangeEnd() >= rangeStart() && rangeEnd() <= dist_matrix_.nrTiles().get(CoordType),
                rangeStart(), end_idx, dist_matrix_.nrTiles().get(CoordType));
  }

  /// Return the current start (1D)
  SizeType rangeStart() const noexcept {
    return start_;
  }

  /// Return the current end (1D)
  SizeType rangeEnd() const noexcept {
    return end_;
  }

  /// Return the current start (1D)
  SizeType rangeStartLocal() const noexcept {
    return start_local_;
  }

  /// Return the current end (1D)
  SizeType rangeEndLocal() const noexcept {
    return end_local_;
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
  static LocalElementSize computePanelSize(LocalElementSize size, TileElementSize blocksize,
                                           LocalTileSize start) {
    const auto mb = blocksize.rows();
    const auto nb = blocksize.cols();

    const auto mat_size = size.get(CoordType);
    const auto i_tile = start.get(CoordType);

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
  static Matrix<T, D> setupInternalMatrix(const Distribution& dist, const GlobalTileSize start) {
    constexpr auto CT = CoordType;

    const LocalTileSize start_loc(CT, dist.template nextLocalTileFromGlobalTile<CT>(start.get(CT)));
    const auto panel_size = computePanelSize(dist.localSize(), dist.blockSize(), start_loc);

    Distribution dist_internal{panel_size, dist.blockSize()};
    auto layout = tileLayout(dist_internal);
    return {std::move(dist_internal), layout};
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
  Panel(matrix::Distribution dist_matrix, GlobalTileSize start)
      : dist_matrix_(dist_matrix), data_(setupInternalMatrix(dist_matrix, start)) {
    DLAF_ASSERT_HEAVY(data_.nrTiles().get(axis) == 1, data_.nrTiles());

    bias_ = dist_matrix_.template nextLocalTileFromGlobalTile<CoordType>(start.get(CoordType));

    setRange(start, dist_matrix_.nrTiles());

    external_.resize(data_.nrTiles().get(CoordType));

    DLAF_ASSERT_HEAVY(data_.distribution().localNrTiles().linear_size() == external_.size(),
                      data_.distribution().localNrTiles().linear_size(), external_.size());
  }

  /// Given a matrix index, compute the internal linear index
  SizeType linearIndex(const LocalTileIndex& index) const noexcept {
    const auto idx = index.get(CoordType);

    DLAF_ASSERT_MODERATE(idx >= rangeStartLocal(), idx, rangeStartLocal());

    return idx - bias_;
  }

  /// Given a matrix index, compute the projected internal index
  ///
  /// It is similar to what linear_index does, so it takes into account the @p start_,
  /// but it computes a 2D index instead of a linear one.
  /// The 2D index is the projection of the given index, i.e. in a Panel<Col> the Col for index
  /// will always be 0 (and relatively for a Panel<Row>)
  LocalTileIndex fullIndex(LocalTileIndex index) const {
    index = LocalTileIndex(CoordType, linearIndex(index));

    DLAF_ASSERT_HEAVY(index.isIn(LocalTileSize(CoordType, rangeEndLocal(), 1)), index,
                      LocalTileSize(CoordType, rangeEndLocal(), 1));

    return index;
  }

  /// Given a matrix index, check if the corresponding tile in the panel is external or not
  bool isExternal(const LocalTileIndex idx_matrix) const noexcept {
    return external_[linearIndex(idx_matrix)].valid();
  }

  ///> Parent matrix which this panel is related to
  Distribution dist_matrix_;

  ///> Local matrix used for storing the panel data
  matrix::Matrix<T, D> data_;

  ///> It represents from where it is necessary to allocate memory (fixed at construction time)
  SizeType bias_;
  ///> It represents from where the panel gives access to tiles
  SizeType start_;
  SizeType start_local_;  // local version of @p start_
  ///> It represents the last tile which this panel gives access to
  SizeType end_;
  SizeType end_local_;  // local version of @p end_

  ///> Container for references to external tiles
  common::internal::vector<hpx::shared_future<ConstTileType>> external_;
  ///> Keep track of usage status of internal tiles (accessed or not)
  std::set<SizeType> internal_;
};

template <Coord axis, class T, Device device>
struct Panel : public Panel<axis, const T, device> {
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;

  explicit Panel(matrix::Distribution distribution, GlobalTileSize start = {0, 0})
      : Panel<axis, const T, device>(std::move(distribution), std::move(start)) {}

  /// Access tile at specified index in readwrite mode
  ///
  /// It is possible to access just internal tiles in RW mode.
  ///
  /// @pre index must point to a tile which is internally managed by the panel
  hpx::future<TileType> operator()(const LocalTileIndex& index) {
    DLAF_ASSERT(!BaseT::isExternal(index), "read-only access on external tiles", index);

    BaseT::internal_.insert(BaseT::linearIndex(index));
    return BaseT::data_(BaseT::fullIndex(index));
  }

protected:
  using BaseT = Panel<axis, const T, device>;
};

}
}
