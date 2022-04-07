//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/types.h"

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

  constexpr static Coord coord = orthogonal(axis);

  using TileType = Tile<T, D>;
  using ConstTileType = Tile<const T, D>;
  using ElementType = const T;
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
    return common::iterate_range2d(LocalTileIndex(coord, rangeStartLocal()),
                                   LocalTileIndex(coord, rangeEndLocal(), 1));
  }

  /// Return the rank which this (local) panel belongs to
  auto rankIndex() const noexcept {
    return dist_matrix_.rankIndex();
  }

  /// Return Distribution used for construction
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
  void setTile(const LocalTileIndex& index, pika::shared_future<ConstTileType> new_tile_fut) {
    DLAF_ASSERT(internal_.count(linearIndex(index)) == 0, "internal tile have been already used", index);
    DLAF_ASSERT(!isExternal(index), "already set to external", index);
    // Note assertion on index done by linearIndex method.

    has_been_used_ = true;

#if defined DLAF_ASSERT_MODERATE_ENABLE
    {
      namespace ex = pika::execution::experimental;

      const auto panel_tile_size = tileSize(index);
      auto assert_tile_size = pika::unwrapping([panel_tile_size](ConstTileType const& tile) {
        DLAF_ASSERT_MODERATE(panel_tile_size == tile.size(), panel_tile_size, tile.size());
      });
      ex::keep_future(new_tile_fut) | ex::then(std::move(assert_tile_size)) | ex::start_detached();
    }
#endif

    external_[linearIndex(index)] = std::move(new_tile_fut);
  }

  /// Access a Tile of the panel in read-only mode
  ///
  /// This method is very similar to the one available in dlaf::Matrix.
  ///
  /// @pre @p index must be a valid index for the current panel size
  pika::shared_future<ConstTileType> read(const LocalTileIndex& index) {
    // Note assertion on index done by linearIndex method.

    has_been_used_ = true;

    const SizeType internal_linear_idx = linearIndex(index);
    if (isExternal(index)) {
      return external_[internal_linear_idx];
    }
    else {
      internal_.insert(internal_linear_idx);
      auto tile = data_.read(fullIndex(index));

      if (dim_ < 0 && (isFirstGlobalTile(index) && isFirstGlobalTileFull()))
        return tile;
      else
        return splitTile(tile, {{0, 0}, tileSize(index)});
    }
  }

  auto read_sender(const LocalTileIndex& index) {
    return pika::execution::experimental::keep_future(read(index));
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
  void setRange(GlobalTileIndex start_idx, GlobalTileIndex end_idx) noexcept {
    DLAF_ASSERT_MODERATE(!hasBeenUsed(), hasBeenUsed());

    start_ = start_idx.get(coord);
    start_local_ = dist_matrix_.template nextLocalTileFromGlobalTile<coord>(start_);
    offset_element_ = start_ * dist_matrix_.blockSize().template get<coord>();

    end_ = end_idx.get(coord);
    end_local_ = dist_matrix_.template nextLocalTileFromGlobalTile<coord>(end_);

    DLAF_ASSERT(rangeStart() <= rangeEnd(), rangeStart(), rangeEnd());
    DLAF_ASSERT(rangeStartLocal() >= bias_, start_idx, bias_);
    DLAF_ASSERT(rangeEnd() <= dist_matrix_.nrTiles().get(coord), end_idx,
                dist_matrix_.nrTiles().get(coord));
  }

  /// Change the start boundary of the range of tiles to which the panel allows access to
  ///
  /// With respect to the parent matrix.
  ///
  /// @pre this can be called as first operation after construction or after reset()
  /// @pre (just the index relevant for the axis of the panel)
  /// @pre start <= current end range of the panel
  /// @pre panel offset on construction <= start
  void setRangeStart(const GlobalTileIndex& start_idx) noexcept {
    DLAF_ASSERT_MODERATE(!hasBeenUsed(), hasBeenUsed());

    start_ = start_idx.get(coord);
    start_local_ = dist_matrix_.nextLocalTileFromGlobalTile<coord>(start_);
    offset_element_ = start_ * dist_matrix_.blockSize().template get<coord>();

    DLAF_ASSERT(rangeStartLocal() >= bias_ && rangeStart() <= rangeEnd(), rangeStart(), rangeEnd(),
                bias_);
  }

  void setRangeStart(const GlobalElementIndex& start) noexcept {
    DLAF_ASSERT_MODERATE(!hasBeenUsed(), hasBeenUsed());

    start_ = dist_matrix_.globalTileFromGlobalElement<coord>(start.get(coord));
    start_local_ = dist_matrix_.nextLocalTileFromGlobalTile<coord>(start_);
    offset_element_ = start.get<coord>();

    const bool has_first_global_tile =
        dist_matrix_.rankGlobalTile<coord>(start_) == dist_matrix_.rankIndex().get(coord);
    if (has_first_global_tile)
      start_offset_ = dist_matrix_.tileElementFromGlobalElement<coord>(start.get(coord));

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
  void setRangeEnd(GlobalTileIndex end_idx) noexcept {
    DLAF_ASSERT_MODERATE(!hasBeenUsed(), hasBeenUsed());

    end_ = end_idx.get(coord);
    end_local_ = dist_matrix_.nextLocalTileFromGlobalTile<coord>(end_);

    DLAF_ASSERT(rangeEnd() >= rangeStart() && rangeEnd() <= dist_matrix_.nrTiles().get(coord),
                rangeStart(), end_idx, dist_matrix_.nrTiles().get(coord));
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

  /// Returns
  /// the global 1D index of the first row of the panel (axis == Coord::Col)
  /// the global 1D index of the first column of the panel (axis == Coord::Row)
  SizeType offsetElement() const noexcept {
    return offset_element_;
  }

  /// Set the width of the col panel.
  ///
  /// By default the width of the panel is parentDistribution().block_size().cols().
  /// This method allows to reduce this value.
  ///
  /// @pre this can be called as first operation after range setting.
  /// @pre @param 0 < width <= parentDistribution().block_size().cols()
  template <Coord A = axis, std::enable_if_t<A == axis && Coord::Col == axis, int> = 0>
  void setWidth(SizeType width) noexcept {
    DLAF_ASSERT_MODERATE(!hasBeenUsed(), hasBeenUsed());
    DLAF_ASSERT(width > 0, width);
    DLAF_ASSERT(width <= parentDistribution().blockSize().cols(), width,
                parentDistribution().blockSize().cols());

    dim_ = width;
  }

  /// Set the height of the row panel.
  ///
  /// By default the height of the panel is parentDistribution().block_size().rows().
  /// This method allows to reduce this value.
  ///
  /// @pre this can be called as first operation after range setting.
  /// @pre @param 0 < height <= parentDistribution().block_size().rows()
  template <Coord A = axis, std::enable_if_t<A == axis && Coord::Row == axis, int> = 0>
  void setHeight(SizeType height) noexcept {
    DLAF_ASSERT_MODERATE(!hasBeenUsed(), hasBeenUsed());
    DLAF_ASSERT(height > 0, height);
    DLAF_ASSERT(height <= parentDistribution().blockSize().rows(), height,
                parentDistribution().blockSize().rows());

    dim_ = height;
  }

  /// Get the current width of the col panel.
  template <Coord A = axis, std::enable_if_t<A == axis && Coord::Col == axis, int> = 0>
  SizeType getWidth() const noexcept {
    return dim_ < 0 ? dist_matrix_.blockSize().template get<axis>() : dim_;
  }

  /// Get the current height of the row panel.
  template <Coord A = axis, std::enable_if_t<A == axis && Coord::Row == axis, int> = 0>
  SizeType getHeight() noexcept {
    return dim_ < 0 ? dist_matrix_.blockSize().template get<axis>() : dim_;
  }

  /// Reset the internal usage status of the panel.
  ///
  /// In particular:
  /// - usage status of each tile is reset
  /// - external tiles references are dropped and internal ones are set back
  /// - The width (Col Panel) or the height (Row panel) are reset.
  void reset() noexcept {
    for (auto& e : external_)
      e = {};
    internal_.clear();
    dim_ = -1;
    has_been_used_ = false;
  }

protected:
  bool isFirstGlobalTileFull() const {
    return start_offset_ == 0;
  }

  bool isFirstGlobalTile(const LocalTileIndex& index) const {
    const bool rank_has_first_global_tile =
        dist_matrix_.rankGlobalTile<coord>(start_) == dist_matrix_.rankIndex().get(coord);
    return rank_has_first_global_tile && (start_local_ == index.get(coord));
  }

  TileElementSize tileSize(const LocalTileIndex& index) const {
    // Transform to global panel index.
    const auto panel_coord = dist_matrix_.globalTileFromLocalTile<coord>(index.get<coord>());
    const GlobalTileIndex panel_index(coord, panel_coord);

    const bool is_first_global_tile = isFirstGlobalTile(index);
    const auto size_coord = dist_matrix_.tileSize(panel_index).template get<coord>() -
                            (is_first_global_tile ? start_offset_ : 0);
    const auto size_axis = dim_ < 0 ? dist_matrix_.blockSize().template get<axis>() : dim_;

    return {axis, size_axis, size_coord};
  }

  static LocalElementSize computePanelSize(LocalElementSize size, TileElementSize blocksize,
                                           LocalTileIndex start) {
    const auto mb = blocksize.rows();
    const auto nb = blocksize.cols();

    const auto mat_size = size.get(coord);
    const auto i_tile = start.get(coord);

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
  static Matrix<T, D> setupInternalMatrix(const Distribution& dist, const GlobalTileIndex start) {
    constexpr auto CT = coord;

    const LocalTileIndex start_loc(CT, dist.template nextLocalTileFromGlobalTile<CT>(start.get(CT)));
    const auto panel_size = computePanelSize(dist.localSize(), dist.blockSize(), start_loc);

    Distribution dist_internal{panel_size, dist.blockSize()};
    auto layout = tileLayout(dist_internal);
    return {std::move(dist_internal), layout};
  }

  /// Create a Panel related to Matrix represented by given Distribution.
  ///
  /// The Panel is strictly related to its parent Matrix via its @p dist_matrix.
  /// In particular, it will be created as a Row or Column (1st axis) with:
  /// - 1st axis size, same as @p dist_matrix, taking into account @p start offset
  /// - 2nd axis size, same as blocksize (on the same axis) of @p dist_matrix
  ///
  /// e.g.
  /// A (38, 15) matrix is distributed over (4, 5) tiles, with blocksize (10, 3).
  /// Using above distribution, together with a (2, 1) start offset results in either:
  /// - Panel<Col>: (18,  3) with (2, 1) tiles,
  /// - Panel<Row>: (10, 12) with (1, 4) tiles,
  Panel(matrix::Distribution dist_matrix, GlobalTileIndex start)
      : dist_matrix_(dist_matrix), data_(setupInternalMatrix(dist_matrix, start)) {
    DLAF_ASSERT_HEAVY(data_.nrTiles().get(axis) == 1, data_.nrTiles());

    bias_ = dist_matrix_.template nextLocalTileFromGlobalTile<coord>(start.get(coord));

    setRange(start, indexFromOrigin(dist_matrix_.nrTiles()));

    external_.resize(data_.nrTiles().get(coord));

    DLAF_ASSERT_HEAVY(data_.distribution().localNrTiles().linear_size() == external_.size(),
                      data_.distribution().localNrTiles().linear_size(), external_.size());
  }

  /// Given a matrix index, compute the internal linear index
  SizeType linearIndex(const LocalTileIndex& index) const noexcept {
    const auto idx = index.get(coord);
    DLAF_ASSERT_MODERATE(rangeStartLocal() <= idx && idx < rangeEndLocal(), idx, rangeStartLocal(),
                         rangeEndLocal());

    return idx - bias_;
  }

  /// Given a matrix index, compute the projected internal index
  ///
  /// It is similar to what linear_index does, so it takes into account the @p start_,
  /// but it computes a 2D index instead of a linear one.
  /// The 2D index is the projection of the given index, i.e. in a Panel<Col> the Col for index
  /// will always be 0 (and relatively for a Panel<Row>)
  LocalTileIndex fullIndex(LocalTileIndex index) const {
    index = LocalTileIndex(coord, linearIndex(index));

    return index;
  }

  /// Given a matrix index, check if the corresponding tile in the panel is external or not
  bool isExternal(const LocalTileIndex idx_matrix) const noexcept {
    return external_[linearIndex(idx_matrix)].valid();
  }

  bool hasBeenUsed() const noexcept {
    return has_been_used_;
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
  ///> It represent the width or height of the panel. Negatives means not set, i.e. block_size.
  SizeType dim_ = -1;

  ///> It represents the offset to use in first global tile
  SizeType start_offset_ = 0;

  SizeType offset_element_ = 0;

  bool has_been_used_ = false;

  ///> Container for references to external tiles
  common::internal::vector<pika::shared_future<ConstTileType>> external_;
  ///> Keep track of usage status of internal tiles (accessed or not)
  std::set<SizeType> internal_;
};

template <Coord axis, class T, Device device>
struct Panel : public Panel<axis, const T, device> {
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;
  using ElementType = T;

  explicit Panel(matrix::Distribution distribution, GlobalTileIndex start = {0, 0})
      : Panel<axis, const T, device>(std::move(distribution), std::move(start)) {}

  /// Access tile at specified index in readwrite mode
  ///
  /// It is possible to access just internal tiles in RW mode.
  ///
  /// @pre index must point to a tile which is internally managed by the panel
  pika::future<TileType> operator()(const LocalTileIndex& index) {
    // Note assertion on index done by linearIndex method.
    DLAF_ASSERT(!BaseT::isExternal(index), "read-write access not allowed on external tiles", index);

    has_been_used_ = true;

    BaseT::internal_.insert(BaseT::linearIndex(index));
    auto tile = BaseT::data_(BaseT::fullIndex(index));
    if (dim_ < 0 && (isFirstGlobalTile(index) && isFirstGlobalTileFull()))
      return tile;
    else
      return splitTile(tile, {{0, 0}, tileSize(index)});
  }

  auto readwrite_sender(const LocalTileIndex& index) {
    // Note: do not use `keep_future`, otherwise dlaf::transform will not handle the lifetime correctly
    return this->operator()(index);
  }

protected:
  using BaseT = Panel<axis, const T, device>;
  using BaseT::dim_;
  using BaseT::has_been_used_;
  using BaseT::tileSize;
  using BaseT::isFirstGlobalTile;
  using BaseT::isFirstGlobalTileFull;
};
}
}
