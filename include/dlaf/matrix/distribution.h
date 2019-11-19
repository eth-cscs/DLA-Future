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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/util_distribution.h"
#include "dlaf/util_math.h"

namespace dlaf {
namespace matrix {

/// Distribution contains the information about the size and distribution of a matrix.
/// More details available in misc/matrix_distribution.md.

class Distribution {
public:
  Distribution() noexcept;

  /// Constructs a distribution for a non distributed matrix of size @p size and block size @p block_size.
  ///
  /// @throw std::invalid_argument if size.rows() < 0 or size.cols() < 0.
  /// @throw std::invalid_argument if block_size.rows() < 1 or block_size.cols() < 1.
  Distribution(const LocalElementSize& size, const TileElementSize& block_size);

  /// Constructs a distribution for a matrix of size @p size and block size @p block_size,
  /// distributed on a 2D grid of processes of size @p comm_size.
  ///
  /// @in rank_index is the rank of the current process,
  /// @in source_rank_index is the rank of the process wich contains the top left tile of the matrix.
  /// @throw std::invalid_argument if size.rows() < 0 or size.cols() < 0.
  /// @throw std::invalid_argument if block_size.rows() < 1 or block_size.cols() < 1.
  /// @throw std::invalid_argument if comm_size.rows() < 1 or comm_size.cols() < 1.
  /// @throw std::invalid_argument if rank_index.rows() < 0 or rank_index.cols() < 0.
  /// @throw std::invalid_argument if source_rank_index.rows() < 0 or source_rank_index.cols() < 0.
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const comm::Size2D& comm_size, const comm::Index2D& rank_index,
               const comm::Index2D& source_rank_index);

  Distribution(const Distribution& rhs) = default;

  Distribution(Distribution&& rhs) noexcept;

  Distribution& operator=(const Distribution& rhs) = default;

  Distribution& operator=(Distribution&& rhs) noexcept;

  bool operator==(const Distribution& rhs) const noexcept {
    return size_ == rhs.size_ && local_size_ == rhs.local_size_ && block_size_ == rhs.block_size_ &&
           global_nr_tiles_ == rhs.global_nr_tiles_ && local_nr_tiles_ == rhs.local_nr_tiles_ &&
           rank_index_ == rhs.rank_index_ && comm_size_ == rhs.comm_size_ &&
           source_rank_index_ == rhs.source_rank_index_;
  }

  bool operator!=(const Distribution& rhs) const noexcept {
    return !operator==(rhs);
  }

  const GlobalElementSize& size() const noexcept {
    return size_;
  }

  const LocalElementSize& localSize() const noexcept {
    return local_size_;
  }

  /// @brief Returns the number of tiles of the global matrix (2D size).
  const GlobalTileSize& nrTiles() const noexcept {
    return global_nr_tiles_;
  }

  /// @brief Returns the number of tiles stored locally (2D size).
  const LocalTileSize& localNrTiles() const noexcept {
    return local_nr_tiles_;
  }

  const TileElementSize& blockSize() const noexcept {
    return block_size_;
  }

  const comm::Index2D& rankIndex() const noexcept {
    return rank_index_;
  }

  const comm::Size2D& commGridSize() const noexcept {
    return comm_size_;
  }

  const comm::Index2D& sourceRankIndex() const noexcept {
    return source_rank_index_;
  }

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile <= nrTiles().get<rc>()
  /// @pre 0 <= tile_element <= blockSize.get<rc>()
  template <RowCol rc>
  SizeType globalElementFromGlobalTileAndTileElement(SizeType global_tile, SizeType tile_element) const
      noexcept {
    assert(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>());
    assert(0 <= tile_element && tile_element <= global_nr_tiles_.get<rc>());
    return util::matrix::elementFromTileAndTileElement(global_tile, tile_element, block_size_.get<rc>());
  }

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with local index @p local_tile in current process.
  ///
  /// @pre 0 <= local_tile <= localNrTiles().get<rc>()
  /// @pre 0 <= tile_element <= blockSize.get<rc>()
  template <RowCol rc>
  SizeType globalElementFromLocalTileAndTileElement(SizeType local_tile, SizeType tile_element) const
      noexcept {
    return globalElementFromGlobalTileAndTileElement<rc>(globalTileFromLocalTile<rc>(local_tile),
                                                         tile_element);
  }

  /// Returns the rank index of the process that stores the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element <= size().get<rc>()
  template <RowCol rc>
  int rankGlobalElement(SizeType global_element) const noexcept {
    return rankGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile <= nrTiles().get<rc>()
  template <RowCol rc>
  int rankGlobalTile(SizeType global_tile) const noexcept {
    assert(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>());
    return util::matrix::rankGlobalTile(global_tile, comm_size_.get<rc>(), source_rank_index_.get<rc>());
  }

  /// Returns the global index of the tile which contains the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element <= size().get<rc>()
  template <RowCol rc>
  SizeType globalTileFromGlobalElement(SizeType global_element) const noexcept {
    assert(0 <= global_element && global_element <= size_.get<rc>());
    return util::matrix::tileFromElement(global_element, block_size_.get<rc>());
  }

  /// Returns the global tile index of the tile that has index @p local_tile
  /// in the current rank.
  ///
  /// @pre 0 <= local_tile <= localNrTiles().get<rc>()
  template <RowCol rc>
  SizeType globalTileFromLocalTile(SizeType local_tile) const noexcept {
    assert(0 <= local_tile && local_tile <= local_nr_tiles_.get<rc>());
    return util::matrix::globalTileFromLocalTile(local_tile, comm_size_.get<rc>(), rank_index_.get<rc>(),
                                                 source_rank_index_.get<rc>());
  }

  /// Returns the local index of the tile which contains the element with global index @p global_element.
  ///
  /// If the element with @p global_element index is not by current rank it returns -1.
  /// @pre 0 <= global_element <= size().get<rc>()
  template <RowCol rc>
  SizeType localTileFromGlobalElement(SizeType global_element) const noexcept {
    return localTileFromGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the local tile index in current process of the tile with index @p global_tile.
  ///
  /// If the tiles with @p global_tile index is not by current rank it returns -1.
  /// @pre 0 <= global_tile <= nrTiles().get<rc>()
  template <RowCol rc>
  SizeType localTileFromGlobalTile(SizeType global_tile) const noexcept {
    assert(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>());
    return util::matrix::localTileFromGlobalTile(global_tile, comm_size_.get<rc>(),
                                                 rank_index_.get<rc>(), source_rank_index_.get<rc>());
  }

  /// Returns the local index in current process of the global tile
  /// whose index is the smallest index larger or equal the index of the global tile
  /// that contains the element with index @p global_element
  /// and which is stored in current process.
  ///
  /// @pre 0 <= global_element <= size().get<rc>()
  template <RowCol rc>
  SizeType nextLocalTileFromGlobalElement(SizeType global_element) const noexcept {
    return nextLocalTileFromGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the local index in current process of the global tile
  /// whose index is the smallest index larger or equal @p global_tile
  /// and which is stored in current process.
  ///
  /// @pre 0 <= global_tile <= nrTiles().get<rc>()
  template <RowCol rc>
  SizeType nextLocalTileFromGlobalTile(SizeType global_tile) const noexcept {
    assert(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>());
    return util::matrix::nextLocalTileFromGlobalTile(global_tile, comm_size_.get<rc>(),
                                                     rank_index_.get<rc>(),
                                                     source_rank_index_.get<rc>());
  }

  /// Returns the index within the tile of the global element with index @p global_element.
  ///
  /// @pre 0 <= global_element <= size().get<rc>()
  template <RowCol rc>
  SizeType tileElementFromGlobalElement(SizeType global_element) const noexcept {
    assert(0 <= global_element && global_element <= size_.get<rc>());
    return util::matrix::tileElementFromElement(global_element, block_size_.get<rc>());
  }

  /// Returns the local index in current process of the global element
  /// whose index is the smallest index larger or equal @p global_element
  /// and which is stored in current process.
  ///
  /// @pre 0 <= global_element <= size().get<rc>()
  template <RowCol rc>
  SizeType nextLocalElementFromGlobalElement(SizeType global_element) const noexcept {
    if (rank_index_.get<rc>() == rankGlobalElement<rc>(global_element))
      return util::matrix::elementFromTileAndTileElement(  //
          localTileFromGlobalElement<rc>(global_element),
          tileElementFromGlobalElement<rc>(global_element), block_size_.get<rc>());
    return util::matrix::elementFromTileAndTileElement(  //
        nextLocalTileFromGlobalElement<rc>(global_element), 0, block_size_.get<rc>());
  }

private:
  /// Computes @p local_size_.
  /// @pre size_ is set.
  void computeLocalSize() noexcept {
    auto row = nextLocalElementFromGlobalElement<RowCol::Row>(size_.rows());
    auto col = nextLocalElementFromGlobalElement<RowCol::Col>(size_.cols());
    local_size_ = LocalElementSize(row, col);
  }

  /// Computes @p size_.
  /// @pre To be used only for non distributed matrices, i.e. comm_size_ == {1, 1}.
  /// @pre local_size_ is set.
  void computeGlobalSize() noexcept {
    assert(comm_size_ == comm::Size2D(1, 1));
    assert(rank_index_ == comm::Index2D(0, 0));
    assert(src_rank_index_ == comm::Index2D(0, 0));
    size_ = GlobalElementSize(local_size_.rows(), local_size_.cols());
  }

  /// computes the number of tiles from sizes.
  /// @pre size_ and local_size_ are set.
  void computeLocalGlobalNrTiles() noexcept {
    global_nr_tiles_ = {util::ceilDiv(size_.rows(), block_size_.rows()),
                        util::ceilDiv(size_.cols(), block_size_.cols())};
    local_nr_tiles_ = {util::ceilDiv(local_size_.rows(), block_size_.rows()),
                       util::ceilDiv(local_size_.cols(), block_size_.cols())};
  }

  /// Sets default values.
  ///
  /// size_              = {0, 0}
  /// local_size_        = {0, 0}
  /// global_nr_tiles_   = {0, 0}
  /// local_nr_tiles_    = {0, 0}
  /// block_size_        = {1, 1}
  /// rank_index_        = {0, 0}
  /// comm_size_         = {1, 1}
  /// source_rank_index_ = {0, 0}
  void setDefaultSizes() noexcept;

  GlobalElementSize size_;
  LocalElementSize local_size_;
  GlobalTileSize global_nr_tiles_;
  LocalTileSize local_nr_tiles_;
  TileElementSize block_size_;

  comm::Index2D rank_index_;
  comm::Size2D comm_size_;
  comm::Index2D source_rank_index_;
};
}
}
