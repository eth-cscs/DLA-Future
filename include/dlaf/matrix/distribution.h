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
  /// Constructs a distribution for a non distributed matrix of size {0, 0} and block size {1, 1}.
  Distribution() noexcept;

  /// Constructs a distribution for a non distributed matrix of size @p size and block size @p block_size.
  ///
  /// @throw std::invalid_argument if @p !size.isValid().
  /// @throw std::invalid_argument if @p !block_size.isValid() or @p block_size_.isEmpty().
  Distribution(const LocalElementSize& size, const TileElementSize& block_size);

  /// Constructs a distribution for a matrix of size @p size and block size @p block_size,
  /// distributed on a 2D grid of processes of size @p grid_size.
  ///
  /// @param[in] rank_index is the rank of the current process,
  /// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix.
  /// @throw std::invalid_argument if @p !size.isValid().
  /// @throw std::invalid_argument if @p !block_size.isValid() or @p block_size_.isEmpty().
  /// @throw std::invalid_argument if @p !grid_size.isValid() or @p grid_size_.isEmpty().
  /// @throw std::invalid_argument if @p !rank_index.isValid() or @p !rank_index_.isIn(grid_size).
  /// @throw std::invalid_argument if @p !source_rank_index.isValid() or @p !source_rank_index_.isIn(grid_size).
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const comm::Size2D& grid_size, const comm::Index2D& rank_index,
               const comm::Index2D& source_rank_index);

  Distribution(const Distribution& rhs) = default;

  Distribution(Distribution&& rhs) noexcept;

  Distribution& operator=(const Distribution& rhs) = default;

  Distribution& operator=(Distribution&& rhs) noexcept;

  bool operator==(const Distribution& rhs) const noexcept {
    return size_ == rhs.size_ && local_size_ == rhs.local_size_ && block_size_ == rhs.block_size_ &&
           global_nr_tiles_ == rhs.global_nr_tiles_ && local_nr_tiles_ == rhs.local_nr_tiles_ &&
           rank_index_ == rhs.rank_index_ && grid_size_ == rhs.grid_size_ &&
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
    return grid_size_;
  }

  const comm::Index2D& sourceRankIndex() const noexcept {
    return source_rank_index_;
  }

  /// Returns the global 2D index of the element
  /// which has index @p tile_element in the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isValid() and global_tile.isIn(nrTiles())
  /// @pre tile_element.isValid() and tile_element.isIn(blockSize())
  GlobalElementIndex globalElementIndex(const GlobalTileIndex& global_tile,
                                        const TileElementIndex& tile_element) const noexcept {
    DLAF_ASSERT_HEAVY((global_tile.isValid() && global_tile.isIn(global_nr_tiles_)));
    DLAF_ASSERT_HEAVY((tile_element.isValid() && tile_element.isIn(block_size_)));

    return {globalElementFromGlobalTileAndTileElement<Coord::Row>(global_tile.row(), tile_element.row()),
            globalElementFromGlobalTileAndTileElement<Coord::Col>(global_tile.col(),
                                                                  tile_element.col())};
  }

  /// Returns the global 2D index of the tile which contains the element with global index @p global_element.
  ///
  /// @pre global_element.isValid() and global_element.isIn(size())
  GlobalTileIndex globalTileIndex(const GlobalElementIndex& global_element) const noexcept {
    DLAF_ASSERT_HEAVY((global_element.isValid() && global_element.isIn(size_)));

    return {globalTileFromGlobalElement<Coord::Row>(global_element.row()),
            globalTileFromGlobalElement<Coord::Col>(global_element.col())};
  }

  /// Returns the global 2D index of the tile that has index @p local_tile
  /// in the current rank.
  ///
  /// @pre local_tile.isValid() and local_tile.isIn(localNrTiles())
  GlobalTileIndex globalTileIndex(const LocalTileIndex& local_tile) const noexcept {
    DLAF_ASSERT_HEAVY((local_tile.isValid() && local_tile.isIn(local_nr_tiles_)));

    return {globalTileFromLocalTile<Coord::Row>(local_tile.row()),
            globalTileFromLocalTile<Coord::Col>(local_tile.col())};
  }

  /// Returns the 2D rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isValid() and global_tile.isIn(nrTiles())
  comm::Index2D rankGlobalTile(const GlobalTileIndex& global_tile) const noexcept {
    DLAF_ASSERT_HEAVY((global_tile.isValid() && global_tile.isIn(global_nr_tiles_)));

    return {rankGlobalTile<Coord::Row>(global_tile.row()),
            rankGlobalTile<Coord::Col>(global_tile.col())};
  }

  /// Returns the local 2D index in current process of the tile with index @p global_tile.
  ///
  /// @throws std::invalid_argument if the global tile is not stored in the current process.
  /// @pre global_tile.isValid() and global_tile.isIn(nrTiles())
  LocalTileIndex localTileIndex(const GlobalTileIndex& global_tile) const {
    DLAF_ASSERT_HEAVY((global_tile.isValid() && global_tile.isIn(global_nr_tiles_)));

    if (rank_index_ != rankGlobalTile(global_tile)) {
      throw std::invalid_argument("Global tile not available in this rank.");
    }
    return {localTileFromGlobalTile<Coord::Row>(global_tile.row()),
            localTileFromGlobalTile<Coord::Col>(global_tile.col())};
  }

  /// Returns the local index in current process of the global element
  /// whose index is the smallest index larger or equal @p global_element
  /// and which is stored in current process.
  ///
  /// @pre global_element.isValid() and global_element.isIn(size())
  TileElementIndex tileElementIndex(const GlobalElementIndex& global_element) const noexcept {
    DLAF_ASSERT_HEAVY((global_element.isValid() && global_element.isIn(size_)));

    return {tileElementFromGlobalElement<Coord::Row>(global_element.row()),
            tileElementFromGlobalElement<Coord::Col>(global_element.col())};
  }

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nrTiles().get<rc>()
  /// @pre 0 <= tile_element < blockSize.get<rc>()
  template <Coord rc>
  SizeType globalElementFromGlobalTileAndTileElement(SizeType global_tile, SizeType tile_element) const
      noexcept {
    DLAF_ASSERT_HEAVY((0 <= global_tile && global_tile < global_nr_tiles_.get<rc>()));
    DLAF_ASSERT_HEAVY((0 <= tile_element && tile_element < block_size_.get<rc>()));
    return util::matrix::elementFromTileAndTileElement(global_tile, tile_element, block_size_.get<rc>());
  }

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with local index @p local_tile in current process.
  ///
  /// @pre 0 <= local_tile < localNrTiles().get<rc>()
  /// @pre 0 <= tile_element < blockSize.get<rc>()
  template <Coord rc>
  SizeType globalElementFromLocalTileAndTileElement(SizeType local_tile, SizeType tile_element) const
      noexcept {
    return globalElementFromGlobalTileAndTileElement<rc>(globalTileFromLocalTile<rc>(local_tile),
                                                         tile_element);
  }

  /// Returns the rank index of the process that stores the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>()
  template <Coord rc>
  int rankGlobalElement(SizeType global_element) const noexcept {
    return rankGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nrTiles().get<rc>()
  template <Coord rc>
  int rankGlobalTile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY((0 <= global_tile && global_tile < global_nr_tiles_.get<rc>()));
    return util::matrix::rankGlobalTile(global_tile, grid_size_.get<rc>(), source_rank_index_.get<rc>());
  }

  /// Returns the global index of the tile which contains the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>()
  template <Coord rc>
  SizeType globalTileFromGlobalElement(SizeType global_element) const noexcept {
    DLAF_ASSERT_HEAVY((0 <= global_element && global_element < size_.get<rc>()));
    return util::matrix::tileFromElement(global_element, block_size_.get<rc>());
  }

  /// Returns the global index of the tile that has index @p local_tile
  /// in the current rank.
  ///
  /// @pre 0 <= local_tile < localNrTiles().get<rc>()
  template <Coord rc>
  SizeType globalTileFromLocalTile(SizeType local_tile) const noexcept {
    DLAF_ASSERT_HEAVY((0 <= local_tile && local_tile < local_nr_tiles_.get<rc>()));
    return util::matrix::globalTileFromLocalTile(local_tile, grid_size_.get<rc>(), rank_index_.get<rc>(),
                                                 source_rank_index_.get<rc>());
  }

  /// Returns the local index of the tile which contains the element with global index @p global_element.
  ///
  /// If the element with @p global_element index is not by current rank it returns -1.
  /// @pre 0 <= global_element < size().get<rc>()
  template <Coord rc>
  SizeType localTileFromGlobalElement(SizeType global_element) const noexcept {
    return localTileFromGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the local index in current process of the tile with index @p global_tile.
  ///
  /// If the tiles with @p global_tile index is not by current rank it returns -1.
  /// @pre 0 <= global_tile < nrTiles().get<rc>()
  template <Coord rc>
  SizeType localTileFromGlobalTile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY((0 <= global_tile && global_tile < global_nr_tiles_.get<rc>()));
    return util::matrix::localTileFromGlobalTile(global_tile, grid_size_.get<rc>(),
                                                 rank_index_.get<rc>(), source_rank_index_.get<rc>());
  }

  /// Returns the local index in current process of the global tile
  /// whose index is the smallest index larger or equal the index of the global tile
  /// that contains the element with index @p global_element
  /// and which is stored in current process.
  ///
  /// @pre 0 <= global_element < size().get<rc>()
  template <Coord rc>
  SizeType nextLocalTileFromGlobalElement(SizeType global_element) const noexcept {
    return nextLocalTileFromGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the local index in current process of the global tile
  /// whose index is the smallest index larger or equal @p global_tile
  /// and which is stored in current process.
  ///
  /// @pre 0 <= global_tile <= nrTiles().get<rc>()
  template <Coord rc>
  SizeType nextLocalTileFromGlobalTile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY((0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>()));
    return util::matrix::nextLocalTileFromGlobalTile(global_tile, grid_size_.get<rc>(),
                                                     rank_index_.get<rc>(),
                                                     source_rank_index_.get<rc>());
  }

  /// Returns the index within the tile of the global element with index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>()
  template <Coord rc>
  SizeType tileElementFromGlobalElement(SizeType global_element) const noexcept {
    DLAF_ASSERT_HEAVY((0 <= global_element && global_element < size_.get<rc>()));
    return util::matrix::tileElementFromElement(global_element, block_size_.get<rc>());
  }

private:
  /// Computes and sets @p size_.
  ///
  /// @pre local_size.rows() >= 0 and local_size.cols() >= 0.
  /// @pre grid_size.rows() == 1 and grid_size.cols() == 1.
  void computeGlobalSizeForNonDistr(const LocalElementSize& size) noexcept;

  /// computes and sets global_tiles_.
  ///
  /// @pre size.rows() >= 0 and size.cols() >= 0.
  /// @pre block_size.rows() >= 1 and block_size.cols() >= 1.
  void computeGlobalNrTiles(const GlobalElementSize& size, const TileElementSize& block_size) noexcept;

  /// Computes and sets @p global_tiles_, @p local_tiles_ and @p local_size_.
  ///
  /// @pre size.rows() >= 0 and size.cols() >= 0.
  /// @pre block_size.rows() >= 1 and block_size.cols() >= 1.
  /// @pre grid_size.rows() >= 1 and grid_size.cols() >= 1.
  /// @pre rank_index.row() >= 0 and rank_index.col() >= 0.
  /// @pre source_rank_index.row() >= 0 and source_rank_index.col() >= 0.
  void computeGlobalAndLocalNrTilesAndLocalSize(const GlobalElementSize& size,
                                                const TileElementSize& block_size,
                                                const comm::Size2D& grid_size,
                                                const comm::Index2D& rank_index,
                                                const comm::Index2D& source_rank_index) noexcept;

  /// computes and sets @p local_tiles_.
  ///
  /// @pre local_size.rows() >= 0 and local_size.cols() >= 0.
  /// @pre block_size.rows() >= 1 and block_size.cols() >= 1.
  void computeLocalNrTiles(const LocalElementSize& size, const TileElementSize& block_size) noexcept;

  /// Sets default values.
  ///
  /// size_              = {0, 0}
  /// local_size_        = {0, 0}
  /// global_nr_tiles_   = {0, 0}
  /// local_nr_tiles_    = {0, 0}
  /// block_size_        = {1, 1}
  /// rank_index_        = {0, 0}
  /// grid_size_         = {1, 1}
  /// source_rank_index_ = {0, 0}
  void setDefaultSizes() noexcept;

  GlobalElementSize size_;
  LocalElementSize local_size_;
  GlobalTileSize global_nr_tiles_;
  LocalTileSize local_nr_tiles_;
  TileElementSize block_size_;

  comm::Index2D rank_index_;
  comm::Size2D grid_size_;
  comm::Index2D source_rank_index_;
};
}
}
