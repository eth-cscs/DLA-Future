//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
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
///
/// More details available in misc/matrix_distribution.md.

class Distribution {
public:
  /// Constructs a distribution for a non distributed matrix of size {0, 0} and block size {1, 1}.
  Distribution() noexcept;

  /// Constructs a distribution for a non distributed matrix of size @p size and block size @p block_size.
  ///
  /// @pre size.isValid(),
  /// @pre !block_size.isEmpty().
  Distribution(const LocalElementSize& size, const TileElementSize& block_size);

  /// Constructs a distribution for a matrix of size @p size and block size @p block_size,
  /// distributed on a 2D grid of processes of size @p grid_size.
  ///
  /// @param[in] rank_index is the rank of the current process,
  /// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
  /// @pre size.isValid(),
  /// @pre !tile_size.isEmpty(),
  /// @pre !grid_size.isEmpty(),
  /// @pre rank_index.isIn(grid_size),
  /// @pre source_rank_index.isIn(grid_size).
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const comm::Size2D& grid_size, const comm::Index2D& rank_index,
               const comm::Index2D& source_rank_index);

  /// Constructs a distribution for a matrix of size @p size
  /// distributed on a 2D grid of processes of size @p grid_size.
  /// i.e. multiple tiles per distribution blocks are allowed.
  ///
  /// @param[in] rank_index is the rank of the current process,
  /// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
  /// @pre size.isValid(),
  /// @pre !block_size.isEmpty(),
  /// @pre !tile_size.isEmpty(),
  /// @pre block_size is divisible by tile_size,
  /// @pre !grid_size.isEmpty(),
  /// @pre rank_index.isIn(grid_size),
  /// @pre source_rank_index.isIn(grid_size).
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const TileElementSize& tile_size, const comm::Size2D& grid_size,
               const comm::Index2D& rank_index, const comm::Index2D& source_rank_index);

  Distribution(const Distribution& rhs) = default;

  Distribution(Distribution&& rhs) noexcept;

  Distribution& operator=(const Distribution& rhs) = default;

  Distribution& operator=(Distribution&& rhs) noexcept;

  bool operator==(const Distribution& rhs) const noexcept {
    return size_ == rhs.size_ && local_size_ == rhs.local_size_ && tile_size_ == rhs.tile_size_ &&
           block_size_ == rhs.block_size_ && global_nr_tiles_ == rhs.global_nr_tiles_ &&
           local_nr_tiles_ == rhs.local_nr_tiles_ && rank_index_ == rhs.rank_index_ &&
           grid_size_ == rhs.grid_size_ && source_rank_index_ == rhs.source_rank_index_;
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

  /// Returns the number of tiles of the global matrix (2D size).
  const GlobalTileSize& nrTiles() const noexcept {
    return global_nr_tiles_;
  }

  /// Returns the number of tiles stored locally (2D size).
  const LocalTileSize& localNrTiles() const noexcept {
    return local_nr_tiles_;
  }

  const TileElementSize& blockSize() const noexcept {
    return block_size_;
  }

  const TileElementSize& baseTileSize() const noexcept {
    return tile_size_;
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

  /// Returns the global 2D index of the element.
  /// which has index @p tile_element in the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isIn(nrTiles()),
  /// @pre tile_element.isIn(blockSize()).
  GlobalElementIndex globalElementIndex(const GlobalTileIndex& global_tile,
                                        const TileElementIndex& tile_element) const noexcept {
    DLAF_ASSERT_HEAVY(global_tile.isIn(global_nr_tiles_), global_tile, global_nr_tiles_);
    DLAF_ASSERT_HEAVY(tile_element.isIn(tile_size_), tile_element, tile_size_);

    return {globalElementFromGlobalTileAndTileElement<Coord::Row>(global_tile.row(), tile_element.row()),
            globalElementFromGlobalTileAndTileElement<Coord::Col>(global_tile.col(),
                                                                  tile_element.col())};
  }

  /// Returns the global 2D index of the tile which contains the element with global index @p global_element.
  ///
  /// @pre global_element.isIn(size()).
  GlobalTileIndex globalTileIndex(const GlobalElementIndex& global_element) const noexcept {
    DLAF_ASSERT_HEAVY(global_element.isIn(size_), global_element, size_);

    return {globalTileFromGlobalElement<Coord::Row>(global_element.row()),
            globalTileFromGlobalElement<Coord::Col>(global_element.col())};
  }

  /// Returns the global 2D index of the tile that has index @p local_tile
  /// in the current rank.
  ///
  /// @pre local_tile.isIn(localNrTiles()).
  GlobalTileIndex globalTileIndex(const LocalTileIndex& local_tile) const noexcept {
    DLAF_ASSERT_HEAVY(local_tile.isIn(local_nr_tiles_), local_tile, local_nr_tiles_);

    return {globalTileFromLocalTile<Coord::Row>(local_tile.row()),
            globalTileFromLocalTile<Coord::Col>(local_tile.col())};
  }

  /// Returns the 2D rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isIn(nrTiles()).
  comm::Index2D rankGlobalTile(const GlobalTileIndex& global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(global_tile.isIn(global_nr_tiles_), global_tile, global_nr_tiles_);

    return {rankGlobalTile<Coord::Row>(global_tile.row()),
            rankGlobalTile<Coord::Col>(global_tile.col())};
  }

  /// Returns the local 2D index in current process of the tile with index @p global_tile.
  ///
  /// @pre global_tile.isIn(nrTiles()),
  /// @pre rank_index == rankGlobalTile(global_tile).
  LocalTileIndex localTileIndex(const GlobalTileIndex& global_tile) const {
    DLAF_ASSERT_HEAVY(global_tile.isIn(global_nr_tiles_), global_tile, global_nr_tiles_);

    DLAF_ASSERT(rank_index_ == rankGlobalTile(global_tile), rank_index_, rankGlobalTile(global_tile));
    return {localTileFromGlobalTile<Coord::Row>(global_tile.row()),
            localTileFromGlobalTile<Coord::Col>(global_tile.col())};
  }

  /// Returns the local index in current process of the global element
  /// whose index is the smallest index larger or equal @p global_element
  /// and which is stored in current process.
  ///
  /// @pre global_element.isIn(size()).
  TileElementIndex tileElementIndex(const GlobalElementIndex& global_element) const noexcept {
    DLAF_ASSERT_HEAVY(global_element.isIn(size_), global_element, size_);

    return {tileElementFromGlobalElement<Coord::Row>(global_element.row()),
            tileElementFromGlobalElement<Coord::Col>(global_element.col())};
  }

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nrTiles().get<rc>(),
  /// @pre 0 <= tile_element < blockSize.get<rc>().
  template <Coord rc>
  SizeType globalElementFromGlobalTileAndTileElement(SizeType global_tile,
                                                     SizeType tile_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile < global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    DLAF_ASSERT_HEAVY(0 <= tile_element && tile_element < tile_size_.get<rc>(), tile_element,
                      tile_size_.get<rc>());
    return util::matrix::elementFromTileAndTileElement(global_tile, tile_element, tile_size_.get<rc>());
  }

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with local index @p local_tile in current process.
  ///
  /// @pre 0 <= local_tile < localNrTiles().get<rc>(),
  /// @pre 0 <= tile_element < blockSize.get<rc>().
  template <Coord rc>
  SizeType globalElementFromLocalTileAndTileElement(SizeType local_tile,
                                                    SizeType tile_element) const noexcept {
    return globalElementFromGlobalTileAndTileElement<rc>(globalTileFromLocalTile<rc>(local_tile),
                                                         tile_element);
  }

  /// Returns the rank index of the process that stores the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  int rankGlobalElement(SizeType global_element) const noexcept {
    return rankGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nrTiles().get<rc>().
  template <Coord rc>
  int rankGlobalTile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile < global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    return util::matrix::rankGlobalTile(global_tile, tilesPerBlock<rc>(), grid_size_.get<rc>(),
                                        source_rank_index_.get<rc>());
  }

  /// Returns the global index of the tile which contains the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType globalTileFromGlobalElement(SizeType global_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_element && global_element < size_.get<rc>(), global_element, size_);
    return util::matrix::tileFromElement(global_element, tile_size_.get<rc>());
  }

  /// Returns the global index of the tile that has index @p local_tile
  /// in the current rank.
  ///
  /// @pre 0 <= local_tile < localNrTiles().get<rc>().
  template <Coord rc>
  SizeType globalTileFromLocalTile(SizeType local_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= local_tile && local_tile < local_nr_tiles_.get<rc>(), local_tile,
                      local_nr_tiles_.get<rc>());
    return util::matrix::globalTileFromLocalTile(local_tile, tilesPerBlock<rc>(), grid_size_.get<rc>(),
                                                 rank_index_.get<rc>(), source_rank_index_.get<rc>());
  }

  /// Returns the local index of the tile which contains the element with global index @p global_element.
  ///
  /// If the element with @p global_element index is not by current rank it returns -1.
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType localTileFromGlobalElement(SizeType global_element) const noexcept {
    return localTileFromGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the local index in current process of the tile with index @p global_tile.
  ///
  /// If the tiles with @p global_tile index is not by current rank it returns -1.
  /// @pre 0 <= global_tile < nrTiles().get<rc>().
  template <Coord rc>
  SizeType localTileFromGlobalTile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile < global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    return util::matrix::localTileFromGlobalTile(global_tile, tilesPerBlock<rc>(), grid_size_.get<rc>(),
                                                 rank_index_.get<rc>(), source_rank_index_.get<rc>());
  }

  /// Returns the local index in current process of the global tile
  /// whose index is the smallest index larger or equal the index of the global tile
  /// that contains the element with index @p global_element
  /// and which is stored in current process.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType nextLocalTileFromGlobalElement(SizeType global_element) const noexcept {
    return nextLocalTileFromGlobalTile<rc>(globalTileFromGlobalElement<rc>(global_element));
  }

  /// Returns the local index in current process of the global tile
  /// whose index is the smallest index larger or equal @p global_tile
  /// and which is stored in current process. If there is no such tile
  /// index, the local tile grid size along @rc is returned.
  ///
  /// @pre 0 <= global_tile <= nrTiles().get<rc>().
  template <Coord rc>
  SizeType nextLocalTileFromGlobalTile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    return util::matrix::nextLocalTileFromGlobalTile(global_tile, tilesPerBlock<rc>(),
                                                     grid_size_.get<rc>(), rank_index_.get<rc>(),
                                                     source_rank_index_.get<rc>());
  }

  /// Returns the index within the tile of the global element with index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType tileElementFromGlobalElement(SizeType global_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_element && global_element < size_.get<rc>(), global_element,
                      size_.get<rc>());
    return util::matrix::tileElementFromElement(global_element, tile_size_.get<rc>());
  }

  template <Coord rc>
  SizeType tileSize(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    SizeType n = size_.get<rc>();
    SizeType nb = tile_size_.get<rc>();
    return std::min(nb, n - global_tile * nb);
  }

  /// Returns the size of the Tile with global index @p index.
  TileElementSize tileSize(const GlobalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nrTiles()), index, nrTiles());
    return {tileSize<Coord::Row>(index.row()), tileSize<Coord::Col>(index.col())};
  }

  /// Returns the size of the tile that contains @p i_gl along the @p rc coordinate.
  template <Coord rc>
  SizeType tileSizeFromGlobalElement(SizeType i_gl) const noexcept {
    SizeType n = size_.get<rc>();
    DLAF_ASSERT_HEAVY(0 <= i_gl && i_gl < n, i_gl, n);
    SizeType tile_n = tile_size_.get<rc>();
    SizeType tile_i = util::matrix::tileFromElement(i_gl, tile_n);
    return std::min(tile_n, n - tile_i * tile_n);
  }

  /// Returns the distance from the global index @p i_gl to the tile adjacent the one containing @p i_gl
  /// along @p rc coordinate.
  template <Coord rc>
  SizeType distanceToAdjacentTile(SizeType i_gl) const noexcept {
    return tileSizeFromGlobalElement<rc>(i_gl) - tileElementFromGlobalElement<rc>(i_gl);
  }

  /// Returns a local linear column-major index of the tile that contains @p ij
  SizeType localTileLinearIndex(LocalTileIndex ij) const noexcept {
    return ij.row() + ij.col() * local_nr_tiles_.rows();
  }

  /// Returns a global linear column-major index of the tile that contains @p i_gl
  SizeType globalTileLinearIndex(GlobalElementIndex i_gl) const noexcept {
    GlobalTileIndex tile_i = globalTileIndex(i_gl);
    return tile_i.row() + tile_i.col() * global_nr_tiles_.rows();
  }

  /// Returns the global element distance between tiles along the @p rc coordinate
  template <Coord rc>
  SizeType globalTileElementDistance(SizeType i_begin, SizeType i_end) const noexcept {
    DLAF_ASSERT_HEAVY(i_begin <= i_end, i_begin, i_end);

    const SizeType el_begin = globalElementFromGlobalTileAndTileElement<rc>(i_begin, 0);

    if (i_end == nrTiles().get<rc>())
      return size().get<rc>() - el_begin;

    return globalElementFromGlobalTileAndTileElement<rc>(i_end, 0) - el_begin;
  }

  GlobalElementSize globalTileElementDistance(GlobalTileIndex begin,
                                              GlobalTileIndex end) const noexcept {
    return GlobalElementSize{globalTileElementDistance<Coord::Row>(begin.row(), end.row()),
                             globalTileElementDistance<Coord::Col>(begin.col(), end.col())

    };
  }

  /// Returns the local element size of the region between global tile indices @p i_begin and @p i_end
  /// along the @p rc coordinate
  template <Coord rc>
  SizeType localElementDistanceFromGlobalTile(SizeType i_begin, SizeType i_end) const noexcept {
    DLAF_ASSERT_HEAVY(i_begin <= i_end, i_begin, i_end);
    DLAF_ASSERT_HEAVY(0 <= i_begin && i_end <= global_nr_tiles_.get<rc>(), i_begin, i_end,
                      global_nr_tiles_.get<rc>());
    DLAF_ASSERT(tilesPerBlock<rc>() == 1, "Multi Tile distribution block is not supported yet");

    // Note the second assert is already done by the following calls.
    SizeType i_loc_begin = nextLocalTileFromGlobalTile<rc>(i_begin);
    SizeType i_loc_last = nextLocalTileFromGlobalTile<rc>(i_end) - 1;
    if (i_loc_begin > i_loc_last)
      return 0;
    SizeType l = local_size_.get<rc>();
    SizeType nb = tile_size_.get<rc>();
    SizeType nbr = std::min(nb, l - i_loc_last * nb);  // size of last local tile along `rc`
    return (i_loc_last - i_loc_begin) * nb + nbr;
  }

  /// \overlap localElementDistanceFromGlobalTile
  ///
  /// This overlap implements the 2D version of the function.
  LocalElementSize localElementDistanceFromGlobalTile(GlobalTileIndex begin,
                                                      GlobalTileIndex end) const noexcept {
    return {localElementDistanceFromGlobalTile<Coord::Row>(begin.row(), end.row()),
            localElementDistanceFromGlobalTile<Coord::Col>(begin.col(), end.col())};
  }

  /// Returns the local element size of the region between the local tile indices @p i_loc_begin and @p
  /// i_loc_end along the @p rc coordinate
  template <Coord rc>
  SizeType localElementDistanceFromLocalTile(SizeType i_loc_begin, SizeType i_loc_end) const noexcept {
    DLAF_ASSERT_HEAVY(i_loc_begin <= i_loc_end, i_loc_begin, i_loc_end);
    DLAF_ASSERT_HEAVY(0 <= i_loc_begin && i_loc_end <= local_nr_tiles_.get<rc>(), i_loc_begin, i_loc_end,
                      local_nr_tiles_.get<rc>());
    DLAF_ASSERT(tilesPerBlock<rc>() == 1, "Multi Tile distribution block is not supported yet");

    SizeType lsz = local_size_.get<rc>();
    SizeType nb = tile_size_.get<rc>();
    SizeType nbr = std::min(nb, lsz - (i_loc_end - 1) * nb);  // size of last local tile along `rc`
    return (i_loc_end - i_loc_begin - 1) * nb + nbr;
  }

  /// \overlap localElementDistanceFromLocalTile
  ///
  /// This overlap implements the 2D version of he function.
  LocalElementSize localElementDistanceFromLocalTile(LocalTileIndex begin,
                                                     LocalTileIndex end) const noexcept {
    return {localElementDistanceFromLocalTile<Coord::Row>(begin.row(), end.row()),
            localElementDistanceFromLocalTile<Coord::Col>(begin.col(), end.col())};
  }

private:
  /// @pre block_size_, and tile_size_ are already set correctly.
  template <Coord rc>
  SizeType tilesPerBlock() const noexcept {
    return block_size_.get<rc>() / tile_size_.get<rc>();
  }

  /// Computes and sets @p size_.
  ///
  /// @pre local_size_, is already set correctly.
  /// @pre grid_size_ == {1,1}.
  void computeGlobalSizeForNonDistr() noexcept;

  /// computes and sets global_tiles_.
  ///
  /// @pre local_size_, and tile_size_ are already set correctly.
  void computeGlobalNrTiles() noexcept;

  /// Computes and sets @p global_tiles_, @p local_tiles_ and @p local_size_.
  ///
  /// @pre size_, block_size_, tile_size_, grid_size_, rank_index and source_rank_index are already set correctly.
  void computeGlobalAndLocalNrTilesAndLocalSize() noexcept;

  /// computes and sets @p local_tiles_.
  ///
  /// @pre local_size_, and tile_size_ are already set correctly.
  void computeLocalNrTiles() noexcept;

  /// Sets default values.
  ///
  /// size_              = {0, 0}
  /// local_size_        = {0, 0}
  /// global_nr_tiles_   = {0, 0}
  /// local_nr_tiles_    = {0, 0}
  /// block_size_        = {1, 1}
  /// tile_size_         = {1, 1}
  /// rank_index_        = {0, 0}
  /// grid_size_         = {1, 1}
  /// source_rank_index_ = {0, 0}
  void setDefaultSizes() noexcept;

  GlobalElementSize size_;
  LocalElementSize local_size_;
  GlobalTileSize global_nr_tiles_;
  LocalTileSize local_nr_tiles_;
  TileElementSize block_size_;
  TileElementSize tile_size_;

  comm::Index2D rank_index_;
  comm::Size2D grid_size_;
  comm::Index2D source_rank_index_;
};
}
}
