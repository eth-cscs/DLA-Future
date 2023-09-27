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

/// @file

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/util_distribution.h>
#include <dlaf/util_math.h>

#define DLAF_DISTRIBUTION_ENABLE_DEPRECATED 1
#if (DLAF_DISTRIBUTION_ENABLE_DEPRECATED)
#define DLAF_DISTRIBUTION_DEPRECATED(x) [[deprecated(x)]]
#else
#define DLAF_DISTRIBUTION_DEPRECATED(x)
#endif

// TODO remove forward declarations when removing deprecated.
namespace dlaf::matrix {
class Distribution;

namespace internal::distribution {
template <Coord rc>
SizeType distance_to_adjacent_tile(const Distribution& dist, SizeType global_element) noexcept;
SizeType local_tile_linear_index(const Distribution& dist, LocalTileIndex ij) noexcept;
SizeType global_tile_linear_index(const Distribution& dist, GlobalElementIndex i_gl) noexcept;
template <Coord rc>
SizeType global_tile_element_distance(const Distribution& dist, SizeType i_begin,
                                      SizeType i_end) noexcept;
template <Coord rc>
SizeType local_element_distance_from_local_tile(const Distribution& dist, SizeType i_loc_begin,
                                                SizeType i_loc_end) noexcept;
template <Coord rc>
SizeType local_element_distance_from_global_tile(const Distribution& dist, SizeType i_begin,
                                                 SizeType i_end) noexcept;
LocalElementSize local_element_distance_from_local_tile(const Distribution& dist, LocalTileIndex begin,
                                                        LocalTileIndex end) noexcept;
GlobalTileIndex global_tile_index_from_sub_distribution(const Distribution& distribution,
                                                        const GlobalElementIndex& sub_offset,
                                                        const Distribution& sub_distribution,
                                                        const GlobalTileIndex& sub_index) noexcept;
TileElementIndex tile_element_offset_from_sub_distribution(const Distribution& distribution,
                                                           const GlobalElementIndex& sub_offset,
                                                           const Distribution& sub_distribution,
                                                           const GlobalTileIndex& sub_index) noexcept;
}

/// Contains information to create a sub-distribution.
struct SubDistributionSpec {
  GlobalElementIndex origin;
  GlobalElementSize size;
};

/// Distribution contains the information about the size and distribution of a matrix.
///
/// More details available in misc/matrix_distribution.md.
class Distribution {
public:
  /// Constructs a distribution for a non distributed matrix of size {0, 0} and block size {1, 1}.
  Distribution() noexcept;

  /// Constructs a distribution for a non distributed matrix of size @p size and block size @p block_size.
  ///
  /// @param[in] element_offset is the element-wise offset of the top left tile of the matrix ,
  /// @pre size.isValid(),
  /// @pre !block_size.isEmpty().
  Distribution(const LocalElementSize& size, const TileElementSize& block_size,
               const GlobalElementIndex& element_offset = {0, 0});

  /// Constructs a distribution for a matrix of size @p size and block size @p block_size,
  /// distributed on a 2D grid of processes of size @p grid_size.
  ///
  /// @param[in] rank_index is the rank of the current process,
  /// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
  /// @param[in] element_offset is the element-wise offset of the top left tile of the matrix ,
  /// @pre size.isValid(),
  /// @pre !block_size.isEmpty(),
  /// @pre !grid_size.isEmpty(),
  /// @pre rank_index.isIn(grid_size),
  /// @pre source_rank_index.isIn(grid_size).
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const comm::Size2D& grid_size, const comm::Index2D& rank_index,
               const comm::Index2D& source_rank_index,
               const GlobalElementIndex& element_offset = {0, 0});

  /// Constructs a distribution for a matrix of size @p size and block size @p block_size,
  /// distributed on a 2D grid of processes of size @p grid_size.
  ///
  /// @param[in] rank_index is the rank of the current process,
  /// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
  /// @param[in] tile_offset is the tile-wise offset of the top left tile of the matrix,
  /// @param[in] element_offset is the element-wise offset of the top left tile
  ///            of the matrix, used in addition to @p tile_offset,
  /// @pre size.isValid(),
  /// @pre !block_size.isEmpty(),
  /// @pre !grid_size.isEmpty(),
  /// @pre rank_index.isIn(grid_size),
  /// @pre source_rank_index.isIn(grid_size).
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const comm::Size2D& grid_size, const comm::Index2D& rank_index,
               const comm::Index2D& source_rank_index, const GlobalTileIndex& tile_offset,
               const GlobalElementIndex& element_offset = {0, 0});

  /// Constructs a distribution for a matrix of size @p size
  /// distributed on a 2D grid of processes of size @p grid_size.
  /// i.e. multiple tiles per distribution blocks are allowed.
  ///
  /// @param[in] rank_index is the rank of the current process,
  /// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
  /// @param[in] element_offset is the element-wise offset of the top left tile of the matrix ,
  /// @pre size.isValid(),
  /// @pre !block_size.isEmpty(),
  /// @pre !tile_size.isEmpty(),
  /// @pre block_size is divisible by tile_size,
  /// @pre !grid_size.isEmpty(),
  /// @pre rank_index.isIn(grid_size),
  /// @pre source_rank_index.isIn(grid_size).
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const TileElementSize& tile_size, const comm::Size2D& grid_size,
               const comm::Index2D& rank_index, const comm::Index2D& source_rank_index,
               const GlobalElementIndex& element_offset = {0, 0});

  /// Constructs a distribution for a matrix of size @p size
  /// distributed on a 2D grid of processes of size @p grid_size.
  /// i.e. multiple tiles per distribution blocks are allowed.
  ///
  /// @param[in] rank_index is the rank of the current process,
  /// @param[in] source_rank_index is the rank of the process which contains the top left tile of the matrix,
  /// @param[in] tile_offset is the tile-wise offset of the top left tile of the matrix,
  /// @param[in] element_offset is the element-wise offset of the top left tile
  ///            of the matrix, used in addition to @p tile_offset,
  /// @pre size.isValid(),
  /// @pre !block_size.isEmpty(),
  /// @pre !tile_size.isEmpty(),
  /// @pre block_size is divisible by tile_size,
  /// @pre !grid_size.isEmpty(),
  /// @pre rank_index.isIn(grid_size),
  /// @pre source_rank_index.isIn(grid_size).
  Distribution(const GlobalElementSize& size, const TileElementSize& block_size,
               const TileElementSize& tile_size, const comm::Size2D& grid_size,
               const comm::Index2D& rank_index, const comm::Index2D& source_rank_index,
               const GlobalTileIndex& tile_offset, const GlobalElementIndex& element_offset = {0, 0});

  Distribution(const Distribution& rhs) = default;

  Distribution(Distribution&& rhs) noexcept;

  Distribution& operator=(const Distribution& rhs) = default;

  Distribution& operator=(Distribution&& rhs) noexcept;

  /// Constructs a sub-distribution based on the given distribution @p dist specified by @p spec.
  ///
  /// @param[in] dist is the input distribution,
  /// @param[in] spec contains the origin and size of the new distribution relative to the input distribution,
  /// @pre spec.origin.isValid()
  /// @pre spec.size.isValid()
  /// @pre spec.origin + spec.size <= dist.size()
  Distribution(Distribution dist, const SubDistributionSpec& spec);

  bool operator==(const Distribution& rhs) const noexcept {
    return size_ == rhs.size_ && local_size_ == rhs.local_size_ && tile_size_ == rhs.tile_size_ &&
           block_size_ == rhs.block_size_ && global_nr_tiles_ == rhs.global_nr_tiles_ &&
           local_nr_tiles_ == rhs.local_nr_tiles_ && rank_index_ == rhs.rank_index_ &&
           grid_size_ == rhs.grid_size_ && source_rank_index_ == rhs.source_rank_index_ &&
           offset_ == rhs.offset_;
  }

  bool operator!=(const Distribution& rhs) const noexcept {
    return !operator==(rhs);
  }

  const GlobalElementSize& size() const noexcept {
    return size_;
  }

  const LocalElementSize& local_size() const noexcept {
    return local_size_;
  }

  /// Returns the number of tiles of the global matrix (2D size).
  const GlobalTileSize& nr_tiles() const noexcept {
    return global_nr_tiles_;
  }

  /// Returns the number of tiles stored locally (2D size).
  const LocalTileSize& local_nr_tiles() const noexcept {
    return local_nr_tiles_;
  }

  const TileElementSize& block_size() const noexcept {
    return block_size_;
  }

  const TileElementSize& tile_size() const noexcept {
    return tile_size_;
  }

  const comm::Index2D& rank_index() const noexcept {
    return rank_index_;
  }

  const comm::Size2D& grid_size() const noexcept {
    return grid_size_;
  }

  const comm::Index2D& source_rank_index() const noexcept {
    return source_rank_index_;
  }

  const GlobalElementIndex offset() const noexcept {
    return offset_;
  }

  /// Returns the global 2D index of the element.
  /// which has index @p tile_element in the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isIn(nr_tiles()),
  /// @pre tile_element.isIn(block_size()).
  GlobalElementIndex global_element_index(const GlobalTileIndex& global_tile,
                                          const TileElementIndex& tile_element) const noexcept {
    DLAF_ASSERT_HEAVY(global_tile.isIn(global_nr_tiles_), global_tile, global_nr_tiles_);
    DLAF_ASSERT_HEAVY(tile_element.isIn(tile_size_of(global_tile)), tile_element,
                      tile_size_of(global_tile));

    return {global_element_from_global_tile_and_tile_element<Coord::Row>(global_tile.row(),
                                                                         tile_element.row()),
            global_element_from_global_tile_and_tile_element<Coord::Col>(global_tile.col(),
                                                                         tile_element.col())};
  }

  /// Returns the global 2D index of the tile which contains the element with global index @p global_element.
  ///
  /// @pre global_element.isIn(size()).
  GlobalTileIndex global_tile_index(const GlobalElementIndex& global_element) const noexcept {
    DLAF_ASSERT_HEAVY(global_element.isIn(size_), global_element, size_);

    return {global_tile_from_global_element<Coord::Row>(global_element.row()),
            global_tile_from_global_element<Coord::Col>(global_element.col())};
  }

  /// Returns the global 2D index of the tile that has index @p local_tile
  /// in the current rank.
  ///
  /// @pre local_tile.isIn(local_nr_tiles()).
  GlobalTileIndex global_tile_index(const LocalTileIndex& local_tile) const noexcept {
    DLAF_ASSERT_HEAVY(local_tile.isIn(local_nr_tiles_), local_tile, local_nr_tiles_);

    return {global_tile_from_local_tile<Coord::Row>(local_tile.row()),
            global_tile_from_local_tile<Coord::Col>(local_tile.col())};
  }

  /// Returns the 2D rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre global_tile.isIn(nr_tiles()).
  comm::Index2D rank_global_tile(const GlobalTileIndex& global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(global_tile.isIn(global_nr_tiles_), global_tile, global_nr_tiles_);

    return {rank_global_tile<Coord::Row>(global_tile.row()),
            rank_global_tile<Coord::Col>(global_tile.col())};
  }

  /// Returns the local 2D index in the current rank of the tile with index @p global_tile.
  ///
  /// @pre global_tile.isIn(nr_tiles()),
  /// @pre rank_index == rank_global_tile(global_tile).
  LocalTileIndex local_tile_index(const GlobalTileIndex& global_tile) const {
    DLAF_ASSERT_HEAVY(global_tile.isIn(global_nr_tiles_), global_tile, global_nr_tiles_);

    DLAF_ASSERT_HEAVY(rank_index_ == rank_global_tile(global_tile), rank_index_,
                      rank_global_tile(global_tile));
    return {local_tile_from_global_tile<Coord::Row>(global_tile.row()),
            local_tile_from_global_tile<Coord::Col>(global_tile.col())};
  }

  /// Returns the local index in the current rank of the global element
  /// whose index is the smallest index larger or equal @p global_element
  /// and which is stored in the current rank.
  ///
  /// @pre global_element.isIn(size()).
  TileElementIndex tile_element_index(const GlobalElementIndex& global_element) const noexcept {
    DLAF_ASSERT_HEAVY(global_element.isIn(size_), global_element, size_);

    return {tile_element_from_global_element<Coord::Row>(global_element.row()),
            tile_element_from_global_element<Coord::Col>(global_element.col())};
  }

  /// Returns the size of the Tile with global index @p index.
  ///
  /// @pre global_element.isIn(size()).
  TileElementSize tile_size_of(const GlobalTileIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(nr_tiles()), index, nr_tiles());
    return {tile_size_of<Coord::Row>(index.row()), tile_size_of<Coord::Col>(index.col())};
  }

  ///////////////////////////////////
  // 1D Helpers returning elements //
  ///////////////////////////////////

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nr_tiles().get<rc>(),
  /// @pre 0 <= tile_element < tile_size.get<rc>().
  template <Coord rc>
  SizeType global_element_from_global_tile_and_tile_element(SizeType global_tile,
                                                            SizeType tile_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile < global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    DLAF_ASSERT_HEAVY(0 <= tile_element && tile_element < tile_size_.get<rc>(), tile_element,
                      tile_size_.get<rc>());
    return util::matrix::element_from_tile_and_tile_element(global_tile, tile_element,
                                                            tile_size_.get<rc>(),
                                                            global_tile_element_offset<rc>());
  }

  /// Returns the global index of the element
  /// which has index @p tile_element in the tile with local index @p local_tile in the current rank.
  ///
  /// @pre 0 <= local_tile < local_nr_tiles().get<rc>(),
  /// @pre 0 <= tile_element < tile_size.get<rc>().
  template <Coord rc>
  SizeType global_element_from_local_tile_and_tile_element(SizeType local_tile,
                                                           SizeType tile_element) const noexcept {
    return global_element_from_global_tile_and_tile_element<rc>(
        global_tile_from_local_tile<rc>(local_tile), tile_element);
  }

  /// Returns the global index of the element
  /// which has index @p local_element in the current rank.
  ///
  /// @pre 0 <= local_element < local_size().get<rc>(),
  template <Coord rc>
  SizeType global_element_from_local_element(const SizeType local_element) const noexcept {
    const auto local_tile = local_tile_from_local_element<rc>(local_element);
    const auto global_tile = global_tile_from_local_tile<rc>(local_tile);
    const auto tile_element = tile_element_from_local_element<rc>(local_element);
    return global_element_from_global_tile_and_tile_element<rc>(global_tile, tile_element);
  }

  /// Returns the local index in the current rank of the element
  /// which has global index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>(),
  template <Coord rc>
  SizeType local_element_from_global_element(const SizeType global_element) const noexcept {
    const auto local_tile = local_tile_from_global_element<rc>(global_element);
    if (local_tile < 0)
      return -1;
    const auto tile_element = tile_element_from_global_element<rc>(global_element);
    return local_element_from_local_tile_and_tile_element<rc>(local_tile, tile_element);
  }

  /// Returns the local index of the element
  /// which has index @p tile_element in the tile with local index @p local_tile.
  ///
  /// @pre 0 <= local_tile < local_nr_tiles().get<rc>(),
  /// @pre 0 <= tile_element < tile_size.get<rc>().
  template <Coord rc>
  SizeType local_element_from_local_tile_and_tile_element(SizeType local_tile,
                                                          SizeType tile_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= local_tile && local_tile < local_nr_tiles_.get<rc>(), local_tile,
                      local_nr_tiles_.get<rc>());
    DLAF_ASSERT_HEAVY(0 <= tile_element && tile_element < tile_size_.get<rc>(), tile_element,
                      tile_size_.get<rc>());
    return util::matrix::element_from_tile_and_tile_element(local_tile, tile_element,
                                                            tile_size_.get<rc>(),
                                                            local_tile_element_offset<rc>());
  }

  ////////////////////////////////////////
  // 1D Helpers returning tiles indices //
  ////////////////////////////////////////

  /// Returns the global index of the tile which contains the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType global_tile_from_global_element(SizeType global_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_element && global_element < size_.get<rc>(), global_element, size_);
    return util::matrix::tile_from_element(global_element, tile_size_.get<rc>(),
                                           global_tile_element_offset<rc>());
  }

  /// Returns the global index of the tile that has index @p local_tile
  /// in the current rank.
  ///
  /// @pre 0 <= local_tile < local_nr_tiles().get<rc>().
  template <Coord rc>
  SizeType global_tile_from_local_tile(SizeType local_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= local_tile && local_tile < local_nr_tiles_.get<rc>(), local_tile,
                      local_nr_tiles_.get<rc>());
    return util::matrix::global_tile_from_local_tile(local_tile, tiles_per_block<rc>(),
                                                     grid_size_.get<rc>(), rank_index_.get<rc>(),
                                                     source_rank_index_.get<rc>(),
                                                     global_tile_offset<rc>());
  }

  /// Returns the local index of the tile which contains the element with global index @p global_element.
  ///
  /// If the element with @p global_element index is not by current rank it returns -1.
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType local_tile_from_global_element(SizeType global_element) const noexcept {
    return local_tile_from_global_tile<rc>(global_tile_from_global_element<rc>(global_element));
  }

  /// Returns the local index in current rank of the tile with index @p global_tile.
  ///
  /// If the tiles with @p global_tile index is not by current rank it returns -1.
  /// @pre 0 <= global_tile < nr_tiles().get<rc>().
  template <Coord rc>
  SizeType local_tile_from_global_tile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile < global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    return util::matrix::local_tile_from_global_tile(global_tile, tiles_per_block<rc>(),
                                                     grid_size_.get<rc>(), rank_index_.get<rc>(),
                                                     source_rank_index_.get<rc>(),
                                                     global_tile_offset<rc>());
  }

  /// Returns the local index of the tile which contains the element with local index @p local_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType local_tile_from_local_element(SizeType local_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= local_element && local_element < local_size_.get<rc>(), local_element,
                      local_size_.get<rc>());
    return util::matrix::tile_from_element(local_element, tile_size_.get<rc>(),
                                           local_tile_element_offset<rc>());
  }

  /// Returns the local index in current rank of the global tile
  /// whose index is the smallest index larger or equal the index of the global tile
  /// that contains the element with index @p global_element
  /// and which is stored in current rank.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType next_local_tile_from_global_element(SizeType global_element) const noexcept {
    return next_local_tile_from_global_tile<rc>(global_tile_from_global_element<rc>(global_element));
  }

  /// Returns the local index in current rank of the global tile
  /// whose index is the smallest index larger or equal @p global_tile
  /// and which is stored in current rank. If there is no such tile
  /// index, the local tile grid size along @rc is returned.
  ///
  /// @pre 0 <= global_tile <= nr_tiles().get<rc>().
  template <Coord rc>
  SizeType next_local_tile_from_global_tile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    return util::matrix::next_local_tile_from_global_tile(global_tile, tiles_per_block<rc>(),
                                                          grid_size_.get<rc>(), rank_index_.get<rc>(),
                                                          source_rank_index_.get<rc>(),
                                                          global_tile_offset<rc>());
  }

  ////////////////////////////////////////
  // 1D Helpers returning tile elements //
  ////////////////////////////////////////

  /// Returns the index within the tile of the global element with index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  SizeType tile_element_from_global_element(SizeType global_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_element && global_element < size_.get<rc>(), global_element,
                      size_.get<rc>());
    return util::matrix::tile_element_from_element(global_element, tile_size_.get<rc>(),
                                                   global_tile_element_offset<rc>());
  }

  /// Returns the index within the tile of the local element with index @p local_element.
  ///
  /// @pre 0 <= local_element < local_size().get<rc>().
  template <Coord rc>
  SizeType tile_element_from_local_element(SizeType local_element) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= local_element && local_element < local_size_.get<rc>(), local_element,
                      local_size_.get<rc>());
    return util::matrix::tile_element_from_element(local_element, tile_size_.get<rc>(),
                                                   local_tile_element_offset<rc>());
  }

  ////////////////////////////////
  // 1D Helpers returning ranks //
  ////////////////////////////////

  /// Returns the rank index of the process that stores the element with global index @p global_element.
  ///
  /// @pre 0 <= global_element < size().get<rc>().
  template <Coord rc>
  int rank_global_element(SizeType global_element) const noexcept {
    return rank_global_tile<rc>(global_tile_from_global_element<rc>(global_element));
  }

  /// Returns the rank index of the process that stores the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nr_tiles().get<rc>().
  template <Coord rc>
  int rank_global_tile(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile < global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    return util::matrix::rank_global_tile(global_tile, tiles_per_block<rc>(), grid_size_.get<rc>(),
                                          source_rank_index_.get<rc>(), global_tile_offset<rc>());
  }

  ////////////////////////////////
  // 1D Helpers returning sizes //
  ////////////////////////////////

  /// Returns the size of the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nr_tiles().get<rc>().
  template <Coord rc>
  SizeType tile_size_of(SizeType global_tile) const noexcept {
    DLAF_ASSERT_HEAVY(0 <= global_tile && global_tile <= global_nr_tiles_.get<rc>(), global_tile,
                      global_nr_tiles_.get<rc>());
    SizeType n = size_.get<rc>();
    SizeType nb = tile_size_.get<rc>();
    if (global_tile == 0) {
      return std::min(nb - global_tile_element_offset<rc>(), n);
    }
    return std::min(nb, n + global_tile_element_offset<rc>() - global_tile * nb);
  }

private:
  /// @pre block_size_, and tile_size_ are already set correctly.
  template <Coord rc>
  SizeType tiles_per_block() const noexcept {
    return block_size_.get<rc>() / tile_size_.get<rc>();
  }

  /// Returns true if the current rank is the source rank along the @p rc coordinate, otherwise false.
  template <Coord rc>
  bool is_source_rank() const noexcept {
    return rank_index_.get<rc>() == source_rank_index_.get<rc>();
  }

  /// Computes the offset inside the first global block in terms of tiles along the @p rc coordinate.
  template <Coord rc>
  SizeType global_tile_offset() const noexcept {
    return offset_.get<rc>() / tile_size_.get<rc>();
  }

  /// Computes the offset inside the first global tile in terms of elements along the @p rc coordinate.
  template <Coord rc>
  SizeType global_tile_element_offset() const noexcept {
    return offset_.get<rc>() % tile_size_.get<rc>();
  }

  /// Computes the offset inside the first local block in terms of tiles along the @p rc coordinate.
  template <Coord rc>
  SizeType local_tile_offset() const noexcept {
    if (is_source_rank<rc>()) {
      return global_tile_offset<rc>();
    }
    else {
      return 0;
    }
  }

  /// Computes the offset inside the first local tile in terms of elements along the @p rc coordinate.
  template <Coord rc>
  SizeType local_tile_element_offset() const noexcept {
    if (is_source_rank<rc>()) {
      return global_tile_element_offset<rc>();
    }
    else {
      return 0;
    }
  }

  /// Computes and sets @p size_.
  ///
  /// @pre local_size_, is already set correctly.
  /// @pre grid_size_ == {1,1}.
  void compute_global_size_for_non_distr() noexcept;

  /// computes and sets global_tiles_.
  ///
  /// @pre local_size_, and tile_size_ are already set correctly.
  void compute_global_nr_tiles() noexcept;

  /// Computes and sets @p global_tiles_, @p local_tiles_ and @p local_size_.
  ///
  /// @pre size_, block_size_, tile_size_, grid_size_, rank_index and source_rank_index are already set correctly.
  void compute_global_and_local_nr_tiles_and_local_size() noexcept;

  /// computes and sets @p local_tiles_.
  ///
  /// @pre local_size_, and tile_size_ are already set correctly.
  void compute_local_nr_tiles() noexcept;

  /// Normalizes @p offset_ and @p source_rank_index_ into a canonical form.
  ///
  /// @pre offset_ and source_rank_index_ are already set correctly.
  /// @post offset_.row() < block_size_.rows() && offset_.col() < block_size_.cols()
  void normalize_source_rank_and_offset() noexcept;

  /// Sets default values.
  ///
  /// offset_            = {0, 0}
  /// size_              = {0, 0}
  /// local_size_        = {0, 0}
  /// global_nr_tiles_   = {0, 0}
  /// local_nr_tiles_    = {0, 0}
  /// block_size_        = {1, 1}
  /// tile_size_         = {1, 1}
  /// rank_index_        = {0, 0}
  /// grid_size_         = {1, 1}
  /// source_rank_index_ = {0, 0}
  void set_default_sizes() noexcept;

  GlobalElementIndex offset_;
  GlobalElementSize size_;
  LocalElementSize local_size_;
  GlobalTileSize global_nr_tiles_;
  LocalTileSize local_nr_tiles_;
  TileElementSize block_size_;
  TileElementSize tile_size_;

  comm::Index2D rank_index_;
  comm::Size2D grid_size_;
  comm::Index2D source_rank_index_;

public:
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  const LocalElementSize& localSize() const noexcept {
    return local_size();
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  const GlobalTileSize& nrTiles() const noexcept {
    return nr_tiles();
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  const LocalTileSize& localNrTiles() const noexcept {
    return local_nr_tiles();
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  const TileElementSize& blockSize() const noexcept {
    return block_size();
  }

  DLAF_DISTRIBUTION_DEPRECATED("use tile_size method")
  const TileElementSize& baseTileSize() const noexcept {
    return tile_size();
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  const comm::Index2D& rankIndex() const noexcept {
    return rank_index();
  }

  DLAF_DISTRIBUTION_DEPRECATED("use grid_size method")
  const comm::Size2D& commGridSize() const noexcept {
    return grid_size();
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  const comm::Index2D& sourceRankIndex() const noexcept {
    return source_rank_index();
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  GlobalElementIndex globalElementIndex(const GlobalTileIndex& global_tile,
                                        const TileElementIndex& tile_element) const noexcept {
    return global_element_index(global_tile, tile_element);
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  GlobalTileIndex globalTileIndex(const GlobalElementIndex& global_element) const noexcept {
    return global_tile_index(global_element);
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  GlobalTileIndex globalTileIndex(const LocalTileIndex& local_tile) const noexcept {
    return global_tile_index(local_tile);
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  comm::Index2D rankGlobalTile(const GlobalTileIndex& global_tile) const noexcept {
    return rank_global_tile(global_tile);
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  LocalTileIndex localTileIndex(const GlobalTileIndex& global_tile) const {
    return local_tile_index(global_tile);
  }

  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  TileElementIndex tileElementIndex(const GlobalElementIndex& global_element) const noexcept {
    return tile_element_index(global_element);
  }

  TileElementSize tileSize(const GlobalTileIndex& index) const noexcept {
    return tile_size_of(index);
  }

  ///////////////////////////////////
  // 1D Helpers returning elements //
  ///////////////////////////////////

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType globalElementFromGlobalTileAndTileElement(SizeType global_tile,
                                                     SizeType tile_element) const noexcept {
    return global_element_from_global_tile_and_tile_element<rc>(global_tile, tile_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType globalElementFromLocalTileAndTileElement(SizeType local_tile,
                                                    SizeType tile_element) const noexcept {
    return global_element_from_local_tile_and_tile_element<rc>(local_tile, tile_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType globalElementFromLocalElement(const SizeType local_element) const noexcept {
    return global_element_from_local_element<rc>(local_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType localElementFromGlobalElement(const SizeType global_element) const noexcept {
    return local_element_from_global_element<rc>(global_element);
  }

  ////////////////////////////////////////
  // 1D Helpers returning tiles indices //
  ////////////////////////////////////////

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType globalTileFromGlobalElement(SizeType global_element) const noexcept {
    return global_tile_from_global_element<rc>(global_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType globalTileFromLocalTile(SizeType local_tile) const noexcept {
    return global_tile_from_local_tile<rc>(local_tile);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType localTileFromGlobalElement(SizeType global_element) const noexcept {
    return local_tile_from_global_element<rc>(global_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType localTileFromLocalElement(SizeType local_element) const noexcept {
    return local_tile_from_local_element<rc>(local_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType localTileFromGlobalTile(SizeType global_tile) const noexcept {
    return local_tile_from_global_tile<rc>(global_tile);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType nextLocalTileFromGlobalElement(SizeType global_element) const noexcept {
    return next_local_tile_from_global_element<rc>(global_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType nextLocalTileFromGlobalTile(SizeType global_tile) const noexcept {
    return next_local_tile_from_global_tile<rc>(global_tile);
  }

  ////////////////////////////////////////
  // 1D Helpers returning tile elements //
  ////////////////////////////////////////

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType tileElementFromGlobalElement(SizeType global_element) const noexcept {
    return tile_element_from_global_element<rc>(global_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  SizeType tileElementFromLocalElement(SizeType local_element) const noexcept {
    return tile_element_from_local_element<rc>(local_element);
  }

  ////////////////////////////////
  // 1D Helpers returning ranks //
  ////////////////////////////////

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  int rankGlobalElement(SizeType global_element) const noexcept {
    return rank_global_element<rc>(global_element);
  }

  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("method has been renamed in snake case")
  int rankGlobalTile(SizeType global_tile) const noexcept {
    return rank_global_tile<rc>(global_tile);
  }

  ////////////////////////////////
  // 1D Helpers returning sizes //
  ////////////////////////////////

  /// Returns the size of the tile with global index @p global_tile.
  ///
  /// @pre 0 <= global_tile < nr_tiles().get<rc>().
  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("Use tile_size_of method")
  SizeType tileSize(SizeType global_tile) const noexcept {
    return tile_size_of<rc>(global_tile);
  }

  //////////////////////////////////////
  // only for tridiagonal eigensolver //
  //////////////////////////////////////

  // permutation
  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  SizeType distanceToAdjacentTile(SizeType global_element) const noexcept {
    return matrix::internal::distribution::distance_to_adjacent_tile<rc>(*this, global_element);
  }

  // tridsolver
  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  SizeType localTileLinearIndex(LocalTileIndex ij_local) const noexcept {
    return matrix::internal::distribution::local_tile_linear_index(*this, ij_local);
  }

  // tridsolver, permutation, test
  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  SizeType globalTileLinearIndex(GlobalElementIndex ij) const noexcept {
    return matrix::internal::distribution::global_tile_linear_index(*this, ij);
  }

  // tridsolver, permutation
  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  SizeType globalTileElementDistance(SizeType i_begin, SizeType i_end) const noexcept {
    return matrix::internal::distribution::global_tile_element_distance<rc>(*this, i_begin, i_end);
  }

  // permutation, test
  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  SizeType localElementDistanceFromGlobalTile(SizeType i_begin, SizeType i_end) const noexcept {
    return matrix::internal::distribution::local_element_distance_from_global_tile<rc>(*this, i_begin,
                                                                                       i_end);
  }

  // copy, tridsolver
  template <Coord rc>
  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  SizeType localElementDistanceFromLocalTile(SizeType i_loc_begin, SizeType i_loc_end) const noexcept {
    return matrix::internal::distribution::local_element_distance_from_local_tile<rc>(*this, i_loc_begin,
                                                                                      i_loc_end);
  }

  // copy, tridsolver
  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  LocalElementSize localElementDistanceFromLocalTile(LocalTileIndex begin,
                                                     LocalTileIndex end) const noexcept {
    return matrix::internal::distribution::local_element_distance_from_local_tile(*this, begin, end);
  }

  ///////////////////////////
  // helpers for submatrix //
  ///////////////////////////

  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  GlobalTileIndex globalTileIndexFromSubDistribution(const GlobalElementIndex& sub_offset,
                                                     const Distribution& sub_distribution,
                                                     const GlobalTileIndex& sub_index) const noexcept {
    return matrix::internal::distribution::global_tile_index_from_sub_distribution(*this, sub_offset,
                                                                                   sub_distribution,
                                                                                   sub_index);
  }

  DLAF_DISTRIBUTION_DEPRECATED("Use method from include/dlaf/matrix/distribution_extensions.h")
  TileElementIndex tileElementOffsetFromSubDistribution(
      const GlobalElementIndex& sub_offset, const Distribution& sub_distribution,
      const GlobalTileIndex& sub_index) const noexcept {
    return matrix::internal::distribution::tile_element_offset_from_sub_distribution(*this, sub_offset,
                                                                                     sub_distribution,
                                                                                     sub_index);
  }
};
}

// TODO remove when removing deprecated.
#include <dlaf/matrix/distribution_extensions.h>
