//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <complex>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>

#include <pika/execution.hpp>

#include <dlaf/common/range2d.h>
#include <dlaf/common/vector.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/allocation.h>
#include <dlaf/matrix/col_major_layout.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/internal/tile_pipeline.h>
#include <dlaf/matrix/matrix_base.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>

namespace dlaf {
namespace matrix {
namespace internal {

template <class T, Device D>
class MatrixRef;

/// Helper function returning a vector with the results of calling a function over a IterableRange2D
///
/// @param f non-void function accepting LocalTileIndex as parameter
template <class Func>
auto selectGeneric(Func&& f, common::IterableRange2D<SizeType, LocalTile_TAG> range) {
  using RetT = decltype(f(LocalTileIndex{}));

  std::vector<RetT> tiles;
  tiles.reserve(to_sizet(std::distance(range.begin(), range.end())));
  std::transform(range.begin(), range.end(), std::back_inserter(tiles),
                 [&](auto idx) { return f(idx); });
  return tiles;
}
}

/// A @c Matrix object represents a collection of tiles which contain all the elements of a matrix.
///
/// The tiles are distributed according to a distribution (see @c Matrix::distribution()),
/// therefore some tiles are stored locally on this rank,
/// while the others are available on other ranks.
template <class T, Device D>
class Matrix : public Matrix<const T, D> {
public:
  static constexpr Device device = D;

  using ElementType = T;
  using TileType = Tile<ElementType, D>;
  using ConstTileType = Tile<const ElementType, D>;
  using TileDataType = internal::TileData<ElementType, D>;
  using ReadWriteSenderType = ReadWriteTileSender<ElementType, D>;
  friend Matrix<const ElementType, D>;
  friend internal::MatrixRef<ElementType, D>;

  ///@{
  /// Create a non distributed matrix of size @p size, block size @p tile_size and
  /// tile_size @p tile_size.
  ///
  /// @param[in] alloc specifies the allocation details, default @p AllocationSpec{}.
  ///
  /// @pre @p size.isValid(),
  /// @pre @p !tile_size.isEmpty(),
  /// @pre @p !tile_size.isEmpty(),
  /// @pre <tt> alloc.ld() == Compact || alloc.ld() == Padded || alloc.ld() >= ld_min, </tt> where
  ///      <tt> ld_min = max(1, size.rows()) </tt> for <tt> alloc.layout() == ColMajor </tt> or
  ///      <tt> ld_min = block_size.rows() </tt> for <tt> alloc.layout() == Blocks </tt> or
  ///      <tt> ld_min = tile_size.rows() </tt> for <tt> alloc.layout() == Tiles </tt>.
  Matrix(const GlobalElementSize& size, const TileElementSize& tile_size,
         const AllocationSpec& alloc = {})
      : Matrix<T, D>(Distribution(size, tile_size, {1, 1}, {0, 0}, {0, 0}), alloc) {}
  Matrix(const LocalElementSize& size, const TileElementSize& tile_size,
         const AllocationSpec& alloc = {})
      : Matrix<T, D>(GlobalElementSize{size.rows(), size.cols()}, tile_size, alloc) {}
  Matrix(std::initializer_list<SizeType> size, const TileElementSize& tile_size,
         const AllocationSpec& alloc = {})
      : Matrix<T, D>((DLAF_ASSERT(size.size() == 2, size.size()),
                      GlobalElementSize{*size.begin(), *(size.begin() + 1)}),
                     tile_size, alloc) {}
  ///@}

  /// Create a distributed matrix of size @p size, block size @p tile_size and tile size @p tile_size
  /// on the given 2D communicator grid @p comm.
  ///
  /// @param[in] alloc specifies the allocation details, default @p AllocationSpec{}.
  ///
  /// @pre @p size.isValid(),
  /// @pre @p !tile_size.isEmpty(),
  /// @pre @p !tile_size.isEmpty(),
  /// @pre <tt> alloc.ld() == Compact || alloc.ld() == Padded || alloc.ld() >= ld_min, </tt> where
  ///      <tt> ld_min = max(1, local_size.rows()) </tt> for <tt> alloc.layout() == ColMajor </tt> or
  ///      <tt> ld_min = block_size.rows() </tt> for <tt> alloc.layout() == Blocks </tt> or
  ///      <tt> ld_min = tile_size.rows() </tt> for <tt> alloc.layout() == Tiles </tt>.
  Matrix(const GlobalElementSize& size, const TileElementSize& tile_size,
         const comm::CommunicatorGrid& comm, const AllocationSpec& alloc = {})
      : Matrix<T, D>(Distribution(size, tile_size, comm.size(), comm.rank(), {0, 0}), alloc) {}

  /// Create a matrix distributed according to the distribution @p distribution.
  ///
  /// @param[in] alloc specifies the allocation details, default @p AllocationSpec{}.
  ///
  /// @pre <tt> alloc.ld() == Compact || alloc.ld() == Padded || alloc.ld() >= ld_min, </tt> where
  ///      <tt> ld_min = max(1, local_size.rows()) </tt> for <tt> alloc.layout() == ColMajor </tt> or
  ///      <tt> ld_min = block_size.rows() </tt> for <tt> alloc.layout() == Blocks </tt> or
  ///      <tt> ld_min = tile_size.rows() </tt> for <tt> alloc.layout() == Tiles </tt>.
  Matrix(Distribution distribution, const AllocationSpec& alloc = {})
      : Matrix<const T, D>(std::move(distribution), alloc.layout()) {
    set_up_non_preallocated_tiles(alloc);
  }

  /// Create a distributed matrix,
  /// which references elements that are already allocated in the memory.
  ///
  /// @param[in] layout_mapper is an object
  ///            (which satisfies AllocationMapping concept (see ColMajorLayout for the methods required))
  ///            which describes how the elements of the local part of the matrix are stored in memory,
  /// @param[in] ptr is the pointer to the first element of the local part of the matrix,
  /// @pre @p ptr refers to an allocated memory region of at least @c layout_mapper.min_mem_size() elements.
  template <class AllocationMapping>
  Matrix(const AllocationMapping& layout_mapper, ElementType* ptr) noexcept
      : Matrix<const T, D>(layout_mapper, ptr) {}

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  /// Returns a sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().local_nr_tiles()).
  ReadWriteSenderType readwrite(const LocalTileIndex& index) noexcept {
    return tile_managers_[tile_linear_index(index)].readwrite();
  }

  /// Returns a sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadWriteSenderType readwrite(const GlobalTileIndex& index) noexcept {
    return readwrite(this->distribution().local_tile_index(index));
  }

  /// Create a sub-pipelined matrix which can be accessed thread-safely with respect to the original
  /// matrix
  ///
  /// All accesses to the sub-pipelined matrix are sequenced after previous accesses and before later
  /// accesses to the original matrix, independently of when tiles are accessed in the sub-pipelined
  /// matrix.
  Matrix sub_pipeline() noexcept {
    return Matrix(*this, SubPipelineTag{});
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  Matrix subPipeline() noexcept {
    return sub_pipeline();
  }

  /// Create a sub-pipelined, retiled matrix which can be accessed thread-safely with respect to the
  /// original matrix
  ///
  /// All accesses to the sub-pipelined matrix are sequenced after previous accesses and before later
  /// accesses to the original matrix, independently of when tiles are accessed in the sub-pipelined
  /// matrix.
  ///
  /// @pre blockSize() is divisible by @p tiles_per_block
  /// @pre blockSize() == tile_size()
  Matrix retiled_sub_pipeline(const LocalTileSize& tiles_per_block) noexcept {
    return Matrix(*this, tiles_per_block);
  }

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  Matrix retiledSubPipeline(const LocalTileSize& tiles_per_block) noexcept {
    return retiled_sub_pipeline(tiles_per_block);
  }

protected:
  using Matrix<const T, D>::tile_linear_index;

private:
  using typename Matrix<const T, D>::SubPipelineTag;
  Matrix(Matrix& mat, const SubPipelineTag tag) noexcept : Matrix<const T, D>(mat, tag) {}

  template <template <class, Device> class MatrixLike>
  Matrix(MatrixLike<const T, D>& mat, const LocalTileSize& tiles_per_block) noexcept
      : Matrix<const T, D>(mat, tiles_per_block) {}

  static SizeType compute_ld(SizeType rows, LdSpec ld) noexcept;
  // Note: safe to use in constructors if:
  // - MatrixBase is initialized correctly.
  void set_up_non_preallocated_tiles(const AllocationSpec&) noexcept;
  // using Matrix<const T, D>::set_up_tiles;
  using Matrix<const T, D>::tile_managers_;
};

template <class T, Device D>
SizeType Matrix<T, D>::compute_ld(SizeType rows, LdSpec ld) noexcept {
  SizeType min_ld = std::max<SizeType>(1, rows);
  SizeType ret = 0;

  if (std::holds_alternative<Ld>(ld)) {
    switch (std::get<Ld>(ld)) {
      case Ld::Compact:
        ret = min_ld;
        break;
      case Ld::Padded: {
        SizeType alignment;
        if constexpr (D == Device::CPU)
          alignment = std::max<SizeType>(1, 64 / sizeof(T));
        else
          alignment = 32;

        ret = (min_ld < alignment ? min_ld : util::ceilDiv(min_ld, alignment) * alignment);
        break;
      }
      default:
        DLAF_UNIMPLEMENTED("Invalid Ld");
    }
  }
  else
    ret = std::get<SizeType>(ld);

  DLAF_ASSERT(ret >= min_ld, ret, min_ld);
  return ret;
}

template <class T, Device D>
void Matrix<T, D>::set_up_non_preallocated_tiles(const AllocationSpec& alloc) noexcept {
  // Warning: Here we use matrix.allocation_layout() instead of alloc.layout() to ensure that the matrix
  //          is constructed consistently even if tune parameters are modified in the mean time.
  //          See AllocationLayoutDefault{} v.s. tune parameter default_allocation_layout.
  const auto& allocation = this->allocation_layout();
  const auto& ld = alloc.ld();

  using MemView = memory::MemoryView<T, D>;
  const Distribution& dist = this->distribution();

  const auto& local_nr_tiles = dist.local_nr_tiles();
  const auto& local_size = dist.local_size();

  if (local_size.isEmpty()) {
    // Needed to check if ld has a valid value.
    compute_ld(0, ld);
    return;
  }

  DLAF_ASSERT(tile_managers_.empty(), "");
  tile_managers_.reserve(to_sizet(local_nr_tiles.linear_size()));

  if (allocation == AllocationLayout::ColMajor) {
    SizeType mat_ld = compute_ld(local_size.rows(), ld);

    ColMajorLayout layout_mapper(dist, mat_ld);
    MemView mem(layout_mapper.min_mem_size());

    for (SizeType j = 0; j < local_nr_tiles.cols(); ++j) {
      for (SizeType i = 0; i < local_nr_tiles.rows(); ++i) {
        LocalTileIndex ij(i, j);
        SizeType offset = layout_mapper.tile_offset(ij);
        TileElementSize tile_size = layout_mapper.tile_size_of(ij);
        SizeType view_size = layout_mapper.min_tile_mem_size(ij);
        DLAF_ASSERT_HEAVY(!tile_size.isEmpty(), tile_size);
        tile_managers_.emplace_back(TileDataType(tile_size, MemView(mem, offset, view_size), mat_ld));
      }
    }
  }
  else if (allocation == AllocationLayout::Blocks) {
    // To be able to reuse tile methods for blocks we create a distribution with a single tile per block
    Distribution helper_dist = internal::create_single_tile_per_block_distribution(dist);

    LocalBlockSize local_nr_blocks = helper_dist.local_nr_blocks();
    DLAF_ASSERT(dist.local_nr_blocks() == local_nr_blocks, dist.local_nr_blocks(), local_nr_blocks);

    common::internal::vector<MemView> mem_views;
    mem_views.reserve(local_nr_blocks.rows());

    // Notation for the next loops:
    // size of the current block: mb x nb
    // size of the current tile: mt x nt
    // Loop over the columns of blocks and then loop over all the columns of tiles needed to fill the block.
    for (SizeType j_bl = 0, j = 0; j_bl < local_nr_blocks.cols(); ++j_bl) {
      SizeType nb = helper_dist.local_tile_size_of<Coord::Col>(j_bl);

      for (SizeType j_el_bl = 0; j_el_bl < nb;) {
        SizeType nt = dist.local_tile_size_of<Coord::Col>(j);
        // Loop over the block rows and then loop over all the rows of tiles needed to fill the block.
        for (SizeType i_bl = 0, i = 0; i_bl < local_nr_blocks.rows(); ++i_bl) {
          SizeType mb = helper_dist.local_tile_size_of<Coord::Row>(i_bl);
          SizeType block_ld = compute_ld(mb, ld);
          if (j_el_bl == 0) {
            // Create a MemView for each block in the block column.
            mem_views.emplace_back(block_ld * nb);
          }
          for (SizeType i_el_bl = 0; i_el_bl < mb;) {
            // Create the tile using the correct MemView
            SizeType mt = dist.local_tile_size_of<Coord::Row>(i);
            DLAF_ASSERT_HEAVY(!(TileElementSize{mt, nt}.isEmpty()), mt, nt);
            SizeType offset = i_el_bl + block_ld * j_el_bl;
            SizeType view_size = (nt - 1) * block_ld + mt;
            tile_managers_.emplace_back(
                TileDataType({mt, nt}, MemView(mem_views[i_bl], offset, view_size), block_ld));
            ++i;
            i_el_bl += mt;
          }
        }
        ++j;
        j_el_bl += nt;
      }
      mem_views.clear();
    }
  }
  else if (allocation == AllocationLayout::Tiles) {
    for (SizeType j = 0; j < local_nr_tiles.cols(); ++j) {
      for (SizeType i = 0; i < local_nr_tiles.rows(); ++i) {
        LocalTileIndex ij(i, j);
        TileElementSize tile_size = dist.tile_size_of(ij);
        DLAF_ASSERT_HEAVY(!tile_size.isEmpty(), tile_size);

        SizeType tile_ld = compute_ld(tile_size.rows(), ld);
        SizeType nr_elements = tile_ld * tile_size.cols();

        tile_managers_.emplace_back(TileDataType(tile_size, MemView(nr_elements), tile_ld));
      }
    }
  }
  else {
    DLAF_UNREACHABLE_PLAIN;
  }
}

template <class T, Device D>
class Matrix<const T, D> : public internal::MatrixBase {
public:
  static constexpr Device device = D;

  using ElementType = T;
  using TileType = Tile<ElementType, D>;
  using ConstTileType = Tile<const ElementType, D>;
  using TileDataType = internal::TileData<ElementType, D>;
  using ReadOnlySenderType = ReadOnlyTileSender<ElementType, D>;
  using ReadWriteSenderType = ReadWriteTileSender<ElementType, D>;
  friend internal::MatrixRef<const ElementType, D>;

  template <class AllocationMapping>
  Matrix(const AllocationMapping& layout_mapper, ElementType* ptr) noexcept
      : MatrixBase(layout_mapper.distribution()), allocation_layout_(layout_mapper.allocation_layout()) {
    memory::MemoryView<ElementType, D> mem(ptr, layout_mapper.min_mem_size());
    set_up_preallocated_tiles(mem, layout_mapper);
  }

  template <class AllocationMapping>
  Matrix(const AllocationMapping& layout_mapper, const ElementType* ptr) noexcept
      : Matrix(layout_mapper, const_cast<ElementType*>(ptr)) {}

  Matrix(const Matrix& rhs) = delete;
  Matrix(Matrix&& rhs) = default;

  Matrix& operator=(const Matrix& rhs) = delete;
  Matrix& operator=(Matrix&& rhs) = default;

  AllocationLayout allocation_layout() const noexcept {
    return allocation_layout_;
  }

  /// Returns a read-only sender of the Tile with local index @p index.
  ///
  /// @pre index.isIn(distribution().local_nr_tiles()).
  ReadOnlySenderType read(const LocalTileIndex& index) noexcept {
    return tile_managers_[tile_linear_index(index)].read();
  }

  /// Returns a read-only sender of the Tile with global index @p index.
  ///
  /// @pre the global tile is stored in the current process,
  /// @pre index.isIn(globalNrTiles()).
  ReadOnlySenderType read(const GlobalTileIndex& index) {
    return read(distribution().local_tile_index(index));
  }

  /// Synchronization barrier for all local tiles in the matrix
  ///
  /// This blocking call does not return until all operations, i.e. both RO and RW,
  /// involving any of the locally available tiles are completed.
  void wait_local_tiles() noexcept;

  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  void waitLocalTiles() noexcept {
    wait_local_tiles();
  }

  /// Create a sub-pipelined matrix which can be accessed thread-safely with respect to the original
  /// matrix
  ///
  /// All accesses to the sub-pipelined matrix are sequenced after previous accesses and before later
  /// accesses to the original matrix, independently of when tiles are accessed in the sub-pipelined
  /// matrix.
  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  Matrix subPipelineConst() {
    return Matrix(*this, SubPipelineTag{});
  }

  /// Create a sub-pipelined, retiled matrix which can be accessed thread-safely with respect to the
  /// original matrix
  ///
  /// All accesses to the sub-pipelined matrix are sequenced after previous accesses and before later
  /// accesses to the original matrix, independently of when tiles are accessed in the sub-pipelined
  /// matrix.
  ///
  /// @pre blockSize() is divisible by @p tiles_per_block
  /// @pre blockSize() == tile_size()
  DLAF_MATRIX_DEPRECATED("method has been renamed in snake case")
  Matrix retiledSubPipelineConst(const LocalTileSize& tiles_per_block) {
    return Matrix(*this, tiles_per_block);
  }

  /// Mark the tile at @p index as done
  ///
  /// Marking a tile as done means it can no longer be accessed. Marking a tile as done also disallows
  /// creation of sub pipelines from the full matrix.
  void done(const LocalTileIndex& index) noexcept {
    const auto i = tile_linear_index(index);
    tile_managers_[i].reset();
  }

  /// Mark the tile at @p index as done
  ///
  /// Marking a tile as done means it can no longer be accessed.  Marking a tile as done also disallows
  /// creation of sub pipelines from the full matrix.
  void done(const GlobalTileIndex& index) noexcept {
    done(distribution().local_tile_index(index));
  }

protected:
  Matrix(Distribution distribution, AllocationLayout alloc)
      : internal::MatrixBase{std::move(distribution)}, allocation_layout_(alloc) {
    DLAF_ASSERT((distribution.offset() == GlobalElementIndex{0, 0}), "not supported",
                distribution.offset());
  }

  struct SubPipelineTag {};
  Matrix(Matrix& mat, const SubPipelineTag) noexcept : MatrixBase(mat.distribution()) {
    set_up_sub_pipelines(mat);
  }

  template <template <class, Device> class MatrixLike>
  Matrix(MatrixLike<const T, D>& mat, const LocalTileSize& tiles_per_block) noexcept
      : MatrixBase(mat.distribution(), tiles_per_block), allocation_layout_(mat.allocation_layout()) {
    // Adjust allocation_layout_ when retiling the first time a tile allocated matrix.
    if (allocation_layout_ == AllocationLayout::Tiles && mat.distribution().single_tile_per_block() &&
        !distribution().single_tile_per_block())
      allocation_layout_ = AllocationLayout::Blocks;

    set_up_retiled_sub_pipelines(mat, tiles_per_block);
  }

  template <class AllocationMapping>
  void set_up_preallocated_tiles(const memory::MemoryView<ElementType, D>& mem,
                                 const AllocationMapping& layout_mapper) noexcept;
  void set_up_sub_pipelines(Matrix<const T, D>&) noexcept;
  template <template <class, Device> class MatrixLike>
  void set_up_retiled_sub_pipelines(MatrixLike<const T, D>&,
                                    const LocalTileSize& tiles_per_block) noexcept;

  std::vector<internal::TilePipeline<T, D>> tile_managers_;
  AllocationLayout allocation_layout_;

private:
  ReadWriteSenderType readwrite(const LocalTileIndex& index) noexcept {
    return tile_managers_[tileLinearIndex(index)].readwrite();
  }

  ReadWriteSenderType readwrite(const GlobalTileIndex& index) noexcept {
    return readwrite(this->distribution().local_tile_index(index));
  }
};

template <class T, Device D>
void Matrix<const T, D>::wait_local_tiles() noexcept {
  // Note:
  // Using a readwrite access to the tile ensures that the access is exclusive and not shared
  // among multiple tasks.

  const auto range_local = common::iterate_range2d(distribution().local_nr_tiles());

  auto s = pika::execution::experimental::when_all_vector(internal::selectGeneric(
      [this](const LocalTileIndex& index) {
        return this->tile_managers_[tileLinearIndex(index)].readwrite();
      },
      range_local));
  [[maybe_unused]] auto tiles = pika::this_thread::experimental::sync_wait(std::move(s));
}

template <class T, Device D>
template <class AllocationMapping>
void Matrix<const T, D>::set_up_preallocated_tiles(const memory::MemoryView<ElementType, D>& mem,
                                                   const AllocationMapping& layout_mapper) noexcept {
  const auto& nr_tiles = layout_mapper.nr_tiles();

  DLAF_ASSERT(tile_managers_.empty(), "");
  tile_managers_.reserve(to_sizet(nr_tiles.linear_size()));

  using MemView = memory::MemoryView<T, D>;

  for (SizeType j = 0; j < nr_tiles.cols(); ++j) {
    for (SizeType i = 0; i < nr_tiles.rows(); ++i) {
      LocalTileIndex ij(i, j);
      tile_managers_.emplace_back(TileDataType(layout_mapper.tile_size_of(ij),
                                               MemView(mem, layout_mapper.tile_offset(ij),
                                                       layout_mapper.min_tile_mem_size(ij)),
                                               layout_mapper.ld_tile(ij)));
    }
  }
}

template <class T, Device D>
void Matrix<const T, D>::set_up_sub_pipelines(Matrix<const T, D>& mat) noexcept {
  namespace ex = pika::execution::experimental;

  // TODO: Optimize read-after-read. This is currently forced to access the base
  // matrix in readwrite mode so that we can move the tile into the
  // sub-pipeline. This is semantically not required and should eventually be
  // optimized.
  tile_managers_.reserve(mat.tile_managers_.size());
  for (auto& tm : mat.tile_managers_) {
    tile_managers_.emplace_back(Tile<T, D>());
    auto s = ex::when_all(tile_managers_.back().readwrite_with_wrapper(), tm.readwrite()) |
             ex::then([](internal::TileAsyncRwMutexReadWriteWrapper<T, D> empty_tile_wrapper,
                         Tile<T, D> tile) { empty_tile_wrapper.get() = std::move(tile); });
    ex::start_detached(std::move(s));
  }
}

template <class T, Device D>
template <template <class, Device> class MatrixLike>
void Matrix<const T, D>::set_up_retiled_sub_pipelines(MatrixLike<const T, D>& mat,
                                                      const LocalTileSize& tiles_per_block) noexcept {
  DLAF_ASSERT(mat.blockSize() == mat.tile_size(), mat.blockSize(), mat.tile_size());

  using common::internal::vector;
  namespace ex = pika::execution::experimental;

  const auto n = to_sizet(distribution().local_nr_tiles().linear_size());
  tile_managers_.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    tile_managers_.emplace_back(Tile<T, D>());
  }

  const auto tile_size = distribution().tile_size();
  vector<SubTileSpec> specs;
  vector<LocalTileIndex> indices;
  specs.reserve(tiles_per_block.linear_size());
  indices.reserve(tiles_per_block.linear_size());

  // TODO: Optimize read-after-read. This is currently forced to access the base matrix in readwrite
  // mode so that we can move the tile into the sub-pipeline. This is semantically not required and
  // should eventually be optimized.
  for (const auto& orig_tile_index : common::iterate_range2d(mat.distribution().local_nr_tiles())) {
    const auto original_tile_size = mat.tileSize(mat.distribution().global_tile_index(orig_tile_index));

    for (SizeType j = 0; j < original_tile_size.cols(); j += tile_size.cols())
      for (SizeType i = 0; i < original_tile_size.rows(); i += tile_size.rows()) {
        indices.emplace_back(
            LocalTileIndex{orig_tile_index.row() * tiles_per_block.rows() + i / tile_size.rows(),
                           orig_tile_index.col() * tiles_per_block.cols() + j / tile_size.cols()});
        specs.emplace_back(SubTileSpec{{i, j},
                                       tileSize(distribution().global_tile_index(indices.back()))});
      }

    auto sub_tiles = splitTileDisjoint(mat.readwrite(orig_tile_index), specs);

    DLAF_ASSERT_HEAVY(specs.size() == indices.size(), specs.size(), indices.size());
    for (SizeType j = 0; j < specs.size(); ++j) {
      const auto i = tile_linear_index(indices[j]);

      // Move subtile to be managed by the tile manager of RetiledMatrix. We
      // use readwrite_with_wrapper to get access to the original tile managed
      // by the underlying async_rw_mutex.
      auto s =
          ex::when_all(tile_managers_[i].readwrite_with_wrapper(), std::move(sub_tiles[to_sizet(j)])) |
          ex::then([](internal::TileAsyncRwMutexReadWriteWrapper<T, D> empty_tile_wrapper,
                      Tile<T, D> sub_tile) { empty_tile_wrapper.get() = std::move(sub_tile); });
      ex::start_detached(std::move(s));
    }

    specs.clear();
    indices.clear();
  }
}

/// Returns a container grouping all the tiles retrieved using Matrix::read
///
/// @pre @p range must be a valid range for @p matrix
template <class MatrixLike>
auto selectRead(MatrixLike& matrix, common::IterableRange2D<SizeType, LocalTile_TAG> range) {
  return internal::selectGeneric([&](auto index) { return matrix.read(index); }, range);
}

/// Returns a container grouping all the tiles retrieved using Matrix::operator()
///
/// @pre @p range must be a valid range for @p matrix
template <class MatrixLike>
auto select(MatrixLike& matrix, common::IterableRange2D<SizeType, LocalTile_TAG> range) {
  return internal::selectGeneric([&](auto index) { return matrix.readwrite(index); }, range);
}

// ETI

#define DLAF_MATRIX_ETI(KWORD, DATATYPE, DEVICE) \
  KWORD template class Matrix<DATATYPE, DEVICE>; \
  KWORD template class Matrix<const DATATYPE, DEVICE>;

DLAF_MATRIX_ETI(extern, float, Device::CPU)
DLAF_MATRIX_ETI(extern, double, Device::CPU)
DLAF_MATRIX_ETI(extern, std::complex<float>, Device::CPU)
DLAF_MATRIX_ETI(extern, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_GPU)
DLAF_MATRIX_ETI(extern, float, Device::GPU)
DLAF_MATRIX_ETI(extern, double, Device::GPU)
DLAF_MATRIX_ETI(extern, std::complex<float>, Device::GPU)
DLAF_MATRIX_ETI(extern, std::complex<double>, Device::GPU)
#endif
}
#ifndef DLAF_DOXYGEN
// Note: Doxygen doesn't deal correctly with template specialized inheritance,
// and this line makes it run infinitely

/// Make dlaf::matrix::Matrix available as dlaf::Matrix.
using matrix::Matrix;
#endif
}
