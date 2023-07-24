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

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/range2d.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/kernels/p2p.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

namespace dlaf::matrix {

/// Copy subset of local tiles from @p source in the range @p idx_source_begin, @p sz to local subset of
/// tiles in @p dest in the range @p idx_dest_begin, @p sz.
///
/// Note: If @p source and @p dest are the same matrix and iteration ranges overlap partially, the result
///       of the copy will be incorrect.
///
template <class T, Device Source, Device Destination>
void copy(LocalTileSize sz, LocalTileIndex idx_source_begin, Matrix<const T, Source>& source,
          LocalTileIndex idx_dest_begin, Matrix<T, Destination>& dest) {
  // If @p source and @p dest is the same matrix and the iteration range fully overlaps, return.
  if constexpr (Source == Destination) {
    if (idx_source_begin == idx_dest_begin && &source == &dest)
      return;
  }

  // Given that `sz` is the same for both `source` and `dest` it is sufficient to only check if the local
  // length of the copied region is the same.
  DLAF_ASSERT(
      source.distribution().localElementDistanceFromLocalTile(idx_source_begin, idx_source_begin + sz) ==
          dest.distribution().localElementDistanceFromLocalTile(idx_dest_begin, idx_dest_begin + sz),
      source, dest);

  namespace ex = pika::execution::experimental;
  for (auto idx_dest : common::iterate_range2d(idx_dest_begin, sz)) {
    LocalTileIndex idx_source = idx_source_begin + (idx_dest - idx_dest_begin);
    ex::start_detached(ex::when_all(source.read(idx_source), dest.readwrite(idx_dest)) |
                       copy(dlaf::internal::Policy<internal::CopyBackend_v<Source, Destination>>{}));
  }
}

/// \overload copy()
///
/// This overload makes sure that both @p source and @p dest local tiles start @p idx_begin.
///
template <class T, Device Source, Device Destination>
void copy(LocalTileIndex idx_begin, LocalTileSize sz, Matrix<const T, Source>& source,
          Matrix<T, Destination>& dest) {
  copy(sz, idx_begin, source, idx_begin, dest);
}

/// \overload copy()
///
/// This overload makes sure that all local tiles of @p source are copied to @p dest starting at tile (0, 0).
///
template <class T, Device Source, Device Destination>
void copy(Matrix<const T, Source>& source, Matrix<T, Destination>& dest) {
  copy(source.distribution().localNrTiles(), LocalTileIndex(0, 0), source, LocalTileIndex(0, 0), dest);
}

/// Copy of a matrix performing data reshuffling
///
/// Copy @p src matrix to @p dst, re-distributing data according to the @p dst distribution.
/// @pre src.size() == dst.size()
/// @pre equal_process_grid(src, dst)
/// @pre mb = min(src.blockSize().rows(), dst.blockSize().rows())
///      src.blockSize().rows() % mb == 0
///      dst.blockSize().rows() % mb == 0
/// @pre nb = min(src.blockSize().cols(), dst.blockSize().cols())
///      src.blockSize().cols() % nb == 0
///      dst.blockSize().cols() % nb == 0
/// @pre src has equal tile and block sizes.
/// @pre dst has equal tile and block sizes.
template <class T, Device Source, Device Destination>
void copy(Matrix<const T, Source>& src, Matrix<T, Destination>& dst, comm::CommunicatorGrid grid) {
  namespace ex = pika::execution::experimental;

  DLAF_ASSERT_MODERATE(equal_size(src, dst), src.size(), dst.size());
  DLAF_ASSERT_MODERATE(equal_process_grid(src, grid), src.commGridSize(), grid.size());
  DLAF_ASSERT_MODERATE(equal_process_grid(dst, grid), dst.commGridSize(), grid.size());

  DLAF_ASSERT_MODERATE(single_tile_per_block(src), src);
  DLAF_ASSERT_MODERATE(single_tile_per_block(src), dst);

  // Note:
  // From an algorithmic point of view it would be better to reason in terms of block instead of tiles,
  // with the aim of reducing the number of communications.
  // Current implementation reasons in terms of tiles due to a limitation for retiled matrices, which
  // cannot access the original block of the original matrix, but just their specific tile size (i.e.
  // tiles cannot be upscaled upto block). Dealing with tiles leads to a sub-optimal solution: smaller
  // chunks are communicated, leading to a potentially higher number of communications.
  const TileElementSize tile_size_src = src.baseTileSize();
  const TileElementSize tile_size_dst = dst.baseTileSize();

  const SizeType mb = std::min<SizeType>(tile_size_src.rows(), tile_size_dst.rows());
  const SizeType nb = std::min<SizeType>(tile_size_src.cols(), tile_size_dst.cols());

  DLAF_ASSERT_MODERATE(tile_size_src.rows() % mb == 0, tile_size_src.rows(), mb);
  DLAF_ASSERT_MODERATE(tile_size_dst.rows() % mb == 0, tile_size_dst.rows(), mb);
  DLAF_ASSERT_MODERATE(tile_size_src.cols() % nb == 0, tile_size_src.cols(), nb);
  DLAF_ASSERT_MODERATE(tile_size_dst.cols() % nb == 0, tile_size_dst.cols(), nb);

  const LocalTileSize scale_factor_src{tile_size_src.rows() / mb, tile_size_src.cols() / nb};
  const LocalTileSize scale_factor_dst{tile_size_dst.rows() / mb, tile_size_dst.cols() / nb};

  Matrix<const T, Source> src_retiled = src.retiledSubPipelineConst(scale_factor_src);
  Matrix<T, Destination> dst_retiled = dst.retiledSubPipeline(scale_factor_dst);

  const comm::Index2D rank = grid.rank();
  auto comm_sender = ex::just(grid.fullCommunicator().clone());

  auto tag = [dist = src_retiled.distribution()](GlobalTileIndex ij) -> comm::IndexT_MPI {
    // Note:
    // Source distribution is used as reference for both sending and receiving side.
    // The tag is computed as a function of local tile index, in order to keep its value as small as
    // possible. Moreover, since the local size of the matrix might change on different ranks, linear
    // local tile index is computed using a rank-independent ld.
    const auto size = dist.commGridSize();
    const LocalTileIndex source_ij_lc{ij.row() / size.rows(), ij.col() / size.cols()};
    const SizeType ld = dlaf::util::ceilDiv(dist.nrTiles().rows(), to_SizeType(size.rows()));
    return to_int(ij.row() + ij.col() * ld);
  };

  for (const LocalTileIndex ij_lc : common::iterate_range2d(src_retiled.distribution().localNrTiles())) {
    const GlobalTileIndex ij = src_retiled.distribution().globalTileIndex(ij_lc);
    const comm::Index2D src_rank = src_retiled.distribution().rankGlobalTile(ij);
    const comm::Index2D dst_rank = dst_retiled.distribution().rankGlobalTile(ij);

    const bool src_is_mine = rank == src_rank;
    const bool dst_is_mine = rank == dst_rank;

    if (src_is_mine != dst_is_mine) {
      ex::start_detached(comm::scheduleSend(ex::make_unique_any_sender(comm_sender),
                                            grid.rankFullCommunicator(dst_rank), tag(ij),
                                            src_retiled.read(ij_lc)));
    }
  }

  const dlaf::internal::Policy<matrix::internal::CopyBackend_v<Source, Destination>> policy;
  for (const LocalTileIndex ij_lc : common::iterate_range2d(dst_retiled.distribution().localNrTiles())) {
    const GlobalTileIndex ij = dst_retiled.distribution().globalTileIndex(ij_lc);
    const comm::Index2D src_rank = src_retiled.distribution().rankGlobalTile(ij);
    const comm::Index2D dst_rank = dst_retiled.distribution().rankGlobalTile(ij);

    const bool src_is_mine = rank == src_rank;
    const bool dst_is_mine = rank == dst_rank;

    if (src_is_mine == dst_is_mine) {
      ex::start_detached(ex::when_all(src_retiled.read(ij), dst_retiled.readwrite(ij_lc)) |
                         matrix::copy(policy));
    }
    else {
      ex::start_detached(comm::scheduleRecv(ex::make_unique_any_sender(comm_sender),
                                            grid.rankFullCommunicator(src_rank), tag(ij),
                                            dst_retiled.readwrite(ij_lc)));
    }
  }
}

}
