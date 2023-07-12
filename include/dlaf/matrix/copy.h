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

#include <dlaf/common/pipeline.h>
#include <dlaf/common/range2d.h>
#include <dlaf/communication/kernels/p2p.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/retiled_matrix.h>
#include <dlaf/types.h>
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

template <class T>
void copy(Matrix<T, Device::CPU>& src,  // TODO this should be const
          Matrix<T, Device::CPU>& dst, comm::CommunicatorGrid grid) {
  namespace ex = pika::execution::experimental;

  DLAF_ASSERT_MODERATE(equal_size(src, dst), src.size(), dst.size());
  DLAF_ASSERT_MODERATE(equal_process_grid(src, grid), src.commGridSize(), grid.size());
  DLAF_ASSERT_MODERATE(equal_process_grid(dst, grid), dst.commGridSize(), grid.size());

  const TileElementSize tile_size_src = src.blockSize();
  const TileElementSize tile_size_dst = dst.blockSize();

  const SizeType mb = std::min<SizeType>(tile_size_src.rows(), tile_size_dst.rows());
  const SizeType nb = std::min<SizeType>(tile_size_src.cols(), tile_size_dst.cols());

  DLAF_ASSERT_MODERATE(tile_size_src.rows() % mb == 0, tile_size_src.rows(), mb);
  DLAF_ASSERT_MODERATE(tile_size_dst.rows() % mb == 0, tile_size_dst.rows(), mb);
  DLAF_ASSERT_MODERATE(tile_size_src.cols() % nb == 0, tile_size_src.cols(), nb);
  DLAF_ASSERT_MODERATE(tile_size_dst.cols() % nb == 0, tile_size_dst.cols(), nb);

  const LocalTileSize tiles_per_block_src{tile_size_src.rows() / mb, tile_size_src.cols() / nb};
  const LocalTileSize tiles_per_block_dst{tile_size_dst.rows() / mb, tile_size_dst.cols() / nb};

  // TODO src_ should be const
  RetiledMatrix<T, Device::CPU> src_retiled(src, tiles_per_block_src);
  RetiledMatrix<T, Device::CPU> dst_retiled(dst, tiles_per_block_dst);

  const comm::Index2D rank = grid.rank();
  common::Pipeline<comm::Communicator> comm_pipeline(grid.fullCommunicator().clone());

  for (const LocalTileIndex ij_lc : common::iterate_range2d(src_retiled.distribution().localNrTiles())) {
    const GlobalTileIndex ij = src_retiled.distribution().globalTileIndex(ij_lc);
    const comm::Index2D src_rank = src_retiled.distribution().rankGlobalTile(ij);
    const comm::Index2D dst_rank = dst_retiled.distribution().rankGlobalTile(ij);

    const bool src_is_mine = rank == src_rank;
    const bool dst_is_mine = rank == dst_rank;

    if (src_is_mine != dst_is_mine) {
      ex::start_detached(comm::scheduleSend(comm_pipeline(), grid.rankFullCommunicator(dst_rank), 0,
                                            src_retiled.read(ij_lc)));
    }
  }

  const dlaf::internal::Policy<matrix::internal::CopyBackend_v<Device::CPU, Device::CPU>> policy;
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
      ex::start_detached(comm::scheduleRecv(comm_pipeline(), grid.rankFullCommunicator(src_rank), 0,
                                            dst_retiled.readwrite(ij_lc)));
    }
  }
}

}
