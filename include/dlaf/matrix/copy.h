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

  using namespace dlaf;

  // VERIFY CONSTRAINTS
  // - same global size
  // - same grid
  // - over/sub scale with integral factor

  DLAF_ASSERT_MODERATE(matrix::equal_size(src, dst), src.size(), dst.size());
  DLAF_ASSERT_MODERATE(matrix::equal_process_grid(src, grid), src.commGridSize(), grid.size());
  DLAF_ASSERT_MODERATE(matrix::equal_process_grid(dst, grid), dst.commGridSize(), grid.size());

  const TileElementSize src_block = src.blockSize();
  const TileElementSize dst_block = dst.blockSize();

  const SizeType mb = std::min<SizeType>(src_block.rows(), dst_block.rows());
  const SizeType nb = std::min<SizeType>(src_block.cols(), dst_block.cols());

  DLAF_ASSERT_MODERATE(src_block.rows() % mb == 0, src_block.rows(), mb);
  DLAF_ASSERT_MODERATE(dst_block.rows() % mb == 0, dst_block.rows(), mb);
  DLAF_ASSERT_MODERATE(src_block.cols() % nb == 0, src_block.cols(), nb);
  DLAF_ASSERT_MODERATE(dst_block.cols() % nb == 0, dst_block.cols(), nb);

  const LocalTileSize src_tiles_per_block{src_block.rows() / mb, src_block.cols() / nb};
  const LocalTileSize dst_tiles_per_block{dst_block.rows() / mb, dst_block.cols() / nb};

  // TODO src_ should be const
  matrix::RetiledMatrix<T, Device::CPU> src_(src, src_tiles_per_block);
  matrix::RetiledMatrix<T, Device::CPU> dst_(dst, dst_tiles_per_block);

  const comm::Index2D rank = grid.rank();
  common::Pipeline<comm::Communicator> pipeline(grid.fullCommunicator().clone());

  for (const LocalTileIndex ij_loc : common::iterate_range2d(src_.distribution().localNrTiles())) {
    const GlobalTileIndex ij = src_.distribution().globalTileIndex(ij_loc);
    const comm::Index2D src_rank = src_.distribution().rankGlobalTile(ij);
    const comm::Index2D dst_rank = dst_.distribution().rankGlobalTile(ij);

    const bool src_is_mine = rank == src_rank;
    const bool dst_is_mine = rank == dst_rank;

    if (src_is_mine != dst_is_mine) {
      ex::start_detached(comm::scheduleSend(pipeline(), grid.rankFullCommunicator(dst_rank), 0,
                                            src_.read(ij_loc)));
    }
  }

  for (const LocalTileIndex ij_loc : common::iterate_range2d(dst_.distribution().localNrTiles())) {
    const GlobalTileIndex ij = dst_.distribution().globalTileIndex(ij_loc);
    const comm::Index2D src_rank = src_.distribution().rankGlobalTile(ij);
    const comm::Index2D dst_rank = dst_.distribution().rankGlobalTile(ij);

    const bool src_is_mine = rank == src_rank;
    const bool dst_is_mine = rank == dst_rank;

    if (src_is_mine == dst_is_mine) {
      namespace di = dlaf::internal;
      ex::start_detached(
          ex::when_all(src_.read(ij), dst_.readwrite(ij_loc)) |
          matrix::copy(di::Policy<matrix::internal::CopyBackend_v<Device::CPU, Device::CPU>>{}));
    }
    else {
      ex::start_detached(comm::scheduleRecv(pipeline(), grid.rankFullCommunicator(src_rank), 0,
                                            dst_.readwrite(ij_loc)));
    }
  }
}

}
