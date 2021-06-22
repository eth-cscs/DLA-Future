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

/// @file

#include <hpx/local/future.hpp>

#include "dlaf/communication/message.h"
#include "dlaf/executors.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace comm {
namespace internal {

// helper function that identifies the owner of a transposed coordinate,
// it returns both the component of the rank in the transposed dimension and
// its global cross coordinate (i.e. row == col in the global frame of reference)
template <Coord dst_coord>
std::pair<SizeType, comm::IndexT_MPI> transposedOwner(const matrix::Distribution& dist,
                                                      const LocalTileIndex idx) {
  const auto idx_cross = dist.template globalTileFromLocalTile<dst_coord>(idx.get(dst_coord));
  const auto rank_owner = dist.template rankGlobalTile<orthogonal(dst_coord)>(idx_cross);
  return std::make_pair(idx_cross, rank_owner);
}
}

/// Broadcast
///
/// Given a source panel on a rank, it gets broadcasted to make it available to all other ranks.
///
/// It does not give access to all the tiles, but just the ones of interest for each rank.
///
/// @param rank_root    on which rank the @p panel contains data to be broadcasted
/// @param panel        on @p rank_root it is the source panel
///                     on other ranks it is the destination panel
/// @param serial_comm  where to pipeline the tasks for communications.
/// @pre Communicator in @p serial_comm must be orthogonal to panel axis
template <class T, Device device, Coord axis, class = std::enable_if_t<!std::is_const<T>::value>>
void broadcast(const comm::Executor& ex, comm::IndexT_MPI rank_root,
               matrix::Panel<axis, T, device>& panel,
               common::Pipeline<comm::Communicator>& serial_comm) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  constexpr auto comm_coord = axis;

  // do not schedule communication tasks if there is no reason to do so...
  if (panel.parentDistribution().commGridSize().get(comm_coord) <= 1)
    return;

  const auto rank = panel.rankIndex().get(comm_coord);

  for (const auto& index : panel.iteratorLocal()) {
    if (rank == rank_root)
      scheduleSendBcast(ex, panel.read(index), serial_comm());
    else
      scheduleRecvBcast(ex, panel(index), rank_root, serial_comm());
  }
}

/// Broadcast
///
/// Given a source panel on a rank, this communication pattern makes every rank access tiles of both:
/// a. the source panel
/// b. it's tranposed variant (just tile coordinates, data is not transposed) w.r.t. the main diagonal of
/// the parent matrix
//
/// In particular, it does not give access to all the tiles, but just the ones of interest for
/// each rank, i.e. the rows and columns of a distributed matrix that the ranks stores locally.
///
/// This is achieved by either:
/// - linking as external tile, if the tile is already available locally for the rank
/// - receiving the tile from the owning rank (via a broadcast)
///
/// @param rank_root specifies on which rank the @p panel is the source of the data
/// @param panel
///   on rank_root it is the source panel (a)
///   on others it represents the destination for the broadcast (b)
/// @param panelT it represents the destination panel for the "transposed" variant of the panel
/// @param row_task_chain where to pipeline the tasks for row-wise communications
/// @param col_task_chain where to pipeline the tasks for col-wise communications
/// @param grid_size shape of the grid of row and col communicators from @p row_task_chain and @p col_task_chain
///
/// @pre both panels are child of a matrix (even not the same) with the same Distribution
/// @pre both panels parent matrices should be square matrices with square blocksizes
/// @pre both panels offsets should lay on the main diagonal of the parent matrix
template <class T, Device device, Coord axis, class = std::enable_if_t<!std::is_const<T>::value>>
void broadcast(const comm::Executor& ex, comm::IndexT_MPI rank_root,
               matrix::Panel<axis, T, device>& panel, matrix::Panel<orthogonal(axis), T, device>& panelT,
               common::Pipeline<comm::Communicator>& row_task_chain,
               common::Pipeline<comm::Communicator>& col_task_chain) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  constexpr Coord axisT = orthogonal(axis);

  constexpr Coord coord = std::decay_t<decltype(panel)>::CoordType;
  constexpr Coord coordT = std::decay_t<decltype(panelT)>::CoordType;

  auto get_taskchain = [&](Coord comm_dir) -> auto& {
    return comm_dir == Coord::Row ? row_task_chain : col_task_chain;
  };

  // Note:
  // Given a source panel, this communication pattern makes every rank access tiles of both the
  // source panel and it's tranposed variant (just tile coordinates, data is not transposed).
  // In particular, it does not give access to all the tiles, but just the ones of interest for
  // each rank, i.e. the rows and columns of a distributed matrix that the ranks stores locally.
  //
  // This happens in two steps (for the sake of example, let's consider a column -> row broadcast,
  // the opposite is dual):
  //
  // 1. broadcast the source panel to panel with the same shape on other ranks
  // 2. populate the transposed destination panel
  //
  // Once the source panel is share by all ranks, the transposed panel can be easily populated,
  // because each destination tile can be populated with data from the rank owning the diagonal one,
  // the point of contact between the row and the column (row == col).
  //
  // If it is already available locally, the tile is not copied and it just gets "linked", in order
  // to easily access it via panel coordinates with minimal (to null) overhead.
  // For this reason, the destination panel will depend on the source panel (as the source panel
  // may already depend on the matrix).

  DLAF_ASSERT(panel.parentDistribution() == panelT.parentDistribution(),
              "they must refer to the same matrix");

  const auto& dist = panel.parentDistribution();

  DLAF_ASSERT(square_size(dist), dist.size());
  DLAF_ASSERT(square_blocksize(dist), dist.blockSize());

  // Note:
  // This algorithm allow to broadcast panel to panelT using as mirror the parent matrix main diagonal.
  // This means that it is possible to broadcast panels with different axes just if their global offset
  // lie on the diagonal.
  DLAF_ASSERT(panel.rangeStart() == panelT.rangeStart(), panel.rangeStart(), panelT.rangeStart());
  DLAF_ASSERT_MODERATE(panel.rangeEnd() == panelT.rangeEnd(), panel.rangeEnd(), panelT.rangeEnd());

  // STEP 1
  constexpr auto comm_dir_step1 = orthogonal(axis);
  auto& chain_step1 = get_taskchain(comm_dir_step1);

  broadcast(ex, rank_root, panel, chain_step1);

  // STEP 2
  constexpr auto comm_dir_step2 = orthogonal(axisT);
  constexpr auto comm_coord_step2 = axisT;

  auto& chain_step2 = get_taskchain(comm_dir_step2);

  for (const auto& indexT : panelT.iteratorLocal()) {
    SizeType index_diag;
    comm::IndexT_MPI owner_diag;

    std::tie(index_diag, owner_diag) = internal::transposedOwner<coordT>(dist, indexT);

    if (dist.rankIndex().get(coord) == owner_diag) {
      const auto index_diag_local = dist.template localTileFromGlobalTile<coord>(index_diag);
      panelT.setTile(indexT, panel.read({coord, index_diag_local}));

      if (dist.commGridSize().get(comm_coord_step2) > 1)
        scheduleSendBcast(ex, panelT.read(indexT), chain_step2());
    }
    else {
      if (dist.commGridSize().get(comm_coord_step2) > 1)
        scheduleRecvBcast(ex, panelT(indexT), owner_diag, chain_step2());
    }
  }
}
}
}
