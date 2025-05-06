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

#include <type_traits>
#include <utility>

#include <pika/execution.hpp>

#include <dlaf/common/index2d.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels/broadcast.h>
#include <dlaf/communication/message.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::comm {
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
template <class T, Device D, Coord axis, matrix::StoreTransposed storage,
          std::enable_if_t<!std::is_const_v<T>, int> = 0>
void broadcast(comm::IndexT_MPI rank_root, matrix::Panel<axis, T, D, storage>& panel,
               comm::CommunicatorPipeline<coord_to_communicator_type(orthogonal(axis))>& serial_comm) {
  constexpr auto comm_coord = axis;

  // do not schedule communication tasks if there is no reason to do so...
  if (panel.parentDistribution().commGridSize().get(comm_coord) <= 1)
    return;

  const auto rank = panel.parentDistribution().rankIndex().get(comm_coord);

  namespace ex = pika::execution::experimental;
  for (const auto& index : panel.iteratorLocal()) {
    if (rank == rank_root)
      ex::start_detached(schedule_bcast_send(serial_comm.exclusive(), panel.read(index)));
    else
      ex::start_detached(schedule_bcast_recv(serial_comm.exclusive(), rank_root,
                                             panel.readwrite(index)));
  }
}

namespace internal {
template <Coord C>
auto& get_taskchain(comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                    comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain) {
  if constexpr (C == Coord::Row) {
    return row_task_chain;
  }
  else {
    return col_task_chain;
  }
}
}  // namespace internal

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
/// Be aware that the last tile will just be available on @p panel, but it won't be transposed to
/// @p panelT.
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
template <class T, Device D, Coord axis, matrix::StoreTransposed storage,
          matrix::StoreTransposed storageT, std::enable_if_t<!std::is_const_v<T>, int> = 0>
void broadcast(comm::IndexT_MPI rank_root, matrix::Panel<axis, T, D, storage>& panel,
               matrix::Panel<orthogonal(axis), T, D, storageT>& panelT,
               comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
               comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain) {
  constexpr Coord coord = orthogonal(axis);
  constexpr Coord coordT = axis;

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
  DLAF_ASSERT(panel.rangeStart() <= panelT.rangeStart(), panel.rangeStart(), panelT.rangeStart());
  DLAF_ASSERT(panel.rangeEnd() >= panelT.rangeEnd(), panel.rangeEnd(), panelT.rangeEnd());

  // if no panel tiles, just skip it
  if (panel.rangeStart() == panel.rangeEnd())
    return;

  // STEP 1
  auto& chain_step1 = internal::get_taskchain<coord>(row_task_chain, col_task_chain);

  broadcast(rank_root, panel, chain_step1);

  // STEP 2
  auto& chain_step2 = internal::get_taskchain<coordT>(row_task_chain, col_task_chain);
  const auto& my_rank = dist.rank_index();

  const auto comm_size = dist.commGridSize().template get<coord>();

  for (SizeType k = panelT.rangeStart(); k < panelT.rangeEnd(); ++k) {
    const auto kk_rank = dist.rank_global_tile({k, k});
    if (kk_rank.template get<coordT>() == my_rank.template get<coordT>()) {
      const auto k_local_T = dist.template local_tile_from_global_tile<coordT>(k);

      namespace ex = pika::execution::experimental;
      if (kk_rank.template get<coord>() == my_rank.template get<coord>()) {
        const auto k_local = dist.template local_tile_from_global_tile<coord>(k);
        panelT.setTile({coordT, k_local_T}, panel.read({coord, k_local}));

        if (comm_size > 1)
          ex::start_detached(schedule_bcast_send(chain_step2.exclusive(),
                                                 panelT.read({coordT, k_local_T})));
      }
      else {
        if (comm_size > 1)
          ex::start_detached(schedule_bcast_recv(chain_step2.exclusive(), kk_rank.template get<coord>(),
                                                 panelT.readwrite({coordT, k_local_T})));
      }
    }
  }
}
}
