//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "grid.h"
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf_c/grid.h>

#include <limits>

std::unordered_map<int, dlaf::comm::CommunicatorGrid> dlaf_grids;

int dlaf_create_grid(MPI_Comm comm, int nprow, int npcol, char order) {
  // TODO: Use a SizeType larger than BLACS context? TBD
  // dlaf_context starts from INT_MAX to reeduce the likelihood of clashes with blacs contexts
  // blacs starts to number contexts from 0
  int dlaf_context = std::numeric_limits<int>::max() - std::size(dlaf_grids);

  auto dlaf_order =
      order == 'C' ? dlaf::common::Ordering::ColumnMajor : dlaf::common::Ordering::RowMajor;

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(comm));

  dlaf_grids.try_emplace(dlaf_context, comm, nprow, npcol, dlaf_order);

  return dlaf_context;
}

void dlaf_create_grid_from_blacs(int blacs_ctxt) {
  int system_ctxt;
  // SGET_BLACSCONTXT == 10
  Cblacs_get(blacs_ctxt, 10, &system_ctxt);

  MPI_Comm communicator = Cblacs2sys_handle(system_ctxt);

  dlaf::comm::Communicator world(communicator);
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  int dims[2] = {0, 0};
  int coords[2] = {-1, -1};

  Cblacs_gridinfo(blacs_ctxt, &dims[0], &dims[1], &coords[0], &coords[1]);

  // TODO: Get ordering from BLACS

  dlaf_grids.try_emplace(blacs_ctxt, world, dims[0], dims[1], dlaf::common::Ordering::RowMajor);
}

void dlaf_free_grid(int blacs_ctxt) {
  dlaf_grids.erase(blacs_ctxt);
}
