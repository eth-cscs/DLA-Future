//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf_c/grid.h>
#include "grid.h"

std::unordered_map<int, dlaf::comm::CommunicatorGrid> dlaf_grids;

void dlaf_create_grid_from_blacs(int blacs_ctxt){
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

void dlaf_free_grid(int blacs_ctxt){
  dlaf_grids.erase(blacs_ctxt);
}
