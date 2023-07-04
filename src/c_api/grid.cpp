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

#include <limits>

#include <mpi.h>

#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf_c/grid.h>

#include "blacs.h"
#include "dlaf/communication/error.h"

std::unordered_map<int, dlaf::comm::CommunicatorGrid> dlaf_grids;

int dlaf_create_grid(MPI_Comm comm, int nprow, int npcol, char order) {
  // dlaf_context starts from INT_MAX to reeduce the likelihood of clashes with blacs contexts
  // blacs starts to number contexts from 0
  int dlaf_context = std::numeric_limits<int>::max() - static_cast<int>(std::size(dlaf_grids));

  auto dlaf_order = order == 'C' or order == 'c' ? dlaf::common::Ordering::ColumnMajor
                                                 : dlaf::common::Ordering::RowMajor;

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(comm));

  dlaf_grids.try_emplace(dlaf_context, comm, nprow, npcol, dlaf_order);

  return dlaf_context;
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_create_grid_from_blacs(int blacs_ctxt) {
  int system_ctxt;
  int get_blacs_contxt = 10;  // SGET_BLACSCONTXT == 10
  Cblacs_get(blacs_ctxt, get_blacs_contxt, &system_ctxt);

  MPI_Comm communicator = Cblacs2sys_handle(system_ctxt);

  dlaf::comm::Communicator world(communicator);
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  int dims[2] = {0, 0};
  int coords[2] = {-1, -1};

  Cblacs_gridinfo(blacs_ctxt, &dims[0], &dims[1], &coords[0], &coords[1]);

  auto order = grid_ordering(communicator, dims[0], dims[1], coords[0], coords[1]);
  auto dlaf_order =
      order == 'C' ? dlaf::common::Ordering::ColumnMajor : dlaf::common::Ordering::RowMajor;

  dlaf_grids.try_emplace(blacs_ctxt, world, dims[0], dims[1], dlaf_order);
}
#endif

void dlaf_free_grid(int ctxt) {
  dlaf_grids.erase(ctxt);
}

char grid_ordering(MPI_Comm comm, int nprow, int npcol, int myprow, int mypcol) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  bool _row_major = false, _col_major = false;
  bool row_major, col_major;

  if (rank == myprow * npcol + mypcol) {
    _row_major = true;
  }
  if (rank == mypcol * nprow + myprow) {
    _col_major = true;
  }

  MPI_Allreduce(&_row_major, &row_major, 1, MPI_C_BOOL, MPI_LAND, comm);
  MPI_Allreduce(&_col_major, &col_major, 1, MPI_C_BOOL, MPI_LAND, comm);

  if (!row_major and !col_major) {
    std::cout << "Grid layout must be row major or columns majord." << std::endl;
    exit(-1);
  }

  return col_major ? 'C' : 'R';
}
