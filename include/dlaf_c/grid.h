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

#include <mpi.h>

#include "dlaf_c/utils.h"

/// Create communication grid
///
/// @warning Grids created here are indexed starting from INT_MAX, to avoid clashes with
/// BLACS contexts (which start from 0).
///
/// Grid ordering can be column-major ("C") or row-major ("R").
///
/// @param MPI communicator
/// @param nprow Number of process rows in the communicator grid
/// @param npcol Number of process columns in the communicator grid
/// @param order Grid ordering
/// @return DLA-Future context
DLAF_EXTERN_C int dlaf_create_grid(MPI_Comm comm, int nprow, int npcol, char order);

/// Free communicator grid for the given context
///
/// @warning This function only frees the DLA-Future grids stored internally. If you
/// created a BLACS grid with blacs_gridinit you still need to call blacs_gridexit
///
/// @param context BLACS or DLA-Future context associated to the grid being released
DLAF_EXTERN_C void dlaf_free_grid(int context);

#ifdef DLAF_WITH_SCALAPACK

/// Create communication grid from BLACS context
///
/// Grids created here are indexed by their corresponding BLACS context
/// @param blacs_ctxt.
///
/// The grid ordering is automatically inferred from the BLACS grid ordering.
/// Only row-major and column-major grids are supported (created with
/// blacs_gridinit). Grids created with blacs_gridmap are not supported.
///
/// @param blacs_ctxt BLACS context
DLAF_EXTERN_C void dlaf_create_grid_from_blacs(int blacs_ctxt);
#endif

/// Determine grid ordering
///
/// @remark For DLA-Future grids, the MPI communicator needs to be the original
/// communicator used to create the grid, not the full communicator stored within
/// the grid (after creation). DLA-Future automatically re-orders the communicator
/// within a DLA-Future grid in a row-major format.
///
/// @param comm MPI communicator
/// @param nprow Number of process rows in the grid
/// @param npcol Number of process columns in the grid
/// @param myprow Process row of the calling process
/// @param mypcol Process column of the calling process
/// @return Grid order ("R" for row-major, "C" cor column-major)
DLAF_EXTERN_C char grid_ordering(MPI_Comm comm, int nprow, int npcol, int myprow, int mypcol);
