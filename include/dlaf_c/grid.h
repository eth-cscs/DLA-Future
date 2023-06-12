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

#include <mpi.h>
#include "dlaf_c/utils.h"

DLAF_EXTERN_C MPI_Comm Cblacs2sys_handle(int ictxt);
DLAF_EXTERN_C void Cblacs_get(int ictxt, int inum, int* comm);
DLAF_EXTERN_C void Cblacs_gridinfo(int ictxt, int* np, int* mp, int* px, int* py);

DLAF_EXTERN_C int dlaf_create_grid(MPI_Comm comm, int nprow, int npcol, char order = 'R');
DLAF_EXTERN_C void dlaf_create_grid_from_blacs(int blacs_ctxt);
DLAF_EXTERN_C void dlaf_free_grid(int blacs_ctxt);
