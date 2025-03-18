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

#include <mpi.h>

#include <dlaf_c/utils.h>

DLAF_EXTERN_C void Cblacs_gridinit(int* ictxt, char* layout, int nprow, int npcol);
DLAF_EXTERN_C void Cblacs_gridexit(int ictxt);
DLAF_EXTERN_C MPI_Comm Cblacs2sys_handle(int ictxt);
DLAF_EXTERN_C void Cblacs_get(int ictxt, int what, int* val);
DLAF_EXTERN_C void Cblacs_gridinfo(int ictxt, int* np, int* mp, int* px, int* py);
