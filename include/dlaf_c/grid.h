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

extern "C" {
MPI_Comm Cblacs2sys_handle(int ictxt);
void Cblacs_get(int ictxt, int inum, int* comm);
extern "C" void Cblacs_gridinfo(int ictxt, int* np, int* mp, int* px, int* py);

void dlaf_create_grid_from_blacs(int blacs_ctxt);
void dlaf_free_grid(int blacs_ctxt);
}
