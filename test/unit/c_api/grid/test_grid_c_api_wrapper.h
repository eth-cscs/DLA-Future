//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <mpi.h>

#include <dlaf_c/utils.h>

DLAF_EXTERN_C char C_grid_ordering(MPI_Comm comm, int nprow, int npcol, int myprow, int mypcol);
