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
void dlaf_cholesky_d(char uplo, double* a, int m, int n, int mb, int nb, int lld,
                     const MPI_Comm& communicator, int nprow, int npcol);

void dlaf_cholesky_s(char uplo, float* a, int m, int n, int mb, int nb, int lld,
                     const MPI_Comm& communicator, int nprow, int npcol);
void dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int& info);

void dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int& info);
}
