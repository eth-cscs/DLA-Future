//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "cholesky.h"
#include <dlaf_c/factorization/cholesky.h>

#include <mpi.h>

extern "C" {

void dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info) {
  pxpotrf<double>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int* info) {
  pxpotrf<float>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_cholesky_d(int dlaf_context, char uplo, double* a, DLAF_descriptor dlaf_desca) {
  cholesky<double>(dlaf_context, uplo, a, dlaf_desca);
}
//
// void dlaf_cholesky_s(char uplo, float* a, int m, int n, int mb, int nb, int lld, const MPI_Comm&
// communicator,
//                 int nprow, int npcol) {
//   pxpotrf<float>(uplo, a, m, n, mb, nb, lld, communicator, nprow, npcol);
// }
}
