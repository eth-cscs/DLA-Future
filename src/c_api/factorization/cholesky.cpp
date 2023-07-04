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

void dlaf_cholesky_d(int dlaf_context, char uplo, double* a, DLAF_descriptor dlaf_desca) {
  cholesky<double>(dlaf_context, uplo, a, dlaf_desca);
}

void dlaf_cholesky_s(int dlaf_context, char uplo, float* a, DLAF_descriptor dlaf_desca) {
  cholesky<float>(dlaf_context, uplo, a, dlaf_desca);
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info) {
  pxpotrf<double>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int* info) {
  pxpotrf<float>(uplo, n, a, ia, ja, desca, *info);
}

#endif
