//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_cholesky_c_api_wrapper.h"

#include <dlaf_c/factorization/cholesky.h>
#include <dlaf_c/utils.h>

#ifdef DLAF_WITH_SCALAPACK
void C_dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info) {
  dlaf_pdpotrf(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int* info) {
  dlaf_pspotrf(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pzpotrf(char uplo, int n, dlaf_complex_z* a, int ia, int ja, int* desca, int* info) {
  dlaf_pzpotrf(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pcpotrf(char uplo, int n, dlaf_complex_c* a, int ia, int ja, int* desca, int* info) {
  dlaf_pcpotrf(uplo, n, a, ia, ja, desca, info);
}
#endif

void C_dlaf_cholesky_d(int dlaf_context, char uplo, double* a, struct DLAF_descriptor desca) {
  dlaf_cholesky_d(dlaf_context, uplo, a, desca);
}

void C_dlaf_cholesky_s(int dlaf_context, char uplo, float* a, struct DLAF_descriptor desca) {
  dlaf_cholesky_s(dlaf_context, uplo, a, desca);
}

void C_dlaf_cholesky_z(int dlaf_context, char uplo, dlaf_complex_z* a, struct DLAF_descriptor desca) {
  dlaf_cholesky_z(dlaf_context, uplo, a, desca);
}

void C_dlaf_cholesky_c(int dlaf_context, char uplo, dlaf_complex_c* a, struct DLAF_descriptor desca) {
  dlaf_cholesky_c(dlaf_context, uplo, a, desca);
}
