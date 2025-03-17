//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_cholesky_c_api_wrapper.h"

#include <dlaf_c/factorization/cholesky.h>
#include <dlaf_c/utils.h>

int C_dlaf_cholesky_factorization_s(const int dlaf_context, const char uplo, float* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_cholesky_factorization_s(dlaf_context, uplo, a, desca);
}

int C_dlaf_cholesky_factorization_d(const int dlaf_context, const char uplo, double* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_cholesky_factorization_d(dlaf_context, uplo, a, desca);
}

int C_dlaf_cholesky_factorization_c(const int dlaf_context, const char uplo, dlaf_complex_c* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_cholesky_factorization_c(dlaf_context, uplo, a, desca);
}

int C_dlaf_cholesky_factorization_z(const int dlaf_context, const char uplo, dlaf_complex_z* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_cholesky_factorization_z(dlaf_context, uplo, a, desca);
}

#ifdef DLAF_WITH_SCALAPACK
void C_dlaf_pdpotrf(const char uplo, const int n, double* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pdpotrf(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pspotrf(const char uplo, const int n, float* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pspotrf(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pzpotrf(const char uplo, const int n, dlaf_complex_z* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pzpotrf(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pcpotrf(const char uplo, const int n, dlaf_complex_c* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pcpotrf(uplo, n, a, ia, ja, desca, info);
}
#endif
