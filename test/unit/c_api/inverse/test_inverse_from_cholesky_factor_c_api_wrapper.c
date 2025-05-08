//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_inverse_from_cholesky_factor_c_api_wrapper.h"

#include <dlaf_c/inverse/cholesky.h>
#include <dlaf_c/utils.h>

int C_dlaf_inverse_from_cholesky_factor_s(const int dlaf_context, const char uplo, float* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_inverse_from_cholesky_factor_s(dlaf_context, uplo, a, desca);
}

int C_dlaf_inverse_from_cholesky_factor_d(const int dlaf_context, const char uplo, double* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_inverse_from_cholesky_factor_d(dlaf_context, uplo, a, desca);
}

int C_dlaf_inverse_from_cholesky_factor_c(const int dlaf_context, const char uplo, dlaf_complex_c* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_inverse_from_cholesky_factor_c(dlaf_context, uplo, a, desca);
}

int C_dlaf_inverse_from_cholesky_factor_z(const int dlaf_context, const char uplo, dlaf_complex_z* a,
                                    const struct DLAF_descriptor desca) {
  return dlaf_inverse_from_cholesky_factor_z(dlaf_context, uplo, a, desca);
}

#ifdef DLAF_WITH_SCALAPACK
void C_dlaf_pdpotri(const char uplo, const int n, double* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pdpotri(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pspotri(const char uplo, const int n, float* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pspotri(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pzpotri(const char uplo, const int n, dlaf_complex_z* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pzpotri(uplo, n, a, ia, ja, desca, info);
}

void C_dlaf_pcpotri(const char uplo, const int n, dlaf_complex_c* a, const int ia, const int ja,
                    const int desca[9], int* info) {
  dlaf_pcpotri(uplo, n, a, ia, ja, desca, info);
}
#endif
