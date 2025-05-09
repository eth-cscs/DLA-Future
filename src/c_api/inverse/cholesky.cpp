//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>

#include <dlaf_c/inverse/cholesky.h>
#include <dlaf_c/utils.h>

#include "cholesky.h"

int dlaf_inverse_from_cholesky_factor_s(const int dlaf_context, const char uplo, float* a,
                                        const DLAF_descriptor dlaf_desca) noexcept {
  return inverse_from_cholesky_factor<float>(dlaf_context, uplo, a, dlaf_desca);
}

int dlaf_inverse_from_cholesky_factor_d(const int dlaf_context, const char uplo, double* a,
                                        const DLAF_descriptor dlaf_desca) noexcept {
  return inverse_from_cholesky_factor<double>(dlaf_context, uplo, a, dlaf_desca);
}

int dlaf_inverse_from_cholesky_factor_c(const int dlaf_context, const char uplo, dlaf_complex_c* a,
                                        const DLAF_descriptor dlaf_desca) noexcept {
  return inverse_from_cholesky_factor<std::complex<float>>(dlaf_context, uplo, a, dlaf_desca);
}

int dlaf_inverse_from_cholesky_factor_z(const int dlaf_context, const char uplo, dlaf_complex_z* a,
                                        const DLAF_descriptor dlaf_desca) noexcept {
  return inverse_from_cholesky_factor<std::complex<double>>(dlaf_context, uplo, a, dlaf_desca);
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_pspotri(const char uplo, const int n, float* a, const int ia, const int ja, const int desca[9],
                  int* info) noexcept {
  pxpotri<float>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pdpotri(const char uplo, const int n, double* a, const int ia, const int ja,
                  const int desca[9], int* info) noexcept {
  pxpotri<double>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pcpotri(const char uplo, const int n, dlaf_complex_c* a, const int ia, const int ja,
                  const int desca[9], int* info) noexcept {
  pxpotri<std::complex<float>>(uplo, n, a, ia, ja, desca, *info);
}

void dlaf_pzpotri(const char uplo, const int n, dlaf_complex_z* a, const int ia, const int ja,
                  const int desca[9], int* info) noexcept {
  pxpotri<std::complex<double>>(uplo, n, a, ia, ja, desca, *info);
}

#endif
