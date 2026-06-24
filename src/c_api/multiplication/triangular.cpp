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

#include <dlaf_c/multiplication/triangular.h>
#include <dlaf_c/utils.h>

#include "triangular.h"

int dlaf_triangular_multiplication_s(const int dlaf_context, const char side, const char uplo,
                                     const char op, const char diag, const float alpha, const float* a,
                                     const DLAF_descriptor dlaf_desca, float* b,
                                     const DLAF_descriptor dlaf_descb) noexcept {
  return triangular_multiplication<float>(dlaf_context, side, uplo, op, diag, alpha, a, dlaf_desca, b,
                                          dlaf_descb);
}

int dlaf_triangular_multiplication_d(const int dlaf_context, const char side, const char uplo,
                                     const char op, const char diag, const double alpha, const double* a,
                                     const DLAF_descriptor dlaf_desca, double* b,
                                     const DLAF_descriptor dlaf_descb) noexcept {
  return triangular_multiplication<double>(dlaf_context, side, uplo, op, diag, alpha, a, dlaf_desca, b,
                                           dlaf_descb);
}

int dlaf_triangular_multiplication_c(const int dlaf_context, const char side, const char uplo,
                                     const char op, const char diag, const dlaf_complex_c alpha,
                                     const dlaf_complex_c* a, const DLAF_descriptor dlaf_desca,
                                     dlaf_complex_c* b, const DLAF_descriptor dlaf_descb) noexcept {
  return triangular_multiplication<std::complex<float>>(dlaf_context, side, uplo, op, diag, alpha, a,
                                                        dlaf_desca, b, dlaf_descb);
}

int dlaf_triangular_multiplication_z(const int dlaf_context, const char side, const char uplo,
                                     const char op, const char diag, const dlaf_complex_z alpha,
                                     const dlaf_complex_z* a, const DLAF_descriptor dlaf_desca,
                                     dlaf_complex_z* b, const DLAF_descriptor dlaf_descb) noexcept {
  return triangular_multiplication<std::complex<double>>(dlaf_context, side, uplo, op, diag, alpha, a,
                                                         dlaf_desca, b, dlaf_descb);
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_pstrmm(const char side, const char uplo, const char op, const char diag, const int m,
                 const int n, const float alpha, const float* a, const int ia, const int ja,
                 const int desca[9], float* b, const int ib, const int jb, const int descb[9]) noexcept {
  pxtrmm<float>(side, uplo, op, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

void dlaf_pdtrmm(const char side, const char uplo, const char op, const char diag, const int m,
                 const int n, const double alpha, const double* a, const int ia, const int ja,
                 const int desca[9], double* b, const int ib, const int jb,
                 const int descb[9]) noexcept {
  pxtrmm<double>(side, uplo, op, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

void dlaf_pctrmm(const char side, const char uplo, const char op, const char diag, const int m,
                 const int n, const dlaf_complex_c alpha, const dlaf_complex_c* a, const int ia,
                 const int ja, const int desca[9], dlaf_complex_c* b, const int ib, const int jb,
                 const int descb[9]) noexcept {
  pxtrmm<std::complex<float>>(side, uplo, op, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

void dlaf_pztrmm(const char side, const char uplo, const char op, const char diag, const int m,
                 const int n, const dlaf_complex_z alpha, const dlaf_complex_z* a, const int ia,
                 const int ja, const int desca[9], dlaf_complex_z* b, const int ib, const int jb,
                 const int descb[9]) noexcept {
  pxtrmm<std::complex<double>>(side, uplo, op, diag, m, n, alpha, a, ia, ja, desca, b, ib, jb, descb);
}

#endif
