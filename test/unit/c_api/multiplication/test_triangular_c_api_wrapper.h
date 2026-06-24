//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

// Tests are in C++ (GTest + DLA-Future)
// The C_ wrappers ensure that the DLA-Future C API can be used in C

DLAF_EXTERN_C int C_dlaf_triangular_multiplication_s(const int dlaf_context, const char side,
                                                     const char uplo, const char op, const char diag,
                                                     const float alpha, const float* a,
                                                     const struct DLAF_descriptor desca, float* b,
                                                     const struct DLAF_descriptor descb);

DLAF_EXTERN_C int C_dlaf_triangular_multiplication_d(const int dlaf_context, const char side,
                                                     const char uplo, const char op, const char diag,
                                                     const double alpha, const double* a,
                                                     const struct DLAF_descriptor desca, double* b,
                                                     const struct DLAF_descriptor descb);

DLAF_EXTERN_C int C_dlaf_triangular_multiplication_c(
    const int dlaf_context, const char side, const char uplo, const char op, const char diag,
    const dlaf_complex_c alpha, const dlaf_complex_c* a, const struct DLAF_descriptor desca,
    dlaf_complex_c* b, const struct DLAF_descriptor descb);

DLAF_EXTERN_C int C_dlaf_triangular_multiplication_z(
    const int dlaf_context, const char side, const char uplo, const char op, const char diag,
    const dlaf_complex_z alpha, const dlaf_complex_z* a, const struct DLAF_descriptor desca,
    dlaf_complex_z* b, const struct DLAF_descriptor descb);

#ifdef DLAF_WITH_SCALAPACK
DLAF_EXTERN_C void C_dlaf_pstrmm(const char side, const char uplo, const char op, const char diag,
                                 const int m, const int n, const float alpha, const float* a,
                                 const int ia, const int ja, const int desca[9], float* b, const int ib,
                                 const int jb, const int descb[9]);

DLAF_EXTERN_C void C_dlaf_pdtrmm(const char side, const char uplo, const char op, const char diag,
                                 const int m, const int n, const double alpha, const double* a,
                                 const int ia, const int ja, const int desca[9], double* b, const int ib,
                                 const int jb, const int descb[9]);

DLAF_EXTERN_C void C_dlaf_pctrmm(const char side, const char uplo, const char op, const char diag,
                                 const int m, const int n, const dlaf_complex_c alpha,
                                 const dlaf_complex_c* a, const int ia, const int ja, const int desca[9],
                                 dlaf_complex_c* b, const int ib, const int jb, const int descb[9]);

DLAF_EXTERN_C void C_dlaf_pztrmm(const char side, const char uplo, const char op, const char diag,
                                 const int m, const int n, const dlaf_complex_z alpha,
                                 const dlaf_complex_z* a, const int ia, const int ja, const int desca[9],
                                 dlaf_complex_z* b, const int ib, const int jb, const int descb[9]);
#endif
