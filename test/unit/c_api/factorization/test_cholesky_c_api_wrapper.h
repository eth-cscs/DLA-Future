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

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

// Tests are in C++ (GTest + DLA-Future)
// The C_ wrappers ensure that the DLA-Future C API can be used in C

DLAF_EXTERN_C int C_dlaf_cholesky_factorization_s(const int dlaf_context, const char uplo, float* a,
                                                  const struct DLAF_descriptor desca);

DLAF_EXTERN_C int C_dlaf_cholesky_factorization_d(const int dlaf_context, const char uplo, double* a,
                                                  const struct DLAF_descriptor desca);

DLAF_EXTERN_C int C_dlaf_cholesky_factorization_c(const int dlaf_context, const char uplo,
                                                  dlaf_complex_c* a, const struct DLAF_descriptor desca);

DLAF_EXTERN_C int C_dlaf_cholesky_factorization_z(const int dlaf_context, const char uplo,
                                                  dlaf_complex_z* a, const struct DLAF_descriptor desca);

#ifdef DLAF_WITH_SCALAPACK
DLAF_EXTERN_C void C_dlaf_pspotrf(const char uplo, const int n, float* a, const int ia, const int ja,
                                  const int desca[9], int* info);

DLAF_EXTERN_C void C_dlaf_pdpotrf(const char uplo, const int n, double* a, const int ia, const int ja,
                                  const int desca[9], int* info);

DLAF_EXTERN_C void C_dlaf_pcpotrf(const char uplo, const int n, dlaf_complex_c* a, const int ia,
                                  const int ja, const int desca[9], int* info);

DLAF_EXTERN_C void C_dlaf_pzpotrf(const char uplo, const int n, dlaf_complex_z* a, const int ia,
                                  const int ja, const int desca[9], int* info);
#endif
