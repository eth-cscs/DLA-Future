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

DLAF_EXTERN_C int C_dlaf_inverse_from_cholesky_factor_s(const int dlaf_context, const char uplo,
                                                        float* a, const struct DLAF_descriptor desca);

DLAF_EXTERN_C int C_dlaf_inverse_from_cholesky_factor_d(const int dlaf_context, const char uplo,
                                                        double* a, const struct DLAF_descriptor desca);

DLAF_EXTERN_C int C_dlaf_inverse_from_cholesky_factor_c(const int dlaf_context, const char uplo,
                                                        dlaf_complex_c* a,
                                                        const struct DLAF_descriptor desca);

DLAF_EXTERN_C int C_dlaf_inverse_from_cholesky_factor_z(const int dlaf_context, const char uplo,
                                                        dlaf_complex_z* a,
                                                        const struct DLAF_descriptor desca);

#ifdef DLAF_WITH_SCALAPACK
DLAF_EXTERN_C void C_dlaf_pspotri(const char uplo, const int n, float* a, const int ia, const int ja,
                                  const int desca[9], int* info);

DLAF_EXTERN_C void C_dlaf_pdpotri(const char uplo, const int n, double* a, const int ia, const int ja,
                                  const int desca[9], int* info);

DLAF_EXTERN_C void C_dlaf_pcpotri(const char uplo, const int n, dlaf_complex_c* a, const int ia,
                                  const int ja, const int desca[9], int* info);

DLAF_EXTERN_C void C_dlaf_pzpotri(const char uplo, const int n, dlaf_complex_z* a, const int ia,
                                  const int ja, const int desca[9], int* info);
#endif
