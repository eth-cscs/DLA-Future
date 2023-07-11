//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

#ifdef DLAF_WITH_SCALAPACK
DLAF_EXTERN_C void C_dlaf_pdsyevd(char uplo, int m, double* a, int ia, int ja, int* desca, double* w,
                                  double* z, int iz, int jz, int* descz, int* info);

DLAF_EXTERN_C void C_dlaf_pssyevd(char uplo, int m, float* a, int ia, int ja, int* desca, float* w,
                                  float* z, int iz, int jz, int* descz, int* info);

DLAF_EXTERN_C void C_dlaf_pzheevd(char uplo, int m, dlaf_complex_z* a, int ia, int ja, int* desca,
                                  double* w, dlaf_complex_z* z, int iz, int jz, int* descz, int* info);

DLAF_EXTERN_C void C_dlaf_pcheevd(char uplo, int m, dlaf_complex_c* a, int ia, int ja, int* desca,
                                  float* w, dlaf_complex_c* z, int iz, int jz, int* descz, int* info);
#endif

DLAF_EXTERN_C int C_dlaf_eigensolver_d(int dlaf_context, char uplo, double* a,
                                       struct DLAF_descriptor desca, double* w, double* z,
                                       struct DLAF_descriptor descz);

DLAF_EXTERN_C int C_dlaf_eigensolver_s(int dlaf_context, char uplo, float* a,
                                       struct DLAF_descriptor desca, float* w, float* z,
                                       struct DLAF_descriptor descz);

DLAF_EXTERN_C int C_dlaf_eigensolver_z(int dlaf_context, char uplo, dlaf_complex_z* a,
                                       struct DLAF_descriptor desca, double* w, dlaf_complex_z* z,
                                       struct DLAF_descriptor descz);

DLAF_EXTERN_C int C_dlaf_eigensolver_c(int dlaf_context, char uplo, dlaf_complex_c* a,
                                       struct DLAF_descriptor desca, float* w, dlaf_complex_c* z,
                                       struct DLAF_descriptor descz);
