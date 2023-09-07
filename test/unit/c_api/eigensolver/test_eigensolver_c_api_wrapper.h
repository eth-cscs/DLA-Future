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

// Tests are in C++ (GTest + DLA-Future)
// The C_ wrappers ensure that the DLA-Future C API can be used in C

DLAF_EXTERN_C int C_dlaf_symmetric_eigensolver_s(const int dlaf_context, const char uplo, float* a,
                                                 const struct DLAF_descriptor desca, float* w, float* z,
                                                 const struct DLAF_descriptor descz);

DLAF_EXTERN_C int C_dlaf_symmetric_eigensolver_d(const int dlaf_context, const char uplo, double* a,
                                                 const struct DLAF_descriptor desca, double* w,
                                                 double* z, const struct DLAF_descriptor descz);

DLAF_EXTERN_C int C_dlaf_hermitian_eigensolver_c(const int dlaf_context, const char uplo,
                                                 dlaf_complex_c* a, const struct DLAF_descriptor desca,
                                                 float* w, dlaf_complex_c* z,
                                                 const struct DLAF_descriptor descz);

DLAF_EXTERN_C int C_dlaf_hermitian_eigensolver_z(const int dlaf_context, const char uplo,
                                                 dlaf_complex_z* a, const struct DLAF_descriptor desca,
                                                 double* w, dlaf_complex_z* z,
                                                 const struct DLAF_descriptor descz);

#ifdef DLAF_WITH_SCALAPACK
DLAF_EXTERN_C void C_dlaf_pssyevd(const char uplo, const int m, float* a, const int ia, const int ja,
                                  const int desca[9], float* w, float* z, const int iz, const int jz,
                                  const int descz[9], int* info);

DLAF_EXTERN_C void C_dlaf_pdsyevd(const char uplo, const int m, double* a, const int ia, const int ja,
                                  const int desca[9], double* w, double* z, const int iz, const int jz,
                                  const int descz[9], int* info);

DLAF_EXTERN_C void C_dlaf_pcheevd(const char uplo, const int m, dlaf_complex_c* a, const int ia,
                                  const int ja, const int desca[9], float* w, dlaf_complex_c* z,
                                  const int iz, const int jz, const int descz[9], int* info);

DLAF_EXTERN_C void C_dlaf_pzheevd(const char uplo, const int m, dlaf_complex_z* a, const int ia,
                                  const int ja, const int desca[9], double* w, dlaf_complex_z* z,
                                  const int iz, const int jz, const int descz[9], int* info);
#endif
