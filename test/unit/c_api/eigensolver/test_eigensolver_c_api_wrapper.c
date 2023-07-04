//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_eigensolver_c_api_wrapper.h"

#include <dlaf_c/desc.h>
#include <dlaf_c/eigensolver/eigensolver.h>
#include <dlaf_c/utils.h>

#ifdef DLAF_WITH_SCALAPACK
void C_dlaf_pdsyevd(char uplo, int m, double* a, int ia, int ja, int* desca, double* w, double* z,
                    int iz, int jz, int* descz, int* info) {
  dlaf_pdsyevd(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, info);
}

void C_dlaf_pssyevd(char uplo, int m, float* a, int ia, int ja, int* desca, float* w, float* z, int iz,
                    int jz, int* descz, int* info) {
  dlaf_pssyevd(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, info);
}

void C_dlaf_pzheevd(char uplo, int m, dlaf_complex_z* a, int ia, int ja, int* desca, double* w,
                    dlaf_complex_z* z, int iz, int jz, int* descz, int* info) {
  dlaf_pzheevd(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, info);
}

void C_dlaf_pcheevd(char uplo, int m, dlaf_complex_c* a, int ia, int ja, int* desca, float* w,
                    dlaf_complex_c* z, int iz, int jz, int* descz, int* info) {
  dlaf_pcheevd(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, info);
}
#endif

void C_dlaf_eigensolver_d(int dlaf_context, char uplo, double* a, struct DLAF_descriptor desca,
                          double* w, double* z, struct DLAF_descriptor descz) {
  dlaf_eigensolver_d(dlaf_context, uplo, a, desca, w, z, descz);
}

void C_dlaf_eigensolver_s(int dlaf_context, char uplo, float* a, struct DLAF_descriptor desca, float* w,
                          float* z, struct DLAF_descriptor descz) {
  dlaf_eigensolver_s(dlaf_context, uplo, a, desca, w, z, descz);
}

void C_dlaf_eigensolver_z(int dlaf_context, char uplo, dlaf_complex_z* a, struct DLAF_descriptor desca,
                          double* w, dlaf_complex_z* z, struct DLAF_descriptor descz) {
  dlaf_eigensolver_z(dlaf_context, uplo, a, desca, w, z, descz);
}

void C_dlaf_eigensolver_c(int dlaf_context, char uplo, dlaf_complex_c* a, struct DLAF_descriptor desca,
                          float* w, dlaf_complex_c* z, struct DLAF_descriptor descz) {
  dlaf_eigensolver_c(dlaf_context, uplo, a, desca, w, z, descz);
}
