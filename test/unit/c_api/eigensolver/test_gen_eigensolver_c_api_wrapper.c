//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_gen_eigensolver_c_api_wrapper.h"

#include <dlaf_c/desc.h>
#include <dlaf_c/eigensolver/gen_eigensolver.h>
#include <dlaf_c/utils.h>

int C_dlaf_symmetric_generalized_eigensolver_s(const int dlaf_context, const char uplo, float* a,
                                               const struct DLAF_descriptor desca, float* b,
                                               const struct DLAF_descriptor descb, float* w, float* z,
                                               const struct DLAF_descriptor descz) {
  return dlaf_symmetric_generalized_eigensolver_s(dlaf_context, uplo, a, desca, b, descb, w, z, descz);
}

int C_dlaf_symmetric_generalized_eigensolver_d(const int dlaf_context, const char uplo, double* a,
                                               const struct DLAF_descriptor desca, double* b,
                                               const struct DLAF_descriptor descb, double* w, double* z,
                                               const struct DLAF_descriptor descz) {
  return dlaf_symmetric_generalized_eigensolver_d(dlaf_context, uplo, a, desca, b, descb, w, z, descz);
}

int C_dlaf_hermitian_generalized_eigensolver_c(const int dlaf_context, const char uplo,
                                               dlaf_complex_c* a, const struct DLAF_descriptor desca,
                                               dlaf_complex_c* b, const struct DLAF_descriptor descb,
                                               float* w, dlaf_complex_c* z,
                                               const struct DLAF_descriptor descz) {
  return dlaf_hermitian_generalized_eigensolver_c(dlaf_context, uplo, a, desca, b, descb, w, z, descz);
}

int C_dlaf_hermitian_generalized_eigensolver_z(const int dlaf_context, const char uplo,
                                               dlaf_complex_z* a, const struct DLAF_descriptor desca,
                                               dlaf_complex_z* b, const struct DLAF_descriptor descb,
                                               double* w, dlaf_complex_z* z,
                                               const struct DLAF_descriptor descz) {
  return dlaf_hermitian_generalized_eigensolver_z(dlaf_context, uplo, a, desca, b, descb, w, z, descz);
}

#ifdef DLAF_WITH_SCALAPACK

void C_dlaf_pssygvx(char uplo, const int m, float* a, const int ia, const int ja, const int desca[9],
                    float* b, const int ib, const int jb, const int descb[9], float* w, float* z,
                    const int iz, const int jz, const int descz[9], int* info) {
  dlaf_pssygvx(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, info);
}

void C_dlaf_pdsygvx(const char uplo, const int m, double* a, const int ia, const int ja,
                    const int desca[9], double* b, const int ib, const int jb, const int descb[9],
                    double* w, double* z, const int iz, const int jz, const int descz[9], int* info) {
  dlaf_pdsygvx(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, info);
}

void C_dlaf_pchegvx(const char uplo, const int m, dlaf_complex_c* a, const int ia, const int ja,
                    const int desca[9], dlaf_complex_c* b, const int ib, const int jb,
                    const int descb[9], float* w, dlaf_complex_c* z, const int iz, const int jz,
                    const int descz[9], int* info) {
  dlaf_pchegvx(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, info);
}

void C_dlaf_pzhegvx(const char uplo, const int m, dlaf_complex_z* a, const int ia, const int ja,
                    const int desca[9], dlaf_complex_z* b, const int ib, const int jb,
                    const int descb[9], double* w, dlaf_complex_z* z, const int iz, const int jz,
                    const int descz[9], int* info) {
  dlaf_pzhegvx(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, info);
}

#endif
