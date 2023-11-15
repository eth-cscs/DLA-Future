//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>

#include <dlaf_c/eigensolver/gen_eigensolver.h>
#include <dlaf_c/init.h>
#include <dlaf_c/utils.h>

#include "dlaf_c/desc.h"
#include "gen_eigensolver.h"

int dlaf_symmetric_generalized_eigensolver_s(const int dlaf_context, const char uplo, float* a,
                                             const struct DLAF_descriptor dlaf_desca, float* b,
                                             const struct DLAF_descriptor dlaf_descb, float* w, float* z,
                                             const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_generalized_eigensolver<float>(dlaf_context, uplo, a, dlaf_desca, b, dlaf_descb, w, z,
                                                  dlaf_descz);
}

int dlaf_symmetric_generalized_eigensolver_d(const int dlaf_context, const char uplo, double* a,
                                             const struct DLAF_descriptor dlaf_desca, double* b,
                                             const struct DLAF_descriptor dlaf_descb, double* w,
                                             double* z,
                                             const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_generalized_eigensolver<double>(dlaf_context, uplo, a, dlaf_desca, b, dlaf_descb, w,
                                                   z, dlaf_descz);
}

int dlaf_hermitian_generalized_eigensolver_c(const int dlaf_context, const char uplo, dlaf_complex_c* a,
                                             const struct DLAF_descriptor dlaf_desca, dlaf_complex_c* b,
                                             const struct DLAF_descriptor dlaf_descb, float* w,
                                             dlaf_complex_c* z,
                                             const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_generalized_eigensolver<std::complex<float>>(dlaf_context, uplo, a, dlaf_desca, b,
                                                                dlaf_descb, w, z, dlaf_descz);
}

int dlaf_hermitian_generalized_eigensolver_z(const int dlaf_context, const char uplo, dlaf_complex_z* a,
                                             const struct DLAF_descriptor dlaf_desca, dlaf_complex_z* b,
                                             const struct DLAF_descriptor dlaf_descb, double* w,
                                             dlaf_complex_z* z,
                                             const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_generalized_eigensolver<std::complex<double>>(dlaf_context, uplo, a, dlaf_desca, b,
                                                                 dlaf_descb, w, z, dlaf_descz);
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_pssygvx(const char uplo, const int m, float* a, const int ia, const int ja, const int desca[9],
                  float* b, const int ib, const int jb, const int descb[9], float* w, float* z,
                  const int iz, const int jz, const int descz[9], int* info) noexcept {
  pxhegvx<float>(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, *info);
}

void dlaf_pdsygvx(const char uplo, const int m, double* a, const int ia, const int ja,
                  const int desca[9], double* b, const int ib, const int jb, const int descb[9],
                  double* w, double* z, const int iz, const int jz, const int descz[9],
                  int* info) noexcept {
  pxhegvx<double>(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, *info);
}

void dlaf_pchegvx(const char uplo, const int m, dlaf_complex_c* a, const int ia, const int ja,
                  const int desca[9], dlaf_complex_c* b, const int ib, const int jb, const int descb[9],
                  float* w, dlaf_complex_c* z, const int iz, const int jz, const int descz[9],
                  int* info) noexcept {
  pxhegvx<std::complex<float>>(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, *info);
}

void dlaf_pzhegvx(const char uplo, const int m, dlaf_complex_z* a, const int ia, const int ja,
                  const int desca[9], dlaf_complex_z* b, const int ib, const int jb, const int descb[9],
                  double* w, dlaf_complex_z* z, const int iz, const int jz, const int descz[9],
                  int* info) noexcept {
  pxhegvx<std::complex<double>>(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, *info);
}

#endif
