//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "eigensolver.h"

#include <complex>

#include <dlaf_c/eigensolver/eigensolver.h>
#include <dlaf_c/init.h>
#include <dlaf_c/utils.h>

int dlaf_eigensolver_s(int dlaf_context, char uplo, float* a, struct DLAF_descriptor dlaf_desca,
                       float* w, float* z, struct DLAF_descriptor dlaf_descz) {
  return eigensolver<float>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}

int dlaf_eigensolver_d(int dlaf_context, char uplo, double* a, struct DLAF_descriptor dlaf_desca,
                       double* w, double* z, struct DLAF_descriptor dlaf_descz) {
  return eigensolver<double>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}

int dlaf_eigensolver_c(int dlaf_context, char uplo, dlaf_complex_c* a, struct DLAF_descriptor dlaf_desca,
                       float* w, dlaf_complex_c* z, struct DLAF_descriptor dlaf_descz) {
  return eigensolver<std::complex<float>>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}

int dlaf_eigensolver_z(int dlaf_context, char uplo, dlaf_complex_z* a, struct DLAF_descriptor dlaf_desca,
                       double* w, dlaf_complex_z* z, struct DLAF_descriptor dlaf_descz) {
  return eigensolver<std::complex<double>>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_pssyevd(char uplo, int m, float* a, int ia, int ja, int* desca, float* w, float* z, int iz,
                  int jz, int* descz, int* info) {
  pxheevd<float>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, *info);
}

void dlaf_pdsyevd(char uplo, int m, double* a, int ia, int ja, int* desca, double* w, double* z, int iz,
                  int jz, int* descz, int* info) {
  pxheevd<double>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, *info);
}

void dlaf_pcheevd(char uplo, int m, dlaf_complex_c* a, int ia, int ja, int* desca, float* w,
                  dlaf_complex_c* z, int iz, int jz, int* descz, int* info) {
  pxheevd<std::complex<float>>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, *info);
}

void dlaf_pzheevd(char uplo, int m, dlaf_complex_z* a, int ia, int ja, int* desca, double* w,
                  dlaf_complex_z* z, int iz, int jz, int* descz, int* info) {
  pxheevd<std::complex<double>>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, *info);
}

#endif
