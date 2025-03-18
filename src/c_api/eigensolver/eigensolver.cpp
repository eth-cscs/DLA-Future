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

#include <dlaf_c/eigensolver/eigensolver.h>
#include <dlaf_c/init.h>
#include <dlaf_c/utils.h>

#include "eigensolver.h"

int dlaf_symmetric_eigensolver_s(const int dlaf_context, const char uplo, float* a,
                                 const struct DLAF_descriptor dlaf_desca, float* w, float* z,
                                 const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_eigensolver<float>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz, 0l,
                                      dlaf_desca.m);
}

int dlaf_symmetric_eigensolver_partial_spectrum_s(const int dlaf_context, const char uplo, float* a,
                                                  const struct DLAF_descriptor dlaf_desca, float* w,
                                                  float* z, const struct DLAF_descriptor dlaf_descz,
                                                  const SizeType eigenvalues_index_begin,
                                                  const SizeType eigenvalues_index_end) noexcept {
  return hermitian_eigensolver<float>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz,
                                      eigenvalues_index_begin, eigenvalues_index_end);
}

int dlaf_symmetric_eigensolver_d(const int dlaf_context, const char uplo, double* a,
                                 const struct DLAF_descriptor dlaf_desca, double* w, double* z,
                                 const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_eigensolver<double>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz, 0l,
                                       dlaf_desca.m);
}

int dlaf_symmetric_eigensolver_partial_spectrum_d(const int dlaf_context, const char uplo, double* a,
                                                  const struct DLAF_descriptor dlaf_desca, double* w,
                                                  double* z, const struct DLAF_descriptor dlaf_descz,
                                                  const SizeType eigenvalues_index_begin,
                                                  const SizeType eigenvalues_index_end) noexcept {
  return hermitian_eigensolver<double>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz,
                                       eigenvalues_index_begin, eigenvalues_index_end);
}

int dlaf_hermitian_eigensolver_c(const int dlaf_context, const char uplo, dlaf_complex_c* a,
                                 const struct DLAF_descriptor dlaf_desca, float* w, dlaf_complex_c* z,
                                 const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_eigensolver<std::complex<float>>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz,
                                                    0l, dlaf_desca.m);
}

int dlaf_hermitian_eigensolver_partial_spectrum_c(
    const int dlaf_context, const char uplo, dlaf_complex_c* a, const struct DLAF_descriptor dlaf_desca,
    float* w, dlaf_complex_c* z, const struct DLAF_descriptor dlaf_descz,
    const SizeType eigenvalues_index_begin, const SizeType eigenvalues_index_end) noexcept {
  return hermitian_eigensolver<std::complex<float>>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz,
                                                    eigenvalues_index_begin, eigenvalues_index_end);
}

int dlaf_hermitian_eigensolver_z(const int dlaf_context, const char uplo, dlaf_complex_z* a,
                                 const struct DLAF_descriptor dlaf_desca, double* w, dlaf_complex_z* z,
                                 const struct DLAF_descriptor dlaf_descz) noexcept {
  return hermitian_eigensolver<std::complex<double>>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz,
                                                     0l, dlaf_desca.m);
}

int dlaf_hermitian_eigensolver_partial_spectrum_z(
    const int dlaf_context, const char uplo, dlaf_complex_z* a, const struct DLAF_descriptor dlaf_desca,
    double* w, dlaf_complex_z* z, const struct DLAF_descriptor dlaf_descz,
    const SizeType eigenvalues_index_begin, const SizeType eigenvalues_index_end) noexcept {
  return hermitian_eigensolver<std::complex<double>>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz,
                                                     eigenvalues_index_begin, eigenvalues_index_end);
}

#ifdef DLAF_WITH_SCALAPACK

void dlaf_pssyevd(const char uplo, const int m, float* a, const int ia, const int ja, const int desca[9],
                  float* w, float* z, const int iz, const int jz, const int descz[9],
                  int* info) noexcept {
  pxheevd<float>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, 1l, m, *info);
}

void dlaf_pssyevd_partial_spectrum(const char uplo, const int m, float* a, const int ia, const int ja,
                                   const int desca[9], float* w, float* z, const int iz, const int jz,
                                   const int descz[9], const SizeType eigenvalues_index_begin,
                                   const SizeType eigenvalues_index_end, int* info) noexcept {
  pxheevd<float>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, eigenvalues_index_begin,
                 eigenvalues_index_end, *info);
}

void dlaf_pdsyevd(const char uplo, const int m, double* a, const int ia, const int ja,
                  const int desca[9], double* w, double* z, const int iz, const int jz,
                  const int descz[9], int* info) noexcept {
  pxheevd<double>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, 1l, m, *info);
}

void dlaf_pdsyevd_partial_spectrum(const char uplo, const int m, double* a, const int ia, const int ja,
                                   const int desca[9], double* w, double* z, const int iz, const int jz,
                                   const int descz[9], const SizeType eigenvalues_index_begin,
                                   const SizeType eigenvalues_index_end, int* info) noexcept {
  pxheevd<double>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, eigenvalues_index_begin,
                  eigenvalues_index_end, *info);
}
void dlaf_pcheevd(const char uplo, const int m, dlaf_complex_c* a, const int ia, const int ja,
                  const int desca[9], float* w, dlaf_complex_c* z, const int iz, const int jz,
                  const int descz[9], int* info) noexcept {
  pxheevd<std::complex<float>>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, 1l, m, *info);
}

void dlaf_pcheevd_partial_spectrum(const char uplo, const int m, dlaf_complex_c* a, const int ia,
                                   const int ja, const int desca[9], float* w, dlaf_complex_c* z,
                                   const int iz, const int jz, const int descz[9],
                                   const SizeType eigenvalues_index_begin,
                                   const SizeType eigenvalues_index_end, int* info) noexcept {
  pxheevd<std::complex<float>>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, eigenvalues_index_begin,
                               eigenvalues_index_end, *info);
}

void dlaf_pzheevd(const char uplo, const int m, dlaf_complex_z* a, const int ia, const int ja,
                  const int desca[9], double* w, dlaf_complex_z* z, const int iz, const int jz,
                  const int descz[9], int* info) noexcept {
  pxheevd<std::complex<double>>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, 1l, m, *info);
}

void dlaf_pzheevd_partial_spectrum(const char uplo, const int m, dlaf_complex_z* a, const int ia,
                                   const int ja, const int desca[9], double* w, dlaf_complex_z* z,
                                   const int iz, const int jz, const int descz[9],
                                   const SizeType eigenvalues_index_begin,
                                   const SizeType eigenvalues_index_end, int* info) noexcept {
  pxheevd<std::complex<double>>(uplo, m, a, ia, ja, desca, w, z, iz, jz, descz, eigenvalues_index_begin,
                                eigenvalues_index_end, *info);
}

#endif
