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

/// @file

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

/// Eigensolver
///
/// @pre The matrices \f$\mathbf{A}\f$ and \f$\mathbf{Z}\f$ are assumed to be distributed and in host
/// memory. The vector of eigenvalues \f$\mathbf{w}\f$ is assumed to be local (non-distributed) and in
/// host memory. Moving to and from GPU memory is handled internally.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with @ref dlaf_create_grid
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param dlaf_desca DLA-Future descriptor of the global matrix \f$\mathbf{A}\f$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global matrix \f$\mathbf{Z}\f$
/// @param dlaf_descz DLA-Future descriptor of the global matrix \f$\mathbf{Z}\f$
/// @return 0 if the eigensolver completed normally
DLAF_EXTERN_C int dlaf_symmetric_eigensolver_s(const int dlaf_context, const char uplo, float* a,
                                               const struct DLAF_descriptor dlaf_desca, float* w,
                                               float* z,
                                               const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_eigensolver_s
DLAF_EXTERN_C int dlaf_symmetric_eigensolver_d(const int dlaf_context, const char uplo, double* a,
                                               const struct DLAF_descriptor dlaf_desca, double* w,
                                               double* z,
                                               const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_eigensolver_s
DLAF_EXTERN_C int dlaf_hermitian_eigensolver_c(
    const int dlaf_context, const char uplo, dlaf_complex_c* a, const struct DLAF_descriptor dlaf_desca,
    float* w, dlaf_complex_c* z, const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_eigensolver_s
DLAF_EXTERN_C int dlaf_hermitian_eigensolver_z(
    const int dlaf_context, const char uplo, dlaf_complex_z* a, const struct DLAF_descriptor dlaf_desca,
    double* w, dlaf_complex_z* z, const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

#ifdef DLAF_WITH_SCALAPACK

/// Eigensolver
///
/// @remark This function is only available when DLAF_WITH_SCALAPACK=ON.
///
/// @pre The matrices \f$\mathbf{A}\f$ and \f$\mathbf{Z}\f$ are assumed to be distributed and in host
/// memory. The vector of eigenvalues \f$\mathbf{w}\f$ is assumed to be local (non-distributed) and in
/// host memory. Moving to and from GPU memory is handled internally.
///
/// @pre Submatrices are currently not supported, so @p n is the size of the full matrices
/// \f$\mathbf{A}\f$ and \f$\mathbf{Z}\f$ and @p ia, @p ja, @p iz, and @p jz need to be 1.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param n order of the sumbatrix \f$\mathbf{A}\f$ used in the computation
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param ia row index of the global matrix \f$\mathbf{A}\f$ identifying the first row of the submatrix
/// \f$\mathbf{A}\f$, has to be 1
/// @param ja column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix \f$\mathbf{A}\f$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global matrix \f$\mathbf{Z}\f$
/// @param iz row index of the global matrix \f$\mathbf{Z}\f$ identifying the first row of the submatrix
/// \f$\mathbf{Z}\f$, has to be 1
/// @param jz column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param descz ScaLAPACK array descriptor of the global matrix \f$\mathbf{Z}\f$
/// @param[out] info 0 if the eigensolver completed normally
DLAF_EXTERN_C void dlaf_pssyevd(const char uplo, const int n, float* a, const int ia, const int ja,
                                const int desca[9], float* w, float* z, const int iz, const int jz,
                                const int descz[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssyevd
DLAF_EXTERN_C void dlaf_pdsyevd(const char uplo, const int n, double* a, const int ia, const int ja,
                                const int desca[9], double* w, double* z, const int iz, const int jz,
                                const int descz[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssyevd
DLAF_EXTERN_C void dlaf_pcheevd(const char uplo, const int n, dlaf_complex_c* a, const int ia,
                                const int ja, const int desca[9], float* w, dlaf_complex_c* z,
                                const int iz, const int jz, const int descz[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssyevd
DLAF_EXTERN_C void dlaf_pzheevd(const char uplo, const int n, dlaf_complex_z* a, const int ia,
                                const int ja, const int desca[9], double* w, dlaf_complex_z* z,
                                const int iz, const int jz, const int descz[9], int* info) DLAF_NOEXCEPT;

#endif
