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

/// Generalized eigensolver
///
/// @pre The matrices \f$\mathbf{A}\f$, \f$\mathbf{B}\f$,  and \f$\mathbf{Z}\f$ are assumed to be
/// distributed and in host memory. The vector of eigenvalues \f$\mathbf{w}\f$ is assumed to be local
/// (non-distributed) and in host memory. Moving to and from GPU memory is handled internally.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with @ref dlaf_create_grid
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param dlaf_desca DLA-Future descriptor of the global matrix \f$\mathbf{A}\f$
/// @param b Local part of the global matrix \f$\mathbf{B}\f$
/// @param dlaf_descb DLA-Future descriptor of the global matrix \f$\mathbf{B}\f$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global matrix \f$\mathbf{Z}\f$
/// @param dlaf_descz DLA-Future descriptor of the global matrix \f$\mathbf{Z}\f$
/// @return 0 if the eigensolver completed normally
DLAF_EXTERN_C int dlaf_symmetric_generalized_eigensolver_s(
    const int dlaf_context, const char uplo, float* a, const struct DLAF_descriptor dlaf_desca, float* b,
    const struct DLAF_descriptor dlaf_descb, float* w, float* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_generalized_eigensolver_s
DLAF_EXTERN_C int dlaf_symmetric_generalized_eigensolver_d(
    const int dlaf_context, const char uplo, double* a, const struct DLAF_descriptor dlaf_desca,
    double* b, const struct DLAF_descriptor dlaf_descb, double* w, double* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_generalized_eigensolver_s
DLAF_EXTERN_C int dlaf_hermitian_generalized_eigensolver_c(
    const int dlaf_context, const char uplo, dlaf_complex_c* a, const struct DLAF_descriptor dlaf_desca,
    dlaf_complex_c* b, const struct DLAF_descriptor dlaf_descb, float* w, dlaf_complex_c* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_generalized_eigensolver_s
DLAF_EXTERN_C int dlaf_hermitian_generalized_eigensolver_z(
    const int dlaf_context, const char uplo, dlaf_complex_z* a, const struct DLAF_descriptor dlaf_desca,
    dlaf_complex_z* b, const struct DLAF_descriptor dlaf_descb, double* w, dlaf_complex_z* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// Generalized eigensolver
///
/// @pre The matrices \f$\mathbf{A}\f$, \f$\mathbf{B}\f$,  and \f$\mathbf{Z}\f$ are assumed to be
/// distributed and in host memory. The vector of eigenvalues \f$\mathbf{w}\f$ is assumed to be local
/// (non-distributed) and in host memory. Moving to and from GPU memory is handled internally.
///
/// @pre The matrix \f$\mathbf{B}\f$ is assumed to be factorized; it is the result of a Cholesky
/// factorization
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with @ref dlaf_create_grid
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param dlaf_desca DLA-Future descriptor of the global matrix \f$\mathbf{A}\f$
/// @param b Local part of the Cholesky factorization of the global matrix \f$\mathbf{B}\f$
/// @param dlaf_descb DLA-Future descriptor of the global matrix \f$\mathbf{B}\f$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global matrix \f$\mathbf{Z}\f$
/// @param dlaf_descz DLA-Future descriptor of the global matrix \f$\mathbf{Z}\f$
/// @return 0 if the eigensolver completed normally
DLAF_EXTERN_C int dlaf_symmetric_generalized_eigensolver_factorized_s(
    const int dlaf_context, const char uplo, float* a, const struct DLAF_descriptor dlaf_desca, float* b,
    const struct DLAF_descriptor dlaf_descb, float* w, float* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_generalized_eigensolver_factorized_s
DLAF_EXTERN_C int dlaf_symmetric_generalized_eigensolver_factorized_d(
    const int dlaf_context, const char uplo, double* a, const struct DLAF_descriptor dlaf_desca,
    double* b, const struct DLAF_descriptor dlaf_descb, double* w, double* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_generalized_eigensolver_factorized_s
DLAF_EXTERN_C int dlaf_hermitian_generalized_eigensolver_factorized_c(
    const int dlaf_context, const char uplo, dlaf_complex_c* a, const struct DLAF_descriptor dlaf_desca,
    dlaf_complex_c* b, const struct DLAF_descriptor dlaf_descb, float* w, dlaf_complex_c* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

/// @copydoc dlaf_symmetric_generalized_eigensolver_factorized_s
DLAF_EXTERN_C int dlaf_hermitian_generalized_eigensolver_factorized_z(
    const int dlaf_context, const char uplo, dlaf_complex_z* a, const struct DLAF_descriptor dlaf_desca,
    dlaf_complex_z* b, const struct DLAF_descriptor dlaf_descb, double* w, dlaf_complex_z* z,
    const struct DLAF_descriptor dlaf_descz) DLAF_NOEXCEPT;

#ifdef DLAF_WITH_SCALAPACK

/// Generalized eigensolver
///
/// @remark This function is only available when DLAF_WITH_SCALAPACK=ON.
///
/// @pre The matrices \f$\mathbf{A}\f$, \f$\mathbf{B}\f$, and \f$\mathbf{Z}\f$ are assumed to be
/// distributed and in host memory. The vector of eigenvalues \f$\mathbf{w}\f$ is assumed to be local
/// (non-distributed) and in host memory. Moving to and from GPU memory is handled internally.
///
/// @pre Submatrices are currently not supported, so @p n is the size of the full matrices
/// \f$\mathbf{A}\f$, \f$\mathbf{B}\f$, and \f$\mathbf{Z}\f$ and @p ia, @p ja, @p ib, @p jb, @p iz, and
/// @p jz need to be 1.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param n order of the sumbatrix \f$\mathbf{A}\f$ used in the computation
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param ia row index of the global matrix \f$\mathbf{A}\f$ identifying the first row of the submatrix
/// $A$, has to be 1
/// @param ja column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix \f$\mathbf{A}\f$
/// @param b Local part of the global matrix \f$\mathbf{B}\f$
/// @param ib row index of the global matrix \f$\mathbf{B}\f$ identifying the first row of the submatrix
/// \f$\mathbf{B}\f$, has to be 1
/// @param jb column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{B}\f$, has to be 1
/// @param descb ScaLAPACK array descriptor of the global matrix \f$\mathbf{B}\f$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global matrix \f$\mathbf{Z}\f$
/// @param iz row index of the global matrix \f$\mathbf{Z}\f$ identifying the first row of the submatrix
/// \f$\mathbf{Z}\f$, has to be 1
/// @param jz column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param descz ScaLAPACK array descriptor of the global matrix \f$\mathbf{Z}\f$
/// @param[out] info 0 if the eigensolver completed normally
DLAF_EXTERN_C void dlaf_pssygvd(const char uplo, const int n, float* a, const int ia, const int ja,
                                const int desca[9], float* b, const int ib, const int jb,
                                const int descb[9], float* w, float* z, const int iz, const int jz,
                                const int descz[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssygvd
DLAF_EXTERN_C void dlaf_pdsygvd(const char uplo, const int n, double* a, const int ia, const int ja,
                                const int desca[9], double* b, const int ib, const int jb,
                                const int descb[9], double* w, double* z, const int iz, const int jz,
                                const int descz[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssygvd
DLAF_EXTERN_C void dlaf_pchegvd(const char uplo, const int n, dlaf_complex_c* a, const int ia,
                                const int ja, const int desca[9], dlaf_complex_c* b, const int ib,
                                const int jb, const int descb[9], float* w, dlaf_complex_c* z,
                                const int iz, const int jz, const int descz[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssygvd
DLAF_EXTERN_C void dlaf_pzhegvd(const char uplo, const int n, dlaf_complex_z* a, const int ia,
                                const int ja, const int desca[9], dlaf_complex_z* b, const int ib,
                                const int jb, const int descb[9], double* w, dlaf_complex_z* z,
                                const int iz, const int jz, const int descz[9], int* info) DLAF_NOEXCEPT;

/// Generalized eigensolver
///
/// @remark This function is only available when DLAF_WITH_SCALAPACK=ON.
///
/// @pre The matrices \f$\mathbf{A}\f$, \f$\mathbf{B}\f$, and \f$\mathbf{Z}\f$ are assumed to be
/// distributed and in host memory. The vector of eigenvalues \f$\mathbf{w}\f$ is assumed to be local
/// (non-distributed) and in host memory. Moving to and from GPU memory is handled internally.
///
/// @pre The matrix \f$\mathbf{B}\f$ is assumed to be factorized; it is the result of a Cholesky
/// factorization
///
/// @pre Submatrices are currently not supported, so @p n is the size of the full matrices
/// \f$\mathbf{A}\f$, \f$\mathbf{B}\f$, and \f$\mathbf{Z}\f$ and @p ia, @p ja, @p ib, @p jb, @p iz, and
/// @p jz need to be 1.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param n order of the sumbatrix \f$\mathbf{A}\f$ used in the computation
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param ia row index of the global matrix \f$\mathbf{A}\f$ identifying the first row of the submatrix
/// $A$, has to be 1
/// @param ja column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix \f$\mathbf{A}\f$
/// @param b Local part of the Cholesky factorization of the global matrix \f$\mathbf{B}\f$
/// @param ib row index of the global matrix \f$\mathbf{B}\f$ identifying the first row of the submatrix
/// \f$\mathbf{B}\f$, has to be 1
/// @param jb column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{B}\f$, has to be 1
/// @param descb ScaLAPACK array descriptor of the global matrix \f$\mathbf{B}\f$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global matrix \f$\mathbf{Z}\f$
/// @param iz row index of the global matrix \f$\mathbf{Z}\f$ identifying the first row of the submatrix
/// \f$\mathbf{Z}\f$, has to be 1
/// @param jz column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param descz ScaLAPACK array descriptor of the global matrix \f$\mathbf{Z}\f$
/// @param[out] info 0 if the eigensolver completed normally
DLAF_EXTERN_C void dlaf_pssygvd_factorized(const char uplo, const int n, float* a, const int ia,
                                           const int ja, const int desca[9], float* b, const int ib,
                                           const int jb, const int descb[9], float* w, float* z,
                                           const int iz, const int jz, const int descz[9],
                                           int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssygvx_factorized
DLAF_EXTERN_C void dlaf_pdsygvd_factorized(const char uplo, const int n, double* a, const int ia,
                                           const int ja, const int desca[9], double* b, const int ib,
                                           const int jb, const int descb[9], double* w, double* z,
                                           const int iz, const int jz, const int descz[9],
                                           int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssygvx_factorized
DLAF_EXTERN_C void dlaf_pchegvd_factorized(const char uplo, const int n, dlaf_complex_c* a, const int ia,
                                           const int ja, const int desca[9], dlaf_complex_c* b,
                                           const int ib, const int jb, const int descb[9], float* w,
                                           dlaf_complex_c* z, const int iz, const int jz,
                                           const int descz[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pssygvx_factorized
DLAF_EXTERN_C void dlaf_pzhegvd_factorized(const char uplo, const int n, dlaf_complex_z* a, const int ia,
                                           const int ja, const int desca[9], dlaf_complex_z* b,
                                           const int ib, const int jb, const int descb[9], double* w,
                                           dlaf_complex_z* z, const int iz, const int jz,
                                           const int descz[9], int* info) DLAF_NOEXCEPT;

#endif
