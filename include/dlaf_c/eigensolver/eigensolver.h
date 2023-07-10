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

/// @file

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

#ifdef DLAF_WITH_SCALAPACK

/// Eigensolver
///
/// @remark This function is available only when DLAF_WITH_SCALAPACK=ON.
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
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param n order of the sumbatrix \f$\mathbf{A}\f$ used in the computation
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param ia row index of the global matrix \f$\mathbf{A}\f$ identifying the first row of the submatrix
/// $A$, has to be 1
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
/// @param info 0 if the eigensolver completed normally
DLAF_EXTERN_C void dlaf_pssyevd(char uplo, int n, float* a, int ia, int ja, int* desca, float* w,
                                float* z, int iz, int jz, int* descz, int* info);

/// @copydoc dlaf_pssyevd
DLAF_EXTERN_C void dlaf_pdsyevd(char uplo, int n, double* a, int ia, int ja, int* desca, double* w,
                                double* z, int iz, int jz, int* descz, int* info);

/// @copydoc dlaf_pssyevd
DLAF_EXTERN_C void dlaf_pcheevd(char uplo, int n, dlaf_complex_c* a, int ia, int ja, int* desca,
                                float* w, dlaf_complex_c* z, int iz, int jz, int* descz, int* info);

/// @copydoc dlaf_pssyevd
DLAF_EXTERN_C void dlaf_pzheevd(char uplo, int n, dlaf_complex_z* a, int ia, int ja, int* desca,
                                double* w, dlaf_complex_z* z, int iz, int jz, int* descz, int* info);

#endif

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
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param dlaf_desca DLA-Future descriptor of the global matrix \f$\mathbf{A}\f$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global matrix \f$\mathbf{Z}\f$
/// @param dlaf_descz DLA-Future descriptor of the global matrix \f$\mathbf{Z}\f$
DLAF_EXTERN_C void dlaf_eigensolver_d(int dlaf_context, char uplo, double* a,
                                      struct DLAF_descriptor dlaf_desca, double* w, double* z,
                                      struct DLAF_descriptor dlaf_descz);

/// @copydoc dlaf_eigensolver_d
DLAF_EXTERN_C void dlaf_eigensolver_s(int dlaf_context, char uplo, float* a,
                                      struct DLAF_descriptor dlaf_desca, float* w, float* z,
                                      struct DLAF_descriptor dlaf_descz);

/// @copydoc dlaf_eigensolver_d
DLAF_EXTERN_C void dlaf_eigensolver_z(int dlaf_context, char uplo, dlaf_complex_z* a,
                                      struct DLAF_descriptor dlaf_desca, double* w, dlaf_complex_z* z,
                                      struct DLAF_descriptor dlaf_descz);

/// @copydoc dlaf_eigensolver_d
DLAF_EXTERN_C void dlaf_eigensolver_c(int dlaf_context, char uplo, dlaf_complex_c* a,
                                      struct DLAF_descriptor dlaf_desca, float* w, dlaf_complex_c* z,
                                      struct DLAF_descriptor dlaf_descz);
