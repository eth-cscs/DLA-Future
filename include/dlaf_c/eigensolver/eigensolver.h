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
/// The matrices $A$ and $Z$ are assumed to be distributed and in host memory.
/// Moving to and from GPU memory is handled internally. The vector of eigenvalues $w$
/// is assumed to be local (non-distributed).
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// Submatrices are currently not supported, so @param n is the size of the full matrix $A$
/// and @param ia, @param ja, @param iz, and @param jz need to be 1.
///
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// $A$ is referenced
/// @param n order of the sumbatrix $A$ used in the computation
/// @param a Local part of the global matrix $A$
/// @param ia row index of the global matrix $A$ identifying the first row of the submatrix $A$, has to be 1
/// @param ja column index of the global matrix $A$ identifying the firs column of the submatrix $A$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix $A$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global eigenvectors matrix $Z$
/// @param iz row index of the global matrix $Z$ identifying the first row of the submatrix $Z$, has to be 1
/// @param jz column index of the global matrix $Z$ identifying the firs column of the submatrix $Z$, has to be 1
/// @param descz ScaLAPACK array descriptor of the global eigenvectors matrix $Z$
/// @param info 0 if the factorization completed normally
DLAF_EXTERN_C void dlaf_pssyevd(char uplo, int n, float* a, int ia, int ja, int* desca, float* w,
                                float* z, int iz, int jz, int* descz, int* info);

/// \overload
DLAF_EXTERN_C void dlaf_pdsyevd(char uplo, int n, double* a, int ia, int ja, int* desca, double* w,
                                double* z, int iz, int jz, int* descz, int* info);

DLAF_EXTERN_C void dlaf_pcheevd(char uplo, int n, dlaf_complex_c* a, int ia, int ja, int* desca,
                                float* w, dlaf_complex_c* z, int iz, int jz, int* descz, int* info);

/// \overload
DLAF_EXTERN_C void dlaf_pzheevd(char uplo, int n, dlaf_complex_z* a, int ia, int ja, int* desca,
                                double* w, dlaf_complex_z* z, int iz, int jz, int* descz, int* info);

#endif

/// Eigensolver
///
/// The matrices $A$ and $Z$ are assumed to be distributed and in host memory.
/// Moving to and from GPU memory is handled internally. The vector of eigenvalues $w$
/// is assumed to be local (non-distributed).
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with  #dlaf_create_grid
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// $A$ is referenced
/// @param a Local part of the global matrix $A$
/// @param dlaf_desca DLA-Future descriptor of the global matrix $A$
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the global eigenvectors matrix $Z$
/// @param dlaf_descz DLA-Future descriptor of the global eigenvectors matrix $Z$
DLAF_EXTERN_C void dlaf_eigensolver_d(int dlaf_context, char uplo, double* a,
                                      struct DLAF_descriptor dlaf_desca, double* w, double* z,
                                      struct DLAF_descriptor dlaf_descz);

/// \overload
DLAF_EXTERN_C void dlaf_eigensolver_s(int dlaf_context, char uplo, float* a,
                                      struct DLAF_descriptor dlaf_desca, float* w, float* z,
                                      struct DLAF_descriptor dlaf_descz);

DLAF_EXTERN_C void dlaf_eigensolver_z(int dlaf_context, char uplo, dlaf_complex_z* a,
                                      struct DLAF_descriptor dlaf_desca, double* w, dlaf_complex_z* z,
                                      struct DLAF_descriptor dlaf_descz);

/// \overload
DLAF_EXTERN_C void dlaf_eigensolver_c(int dlaf_context, char uplo, dlaf_complex_c* a,
                                      struct DLAF_descriptor dlaf_desca, float* w, dlaf_complex_c* z,
                                      struct DLAF_descriptor dlaf_descz);
