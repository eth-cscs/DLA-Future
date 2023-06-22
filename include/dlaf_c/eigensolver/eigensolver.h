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

/// Eigensolver
///
/// The matrix @param a is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// Submatrices are currently not supported, so @param m is the size of the full matrix
/// (ignored, extracted from @param desca).
///
/// @param uplo Specify if upper ('U') or lower ('L') triangular part of @param a will be referenced
/// @param m UNSUPPORTED, ignored
/// @param a Local part of the global matrix
/// @param desca ScaLAPACK descriptor of @param a
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the eigenvectors matrix
/// @param descz ScaLAPACK descriptor of @param z
/// @param info 0 if the factorization completed normally
DLAF_EXTERN_C void dlaf_pssyevd(char uplo, int m, float* a, int* desca, float* w, float* z, int* descz,
                                int* info);

/// Eigensolver
///
/// The matrices @param a and @param z are assumed to be distributed and in host memory.
/// Moving to and from GPU memory is handled internally. The vector of eigenvalues @param w
/// is assumed to be local (non-distributed).
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// Submatrices are currently not supported, so @param m is the size of the full matrices
/// (ignored, extracted from @param desca).
///
/// @param uplo Specify if upper ('U') or lower ('L') triangular part of @param a will be referenced
/// @param m UNSUPPORTED, ignored
/// @param a Local part of the global matrix
/// @param desca ScaLAPACK descriptor of @param a
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the eigenvectors matrix
/// @param descz ScaLAPACK descriptor of @param z
/// @param info 0 if the factorization completed normally
DLAF_EXTERN_C void dlaf_pdsyevd(char uplo, int m, double* a, int* desca, double* w, double* z,
                                int* descz, int* info);

/// Eigensolver
///
/// The matrices @param a and @param z are assumed to be distributed and in host memory.
/// Moving to and from GPU memory is handled internally. The vector of eigenvalues @param w
/// is assumed to be local (non-distributed).
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with dlaf_create_grid
/// @param uplo Specify if upper ('U') or lower ('L') triangular part of @param a will be referenced
/// @param a Local part of the global matrix
/// @param dlaf_desca DLA-Duture descriptor of @param a
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the golbal eigenvalues matrix
/// @param dlaf_descz DLA-Duture descriptor of @param z
DLAF_EXTERN_C void dlaf_eigensolver_s(int dlaf_context, char uplo, float* a,
                                      struct DLAF_descriptor dlaf_desca, float* w, float* z,
                                      struct DLAF_descriptor dlaf_descz);

/// Eigensolver
///
/// The matrices @param a and @param z are assumed to be distributed and in host memory.
/// Moving to and from GPU memory is handled internally. The vector of eigenvalues @param w
/// is assumed to be local (non-distributed).
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with dlaf_create_grid
/// @param uplo Specify if upper ('U') or lower ('L') triangular part of @param a will be referenced
/// @param a Local part of the global matrix
/// @param dlaf_desca DLA-Duture descriptor of @param a
/// @param w Local vector of eigenvalues (non-distributed)
/// @param z Local part of the golbal eigenvalues matrix
/// @param dlaf_descz DLA-Duture descriptor of @param z
DLAF_EXTERN_C void dlaf_eigensolver_d(int dlaf_context, char uplo, double* a,
                                      struct DLAF_descriptor dlaf_desca, double* w, double* z,
                                      struct DLAF_descriptor dlaf_descz);
