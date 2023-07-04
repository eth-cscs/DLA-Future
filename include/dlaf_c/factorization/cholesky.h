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

#ifdef DLAF_WITH_SCALAPACK

/// Cholesky decomposition
///
/// The matrix @param a is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// Submatrices are currently not supported, so @param n is the size of the full matrix
/// and @param ia and @param ja need to be 1.
///
/// @param uplo Specify if upper ('U') or lower ('L') triangular part of @param a will be referenced
/// @param n order of submatrix used in the computation
/// @param a Local part of the global matrix
/// @param ia Global row index denoting the beginning of the submatrix to be operated on, has to be 1
/// @param ja Global column index denoting the beginning of the submatrix to be operated on, has to be 1
/// @param desca ScaLAPACK descriptor of the global matrix
/// @param info 0 if the factorization completed normally
DLAF_EXTERN_C void dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info);

/// \overload
DLAF_EXTERN_C void dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int* info);

#endif

/// Cholesky decomposition
///
/// The matrix @param a is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with dlaf_create_grid
/// @param uplo Specify if upper ('U') or lower ('L') triangular part of @param a will be referenced
/// @param a Local part of the global matrix
/// @param dlaf_desca DLA-Duture descriptor of the global matrix
DLAF_EXTERN_C void dlaf_cholesky_d(int dlaf_context, char uplo, double* a,
                                   struct DLAF_descriptor dlaf_desca);

/// \overload
DLAF_EXTERN_C void dlaf_cholesky_s(int dlaf_context, char uplo, float* a,
                                   struct DLAF_descriptor dlaf_desca);
