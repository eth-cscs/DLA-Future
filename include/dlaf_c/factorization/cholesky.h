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
/// The matrix $A$ is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// Submatrices are currently not supported, so @param n is the size of the full matrix $A$
/// and @param ia, and @param ja need to be 1.
///
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// $A$ is referenced
/// @param n order of the sumbatrix $A$ used in the computation
/// @param a Local part of the global matrix $A$
/// @param ia row index of the global matrix $A$ identifying the first row of the submatrix $A$, has to be 1
/// @param ja column index of the global matrix $A$ identifying the firs column of the submatrix $A$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix $A$
/// @param desca ScaLAPACK descriptor of the global matrix
/// @param info 0 if the factorization completed normally
DLAF_EXTERN_C void dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info);

/// \overload
DLAF_EXTERN_C void dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int* info);

/// \overload
DLAF_EXTERN_C void dlaf_pcpotrf(char uplo, int n, dlaf_complex_c* a, int ia, int ja, int* desca,
                                int* info);

/// \overload
DLAF_EXTERN_C void dlaf_pzpotrf(char uplo, int n, dlaf_complex_z* a, int ia, int ja, int* desca,
                                int* info);

#endif

/// Cholesky decomposition
///
/// The matrix $A$ is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with #dlaf_create_grid
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// $A$ is referenced
/// @param a Local part of the global matrix $A$
/// @param dlaf_desca DLA-Duture descriptor of the global matrix $A$
DLAF_EXTERN_C void dlaf_cholesky_d(int dlaf_context, char uplo, double* a,
                                   struct DLAF_descriptor dlaf_desca);

/// \overload
DLAF_EXTERN_C void dlaf_cholesky_s(int dlaf_context, char uplo, float* a,
                                   struct DLAF_descriptor dlaf_desca);

/// \overload
DLAF_EXTERN_C void dlaf_cholesky_c(int dlaf_context, char uplo, dlaf_complex_c* a,
                                   struct DLAF_descriptor dlaf_desca);
/// \overload
DLAF_EXTERN_C void dlaf_cholesky_z(int dlaf_context, char uplo, dlaf_complex_z* a,
                                   struct DLAF_descriptor dlaf_desca);
