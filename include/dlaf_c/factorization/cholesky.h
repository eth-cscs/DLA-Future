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

/// Cholesky decomposition
///
/// @remark This function is available only when DLAF_WITH_SCALAPACK=ON.
///
/// @pre The matrix \f$\mathbf{A}\f$ is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// @pre Submatrices are currently not supported, so @p n is the size of the full matrix
/// \f$\mathbf{A}\f$ and @p ia, and @p ja need to be 1.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param n order of the sumbatrix \f$\mathbf{A}\f$ used in the computation
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param ia row index of the global matrix \f$\mathbf{A}\f$ identifying the first row of the submatrix
/// \f$\mathbf{A}\f$, has to be 1
/// @param ja column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix \f$\mathbf{A}\f$
/// @param[out] info 0 if the factorization completed normally
DLAF_EXTERN_C void dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info);

/// @copydoc dlaf_pdpotrf
DLAF_EXTERN_C void dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int* info);

/// @copydoc dlaf_pdpotrf
DLAF_EXTERN_C void dlaf_pcpotrf(char uplo, int n, dlaf_complex_c* a, int ia, int ja, int* desca,
                                int* info);

/// @copydoc dlaf_pdpotrf
DLAF_EXTERN_C void dlaf_pzpotrf(char uplo, int n, dlaf_complex_z* a, int ia, int ja, int* desca,
                                int* info);

#endif

/// Cholesky decomposition
///
/// @pre The matrix \f$\mathbf{A}\f$ is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with @ref dlaf_create_grid
/// @param uplo indicates wheather the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param dlaf_desca DLA-Future descriptor of the global matrix \f$\mathbf{A}\f$
/// @return 0 if the factorization completed normally
DLAF_EXTERN_C int dlaf_cholesky_d(int dlaf_context, char uplo, double* a,
                                  struct DLAF_descriptor dlaf_desca);

/// @copydoc dlaf_cholesky_d
DLAF_EXTERN_C int dlaf_cholesky_s(int dlaf_context, char uplo, float* a,
                                  struct DLAF_descriptor dlaf_desca);

/// @copydoc dlaf_cholesky_d
DLAF_EXTERN_C int dlaf_cholesky_c(int dlaf_context, char uplo, dlaf_complex_c* a,
                                  struct DLAF_descriptor dlaf_desca);
/// @copydoc dlaf_cholesky_d
DLAF_EXTERN_C int dlaf_cholesky_z(int dlaf_context, char uplo, dlaf_complex_z* a,
                                  struct DLAF_descriptor dlaf_desca);
