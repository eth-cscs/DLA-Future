//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

/// Cholesky factorization
///
/// @pre The matrix \f$\mathbf{A}\f$ is assumed to be distributed and in host memory. Moving to and from
/// GPU memory is handled internally.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with @ref dlaf_create_grid
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param dlaf_desca DLA-Future descriptor of the global matrix \f$\mathbf{A}\f$
/// @return 0 if the factorization completed normally
DLAF_EXTERN_C int dlaf_cholesky_factorization_s(const int dlaf_context, const char uplo, float* a,
                                                const struct DLAF_descriptor dlaf_desca) DLAF_NOEXCEPT;

/// @copydoc dlaf_cholesky_factorization_s
DLAF_EXTERN_C int dlaf_cholesky_factorization_d(const int dlaf_context, const char uplo, double* a,
                                                const struct DLAF_descriptor dlaf_desca) DLAF_NOEXCEPT;

/// @copydoc dlaf_cholesky_factorization_s
DLAF_EXTERN_C int dlaf_cholesky_factorization_c(const int dlaf_context, const char uplo,
                                                dlaf_complex_c* a,
                                                const struct DLAF_descriptor dlaf_desca) DLAF_NOEXCEPT;

/// @copydoc dlaf_cholesky_factorization_s
DLAF_EXTERN_C int dlaf_cholesky_factorization_z(const int dlaf_context, const char uplo,
                                                dlaf_complex_z* a,
                                                const struct DLAF_descriptor dlaf_desca) DLAF_NOEXCEPT;

#ifdef DLAF_WITH_SCALAPACK

/// Cholesky decomposition
///
/// @remark This function is only available when DLAF_WITH_SCALAPACK=ON.
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
/// @param uplo indicates whether the upper ('U') or lower ('L') triangular part of the global submatrix
/// \f$\mathbf{A}\f$ is referenced
/// @param n order of the sumbatrix \f$\mathbf{A}\f$ used in the computation
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param ia row index of the global matrix \f$\mathbf{A}\f$ identifying the first row of the submatrix
/// \f$\mathbf{A}\f$, has to be 1
/// @param ja column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix \f$\mathbf{A}\f$
/// @param[out] info 0 if the factorization completed normally
DLAF_EXTERN_C void dlaf_pspotrf(const char uplo, const int n, float* a, const int ia, const int ja,
                                const int desca[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pspotrf
DLAF_EXTERN_C void dlaf_pdpotrf(const char uplo, const int n, double* a, const int ia, const int ja,
                                const int desca[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pspotrf
DLAF_EXTERN_C void dlaf_pcpotrf(const char uplo, const int n, dlaf_complex_c* a, const int ia,
                                const int ja, const int desca[9], int* info) DLAF_NOEXCEPT;

/// @copydoc dlaf_pspotrf
DLAF_EXTERN_C void dlaf_pzpotrf(const char uplo, const int n, dlaf_complex_z* a, const int ia,
                                const int ja, const int desca[9], int* info) DLAF_NOEXCEPT;

#endif
