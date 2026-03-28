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

/// Triangular matrix multiplication
///
/// Performs one of the matrix operations
///    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
/// where alpha is a scalar, B is an m by n matrix, A is a unit or non-unit,
/// upper or lower triangular matrix and op( A ) is one of
///    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
///
/// @pre The matrix \f$\mathbf{A}\f$ and \f$\mathbf{B}\f$ are assumed to be distributed and in host
/// memory. Moving to and from GPU memory is handled internally.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param dlaf_context context associated to the DLA-Future grid created with @ref dlaf_create_grid
/// @param side specifies whether op(A) appears on the Left ('L') or Right ('R') of matrix B
/// @param uplo indicates whether the matrix A is Upper ('U') or Lower ('L') triangular
/// @param op specifies the form of op(A) used in the multiplication: NoTrans ('N'), Trans ('T'), or
/// ConjTrans ('C')
/// @param diag specifies if A is assumed to be unit triangular ('U') or not ('N')
/// @param alpha scalar multiplier
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param dlaf_desca DLA-Future descriptor of the global matrix \f$\mathbf{A}\f$
/// @param b Local part of the global matrix \f$\mathbf{B}\f$
/// @param dlaf_descb DLA-Future descriptor of the global matrix \f$\mathbf{B}\f$
/// @return 0 if the operation completed normally
DLAF_EXTERN_C int dlaf_triangular_multiplication_s(const int dlaf_context, const char side,
                                                   const char uplo, const char op, const char diag,
                                                   const float alpha, const float* a,
                                                   const struct DLAF_descriptor dlaf_desca, float* b,
                                                   const struct DLAF_descriptor dlaf_descb)
    DLAF_NOEXCEPT;

/// @copydoc dlaf_triangular_multiplication_s
DLAF_EXTERN_C int dlaf_triangular_multiplication_d(const int dlaf_context, const char side,
                                                   const char uplo, const char op, const char diag,
                                                   const double alpha, const double* a,
                                                   const struct DLAF_descriptor dlaf_desca, double* b,
                                                   const struct DLAF_descriptor dlaf_descb)
    DLAF_NOEXCEPT;

/// @copydoc dlaf_triangular_multiplication_s
DLAF_EXTERN_C int dlaf_triangular_multiplication_c(const int dlaf_context, const char side,
                                                   const char uplo, const char op, const char diag,
                                                   const dlaf_complex_c alpha, const dlaf_complex_c* a,
                                                   const struct DLAF_descriptor dlaf_desca,
                                                   dlaf_complex_c* b,
                                                   const struct DLAF_descriptor dlaf_descb)
    DLAF_NOEXCEPT;

/// @copydoc dlaf_triangular_multiplication_s
DLAF_EXTERN_C int dlaf_triangular_multiplication_z(const int dlaf_context, const char side,
                                                   const char uplo, const char op, const char diag,
                                                   const dlaf_complex_z alpha, const dlaf_complex_z* a,
                                                   const struct DLAF_descriptor dlaf_desca,
                                                   dlaf_complex_z* b,
                                                   const struct DLAF_descriptor dlaf_descb)
    DLAF_NOEXCEPT;

#ifdef DLAF_WITH_SCALAPACK

/// Triangular matrix multiplication (ScaLAPACK interface)
///
/// @remark This function is only available when DLAF_WITH_SCALAPACK=ON.
///
/// Performs one of the matrix operations
///    B := alpha*op( A )*B,   or   B := alpha*B*op( A )
/// where alpha is a scalar, B is an m by n matrix, A is a unit or non-unit,
/// upper or lower triangular matrix and op( A ) is one of
///    op( A ) = A   or   op( A ) = A**T   or   op( A ) = A**H.
///
/// @pre The matrix \f$\mathbf{A}\f$ and \f$\mathbf{B}\f$ are assumed to be distributed and in host
/// memory. Moving to and from GPU memory is handled internally.
///
/// @pre Submatrices are currently not supported, so @p ia, @p ja, @p ib, and @p jb need to be 1.
///
/// @post The pika runtime is resumed when this function is called and suspended when the call
/// terminates.
///
/// @param side specifies whether op(A) appears on the Left ('L') or Right ('R') of matrix B
/// @param uplo indicates whether the matrix A is Upper ('U') or Lower ('L') triangular
/// @param op specifies the form of op(A) used in the multiplication: NoTrans ('N'), Trans ('T'), or
/// ConjTrans ('C')
/// @param diag specifies if A is assumed to be unit triangular ('U') or not ('N')
/// @param m number of rows of the matrix B
/// @param n number of columns of the matrix B
/// @param alpha scalar multiplier
/// @param a Local part of the global matrix \f$\mathbf{A}\f$
/// @param ia row index of the global matrix \f$\mathbf{A}\f$ identifying the first row of the submatrix
/// \f$\mathbf{A}\f$, has to be 1
/// @param ja column index of the global matrix \f$\mathbf{A}\f$ identifying the first column of the
/// submatrix \f$\mathbf{A}\f$, has to be 1
/// @param desca ScaLAPACK array descriptor of the global matrix \f$\mathbf{A}\f$
/// @param b Local part of the global matrix \f$\mathbf{B}\f$
/// @param ib row index of the global matrix \f$\mathbf{B}\f$ identifying the first row of the submatrix
/// \f$\mathbf{B}\f$, has to be 1
/// @param jb column index of the global matrix \f$\mathbf{B}\f$ identifying the first column of the
/// submatrix \f$\mathbf{B}\f$, has to be 1
/// @param descb ScaLAPACK array descriptor of the global matrix \f$\mathbf{B}\f$
DLAF_EXTERN_C void dlaf_pstrmm(const char side, const char uplo, const char op, const char diag,
                               const int m, const int n, const float alpha, const float* a,
                               const int ia, const int ja, const int desca[9], float* b, const int ib,
                               const int jb, const int descb[9]) DLAF_NOEXCEPT;

/// @copydoc dlaf_pstrmm
DLAF_EXTERN_C void dlaf_pdtrmm(const char side, const char uplo, const char op, const char diag,
                               const int m, const int n, const double alpha, const double* a,
                               const int ia, const int ja, const int desca[9], double* b, const int ib,
                               const int jb, const int descb[9]) DLAF_NOEXCEPT;

/// @copydoc dlaf_pstrmm
DLAF_EXTERN_C void dlaf_pctrmm(const char side, const char uplo, const char op, const char diag,
                               const int m, const int n, const dlaf_complex_c alpha,
                               const dlaf_complex_c* a, const int ia, const int ja,
                               const int desca[9], dlaf_complex_c* b, const int ib, const int jb,
                               const int descb[9]) DLAF_NOEXCEPT;

/// @copydoc dlaf_pstrmm
DLAF_EXTERN_C void dlaf_pztrmm(const char side, const char uplo, const char op, const char diag,
                               const int m, const int n, const dlaf_complex_z alpha,
                               const dlaf_complex_z* a, const int ia, const int ja,
                               const int desca[9], dlaf_complex_z* b, const int ib, const int jb,
                               const int descb[9]) DLAF_NOEXCEPT;

#endif
