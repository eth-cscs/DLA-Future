//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include "lapack.hh"
// LAPACKPP includes complex.h which defines the macro I.
// This breaks HPX.
#ifdef I
#undef I
#endif

#include "dlaf/matrix/index.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_tile.h"

namespace dlaf {
namespace tile {
using matrix::Tile;

// See LAPACK documentation for more details.

/// Reduce a Hermitian definite generalized eigenproblem to standard form.
///
/// If @p itype = 1, the problem is A*x = lambda*B*x,
/// and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H).
///
/// If @p itype = 2 or 3, the problem is A*B*x = lambda*x or
/// B*A*x = lambda*x, and A is overwritten by U*A*(U**H) or (L**H)*A*L.
/// B must have been previously factorized as (U**H)*U or L*(L**H) by potrf().
///
/// @pre a must be a complex Hermitian matrix or a symmetric real matrix (A),
/// @pre b must be the triangular factor from the Cholesky factorization of B,
/// @throw std::runtime_error if the tile was not positive definite.
template <class T, Device device>
void hegst(const int itype, const blas::Uplo uplo, const Tile<T, device>& a, const Tile<T, device>& b);

/// Copies all elements from Tile a to Tile b.
///
/// @pre @param a and @param b must have the same size (number of elements).
template <class T>
void lacpy(const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b);

/// Copies a 2D @param region from tile @param in starting at @param in_idx to tile @param out starting
/// at @param out_idx.
///
/// @pre @param region has to fit within @param in and @param out taking into account the starting
/// indices @param in_idx and @param out_idx.
template <class T>
void lacpy(TileElementSize region, TileElementIndex in_idx, const Tile<const T, Device::CPU>& in,
           TileElementIndex out_idx, const Tile<T, Device::CPU>& out);

/// Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any
/// element, of a general rectangular matrix.
///
/// @pre a.size().isValid().
template <class T, Device device>
dlaf::BaseType<T> lange(const lapack::Norm norm, const Tile<T, device>& a) noexcept;

/// Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any
/// element, of a triangular matrix.
///
/// @pre uplo != blas::Uplo::General,
/// @pre a.size().isValid(),
/// @pre a.size().rows() >= a.size().cols() if uplo == blas::Uplo::Lower,
/// @pre a.size().rows() <= a.size().cols() if uplo == blas::Uplo::Upper.
template <class T, Device device>
dlaf::BaseType<T> lantr(const lapack::Norm norm, const blas::Uplo uplo, const blas::Diag diag,
                        const Tile<T, device>& a) noexcept;

/// Compute the cholesky decomposition of a.
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @pre matrix @p a is square,
/// @pre matrix @p a is positive definite.
template <class T, Device device>
void potrf(const blas::Uplo uplo, const Tile<T, device>& a) noexcept;

/// Compute the cholesky decomposition of a (with return code).
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @returns info = 0 on success or info > 0 if the tile is not positive definite.
template <class T, Device device>
long long potrfInfo(const blas::Uplo uplo, const Tile<T, device>& a);

#include "dlaf/lapack_tile.tpp"

}
}
