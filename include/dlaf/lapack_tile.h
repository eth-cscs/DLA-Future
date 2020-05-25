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

#include "dlaf/tile.h"

namespace dlaf {
namespace tile {

// See LAPACK documentation for more details.

/// Copies all elements from Tile a to Tile b
///
/// @pre @param a and @param b must have the same size (number of elements)
template <class T>
void lacpy(const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b);

/// Compute the cholesky decomposition of a.
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @throw std::invalid_argument if a is not square.
/// @throw std::runtime_error if the tile was not positive definite.
template <class T, Device device>
void potrf(blas::Uplo uplo, const Tile<T, device>& a);

/// Compute the cholesky decomposition of a. (with return code)
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @returns info = 0 on success or info > 0 if the tile is not positive definite.
/// @throw std::runtime_error if the tile was not positive definite.
template <class T, Device device>
long long potrfInfo(blas::Uplo uplo, const Tile<T, device>& a);

}
}

#include "dlaf/lapack_tile.tpp"
