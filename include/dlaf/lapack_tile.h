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

#include "lapack.hh"
#include "dlaf/tile.h"

namespace dlaf {
namespace tile {

// See LAPACK documentation for more details.

// Variants that throw an error on failure.

/// @brief Compute the cholesky decomposition of a.
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @throw @c std::invalid_argument if a is not square.
/// @throw @c std::runtime_error if the tile was not positive definite.
template <class T, Device device>
void potrf(blas::Uplo uplo, const Tile<T, device>& a);

// Variants that return info code.

/// @brief Compute the cholesky decomposition of a.
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @returns info = 0 on success or info > 0 if the tile is not positive definite.
/// @throw @c std::runtime_error if the tile was not positive definite.
template <class T, Device device>
long long potrfInfo(blas::Uplo uplo, const Tile<T, device>& a);

#include "dlaf/lapack_tile.ipp"
}
}
