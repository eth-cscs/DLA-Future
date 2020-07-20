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

#include "blas.hh"
#include "dlaf/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace tile {

// See BLAS documentation for more details.

/// Computes general matrix matrix multiplication.
template <class T, Device device>
void gemm(blas::Op op_a, blas::Op op_b, T alpha, const Tile<const T, device>& a,
          const Tile<const T, device>& b, T beta, const Tile<T, device>& c) noexcept;

/// Performs a rank k update of hermitian (symmetric if T is real) tile a.
template <class T, Device device>
void herk(blas::Uplo uplo, blas::Op op, BaseType<T> alpha, const Tile<const T, device>& a,
          BaseType<T> beta, const Tile<T, device>& c) noexcept;

/// Performs a triangular solve.
template <class T, Device device>
void trsm(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
          const Tile<const T, device>& a, const Tile<T, device>& b) noexcept;

#include "dlaf/blas_tile.tpp"
}
}
