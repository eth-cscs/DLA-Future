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
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_tile.h"

namespace dlaf {
namespace tile {
using matrix::Tile;

// See BLAS documentation for more details.

/// Computes general matrix matrix multiplication.
template <class T, Device device>
void gemm(const blas::Op op_a, const blas::Op op_b, const T alpha, const Tile<const T, device>& a,
          const Tile<const T, device>& b, const T beta, const Tile<T, device>& c) noexcept;

/// Computes matrix matrix multiplication where matrix @p a is hermitian (symmetric if T is real).
template <class T, Device device>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha, const Tile<const T, device>& a,
          const Tile<const T, device>& b, const T beta, const Tile<T, device>& c);

/// Performs a rank 2k update of hermitian (symmetric if T is real) tile a.
template <class T, Device device>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, device>& a,
           const Tile<const T, device>& b, const BaseType<T> beta, const Tile<T, device>& c);

/// Performs a rank k update of hermitian (symmetric if T is real) tile @p a.
template <class T, Device device>
void herk(const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const Tile<const T, device>& a, const BaseType<T> beta, const Tile<T, device>& c) noexcept;

/// Performs a triangular solve.
template <class T, Device device>
void trsm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, device>& a, const Tile<T, device>& b) noexcept;

#include "dlaf/blas_tile.tpp"
}
}
