//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file
/// Provides `Tile` wrappers for extra basic linear algebra operations not covered by BLAS.

#include <blas.hh>

#include <dlaf/blas/tile.h>
#include <dlaf/common/callable_object.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/make_sender_algorithm_overloads.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/transform.h>
#include <dlaf/util_matrix.h>

#ifdef DLAF_WITH_GPU
#include <dlaf/gpu/blas/api.h>
#include <dlaf/gpu/blas/error.h>
#include <dlaf/lapack/gpu/add.h>
#include <dlaf/util_cublas.h>
#endif

namespace dlaf {
namespace tile {
using matrix::Tile;

#ifdef DLAF_DOXYGEN

/// Computes A += alpha * B
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void add(T alpha, const matrix::Tile<const T, D>& tile_b, const matrix::Tile<T, D>& tile_a);

/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto add(const dlaf::internal::Policy<B>& p, Sender&& s);

/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto add(const dlaf::internal::Policy<B>& p);

/// Computes A = beta * A
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void scal(T beta, const matrix::Tile<T, D>& tile);

/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto scal(const dlaf::internal::Policy<B>& p, Sender&& s);

/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto scal(const dlaf::internal::Policy<B>& p);

#else

namespace internal {

template <class T>
void add(T alpha, const matrix::Tile<const T, Device::CPU>& tile_b,
         const matrix::Tile<T, Device::CPU>& tile_a) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);
  if (tile_a.size().isEmpty())
    return;
  common::internal::SingleThreadedBlasScope single;
  for (auto j = 0; j < tile_a.size().cols(); ++j)
    blas::axpy(tile_a.size().rows(), alpha, tile_b.ptr({0, j}), 1, tile_a.ptr({0, j}), 1);
}

#ifdef DLAF_WITH_GPU
template <class T>
void add(T alpha, const matrix::Tile<const T, Device::GPU>& tile_b,
         const matrix::Tile<T, Device::GPU>& tile_a, whip::stream_t stream) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);

  gpulapack::add(blas::Uplo::General, tile_a.size().rows(), tile_a.size().cols(), alpha, tile_b.ptr(),
                 tile_b.ld(), tile_a.ptr(), tile_a.ld(), stream);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(add);

template <class T>
void scal(T beta, const matrix::Tile<T, Device::CPU>& tile) {
  common::internal::SingleThreadedBlasScope single;
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, tile.size().rows(),
             tile.size().cols(), 0, T(0), nullptr, tile.ld(), nullptr, 1, beta, tile.ptr(), tile.ld());
}

#ifdef DLAF_WITH_GPU
template <class T>
void scal(cublasHandle_t handle, T beta, const matrix::Tile<T, Device::GPU>& tile) {
  using util::blasToCublasCast;

  const T alpha = 0;
  gpublas::internal::Gemm<T>::call(handle, CUBLAS_OP_N, CUBLAS_OP_N, to_int(tile.size().rows()),
                                   to_int(tile.size().cols()), 0, blasToCublasCast(&alpha),
                                   blasToCublasCast<T*>(nullptr), to_int(tile.ld()),
                                   blasToCublasCast<T*>(nullptr), to_int(1), blasToCublasCast(&beta),
                                   blasToCublasCast(tile.ptr()), to_int(tile.ld()));
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(scal);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Plain, add, internal::add_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, scal, internal::scal_o)

#endif
}
}
