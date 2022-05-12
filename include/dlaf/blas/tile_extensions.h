//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>

#include "dlaf/common/callable_object.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"

#ifdef DLAF_WITH_GPU
#include "dlaf/gpu/blas/api.h"
#include "dlaf/gpu/blas/error.h"
#include "dlaf/util_cublas.h"
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

/// \overload add
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto add(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload add
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto add(const dlaf::internal::Policy<B>& p);

#else

namespace internal {

template <class T>
void add(T alpha, const matrix::Tile<const T, Device::CPU>& tile_b,
         const matrix::Tile<T, Device::CPU>& tile_a) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);
  for (auto j = 0; j < tile_a.size().cols(); ++j)
    blas::axpy(tile_a.size().rows(), alpha, tile_b.ptr({0, j}), 1, tile_a.ptr({0, j}), 1);
}

#ifdef DLAF_WITH_GPU
template <class T>
void add(cublasHandle_t handle, T alpha, const matrix::Tile<const T, Device::GPU>& tile_b,
         const matrix::Tile<T, Device::GPU>& tile_a) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);
  for (auto j = 0; j < tile_a.size().cols(); ++j)
    gpublas::Axpy<T>::call(handle, to_int(tile_a.size().rows()), util::blasToCublasCast(&alpha),
                           util::blasToCublasCast(tile_b.ptr({0, j})), 1,
                           util::blasToCublasCast(tile_a.ptr({0, j})), 1);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(add);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, add, internal::add_o)

#endif
}
}
