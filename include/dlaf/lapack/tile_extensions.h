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
/// Provides `Tile` wrappers for extra LAPACK operations.

#include <blas.hh>

#include <dlaf/blas/tile.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/callable_object.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/make_sender_algorithm_overloads.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/transform.h>
#include <dlaf/types.h>
#include <dlaf/util_lapack.h>
#include <dlaf/util_tile.h>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>

#include <pika/cuda.hpp>

#include <dlaf/gpu/lapack/api.h>
#include <dlaf/gpu/lapack/error.h>
#include <dlaf/lapack/gpu/lacpy.h>
#include <dlaf/lapack/gpu/laset.h>
#include <dlaf/util_cublas.h>
#endif

namespace dlaf::tile {
using matrix::Tile;

#ifdef DLAF_DOXYGEN

/// Assemble the cholesky inverse computing L^H L or U U^H for the triangular tile @a.
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
///
/// This overload blocks until completion of the algorithm.
/// @pre @a is the inverse of the cholesky factor.
template <Backend B, class T, Device D>
auto lauum_workspace(const dlaf::internal::Policy<B>&, const blas::Uplo uplo,
                     const Tile<T, D>& const a Tile<T, D>& ws);

/// \overload lauum
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto lauum_workspace(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload lauum
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto lauum_workspace(const dlaf::internal::Policy<B>& p);

/// Compute the the inverse of the triangular tile @p a.
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @pre matrix @p a is square,
/// @pre matrix @p a is non-singular.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void trtri_workspace(const dlaf::internal::Policy<B>& p, const blas::Uplo uplo, const blas::Diag diag,
                     const Tile<T, D>& a, const Tile<T, D>& ws);

/// \overload trtri
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto trtri_workspace(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload trtri
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto trtri_workspace(const dlaf::internal::Policy<B>& p);

#else

namespace internal {

template <class T>
void lauum_workspace(const blas::Uplo uplo, const Tile<T, Device::CPU>& a,
                     const Tile<T, Device::CPU>& ws) {
  DLAF_ASSERT(square_size(a), a);

  const auto ws_sub = ws.subTileReference({{0, 0}, a.size()});
  const blas::Side side = uplo == blas::Uplo::Lower ? blas::Side::Left : blas::Side::Right;

  common::internal::SingleThreadedBlasScope single;
  set0(ws_sub);
  lapack::lacpy(uplo, a.size().rows(), a.size().cols(), a.ptr(), a.ld(), ws_sub.ptr(), ws_sub.ld());
  trmm(side, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, static_cast<T>(1.), a, ws_sub);
  lapack::lacpy(uplo, a.size().rows(), a.size().cols(), ws_sub.ptr(), ws_sub.ld(), a.ptr(), a.ld());
}

template <class T>
void trtri_workspace(const blas::Uplo uplo, const blas::Diag diag, const Tile<T, Device::CPU>& a,
                     const Tile<T, Device::CPU>& ws) noexcept {
  DLAF_ASSERT(square_size(a), a);

  if (a.size().rows() == 0 || (a.size().rows() == 1 && diag == blas::Diag::Unit))
    return;

  const auto ws_sub = ws.subTileReference({{0, 0}, a.size()});

  common::internal::SingleThreadedBlasScope single;
  laset(blas::Uplo::General, static_cast<T>(0.0), static_cast<T>(1.0), ws_sub);
  trsm(blas::Side::Left, uplo, blas::Op::NoTrans, diag, static_cast<T>(1.), a, ws_sub);
  if (diag == blas::Diag::Unit) {
    TileElementIndex offset =
        uplo == blas::Uplo::Lower ? TileElementIndex{1, 0} : TileElementIndex{0, 1};
    lapack::lacpy(uplo, a.size().rows() - 1, a.size().cols() - 1, ws_sub.ptr(offset), ws_sub.ld(),
                  a.ptr(offset), a.ld());
  }
  else
    lapack::lacpy(uplo, a.size().rows(), a.size().cols(), ws_sub.ptr(), ws_sub.ld(), a.ptr(), a.ld());
}

#ifdef DLAF_WITH_GPU

template <class T>
void lauum_workspace(cublasHandle_t handle, const blas::Uplo uplo, const matrix::Tile<T, Device::GPU>& a,
                     const matrix::Tile<T, Device::GPU>& ws) {
  DLAF_ASSERT(square_size(a), a);

  whip::stream_t stream;
  DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));

  const auto ws_sub = ws.subTileReference({{0, 0}, a.size()});

  const blas::Side side = uplo == blas::Uplo::Lower ? blas::Side::Left : blas::Side::Right;
  set0(ws_sub, stream);
  gpulapack::lacpy(uplo, a.size().rows(), a.size().cols(), a.ptr(), a.ld(), ws_sub.ptr(), ws_sub.ld(),
                   stream);
  trmm(handle, side, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, static_cast<T>(1.), a, ws_sub);
  gpulapack::lacpy(uplo, a.size().rows(), a.size().cols(), ws_sub.ptr(), ws_sub.ld(), a.ptr(), a.ld(),
                   stream);
}

template <class T>
void trtri_workspace(cublasHandle_t handle, const blas::Uplo uplo, const blas::Diag diag,
                     const matrix::Tile<T, Device::GPU>& a, const matrix::Tile<T, Device::GPU>& ws) {
  DLAF_ASSERT(square_size(a), a);

  if (a.size().rows() == 0 || (a.size().rows() == 1 && diag == blas::Diag::Unit))
    return;

  whip::stream_t stream;
  DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));

  const auto ws_sub = ws.subTileReference({{0, 0}, a.size()});

  laset(blas::Uplo::General, static_cast<T>(0.0), static_cast<T>(1.0), ws_sub, stream);
  trsm(handle, blas::Side::Left, uplo, blas::Op::NoTrans, diag, static_cast<T>(1.), a, ws_sub);
  if (diag == blas::Diag::Unit) {
    TileElementIndex offset =
        uplo == blas::Uplo::Lower ? TileElementIndex{1, 0} : TileElementIndex{0, 1};
    gpulapack::lacpy(uplo, a.size().rows() - 1, a.size().cols() - 1, ws_sub.ptr(offset), ws_sub.ld(),
                     a.ptr(offset), a.ld(), stream);
  }
  else
    gpulapack::lacpy(uplo, a.size().rows(), a.size().cols(), ws_sub.ptr(), ws_sub.ld(), a.ptr(), a.ld(),
                     stream);
}

#endif

DLAF_MAKE_CALLABLE_OBJECT(lauum_workspace);
DLAF_MAKE_CALLABLE_OBJECT(trtri_workspace);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Blas, lauum_workspace,
                                     internal::lauum_workspace_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Blas, trtri_workspace,
                                     internal::trtri_workspace_o)
#endif
}
