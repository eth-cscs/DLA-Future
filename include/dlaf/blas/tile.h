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
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"
#include "dlaf/util_blas.h"

#ifdef DLAF_WITH_GPU
#include "dlaf/gpu/blas/api.h"
#include "dlaf/gpu/blas/error.h"
#include "dlaf/util_cublas.h"
#endif

namespace dlaf {
namespace tile {
using matrix::Tile;

#ifdef DLAF_DOXYGEN

/// Computes general matrix matrix multiplication.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void gemm(const blas::Op op_a, const blas::Op op_b, const T alpha, const Tile<const T, D>& a,
          const Tile<const T, D>& b, const T beta, const Tile<T, D>& c);

/// \overload gemm
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto gemm(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload gemm
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto gemm(const dlaf::internal::Policy<B>& p);

/// Computes matrix matrix multiplication where matrix @p a is hermitian (symmetric if T is real).
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha, const Tile<const T, D>& a,
          const Tile<const T, D>& b, const T beta, const Tile<T, D>& c);

/// \overload hemm
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto hemm(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload hemm
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto hemm(const dlaf::internal::Policy<B>& p);

/// Performs a rank 2k update of hermitian (symmetric if T is real) tile @p a.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, D>& a,
           const Tile<const T, D>& b, const BaseType<T> beta, const Tile<T, D>& c);

/// \overload her2k
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto her2k(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload her2k
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto her2k(const dlaf::internal::Policy<B>& p);

/// Performs a rank k update of hermitian (symmetric if T is real) tile @p a.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void herk(const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha, const Tile<const T, D>& a,
          const BaseType<T> beta, const Tile<T, D>& c);

/// \overload herk
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto herk(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload herk
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto herk(const dlaf::internal::Policy<B>& p);

/// Triangular matrix-matrix multiplication.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void trmm(const dlaf::internal::Policy<B>& policy, const blas::Side side, const blas::Uplo uplo,
          const blas::Op op, const blas::Diag diag, const T alpha, const Tile<const T, D>& a,
          const Tile<T, D>& b);

/// \overload trmm
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto trmm(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload trmm
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto trmm(const dlaf::internal::Policy<B>& p);

/// Triangular matrix-matrix multiplication.
/// Version with 3 tile arguments (different output tile).
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void trmm3(const dlaf::internal::Policy<B>& policy, const blas::Side side, const blas::Uplo uplo,
           const blas::Op op, const blas::Diag diag, const T alpha, const Tile<const T, D>& a,
           const Tile<const T, D>& b, const Tile<T, D>& c);

/// \overload trmm3
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto trmm3(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload trmm3
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto trmm3(const dlaf::internal::Policy<B>& p);

/// Performs a triangular solve.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void trsm(const dlaf::internal::Policy<B>& policy, const blas::Side side, const blas::Uplo uplo,
          const blas::Op op, const blas::Diag diag, const T alpha, const Tile<const T, D>& a,
          const Tile<T, D>& b);

/// \overload trsm
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto trsm(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload trsm
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto trsm(const dlaf::internal::Policy<B>& p);
#else

namespace internal {

template <class T>
void gemm(const blas::Op op_a, const blas::Op op_b, const T alpha, const Tile<const T, Device::CPU>& a,
          const Tile<const T, Device::CPU>& b, const T beta, const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getGemmSizes(op_a, op_b, a, b, c);
  blas::gemm(blas::Layout::ColMajor, op_a, op_b, s.m, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(),
             beta, c.ptr(), c.ld());
}

template <class T>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha,
          const Tile<const T, Device::CPU>& a, const Tile<const T, Device::CPU>& b, const T beta,
          const Tile<T, Device::CPU>& c) {
  auto s = tile::internal::getHemmSizes(side, a, b, c);
  blas::hemm(blas::Layout::ColMajor, side, uplo, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

template <class T>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, Device::CPU>& a,
           const Tile<const T, Device::CPU>& b, const BaseType<T> beta,
           const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getHer2kSizes(op, a, b, c);
  blas::her2k(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
              c.ptr(), c.ld());
}

template <class T>
void herk(const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const Tile<const T, Device::CPU>& a, const BaseType<T> beta,
          const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getHerkSizes(op, a, c);
  blas::herk(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), beta, c.ptr(), c.ld());
}

// Triangular matrix-matrix multiplication.
template <class T>
void trmm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) noexcept {
  auto s = tile::internal::getTrmmSizes(side, a, b);
  blas::trmm(blas::Layout::ColMajor, side, uplo, op, diag, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}

// Triangular matrix-matrix multiplication.
// Version with 3 tile arguments (different output tile).
template <class T>
void trmm3(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
           const T alpha, const Tile<const T, Device::CPU>& a, const Tile<const T, Device::CPU>& b,
           const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getTrmm3Sizes(side, a, b, c);
  DLAF_ASSERT(b.ptr() == nullptr || b.ptr() != c.ptr(), b.ptr(), c.ptr());

  matrix::internal::copy(b, c);
  blas::trmm(blas::Layout::ColMajor, side, uplo, op, diag, s.m, s.n, alpha, a.ptr(), a.ld(), c.ptr(),
             c.ld());
}

template <class T>
void trsm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) noexcept {
  auto s = tile::internal::getTrsmSizes(side, a, b);
  blas::trsm(blas::Layout::ColMajor, side, uplo, op, diag, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}

#ifdef DLAF_WITH_GPU

template <class T>
void gemm(cublasHandle_t handle, const blas::Op op_a, const blas::Op op_b, const T alpha,
          const matrix::Tile<const T, Device::GPU>& a, const matrix::Tile<const T, Device::GPU>& b,
          const T beta, const matrix::Tile<T, Device::GPU>& c) {
  using util::blasToCublas;
  using util::blasToCublasCast;
  auto s = getGemmSizes(op_a, op_b, a, b, c);
  gpublas::Gemm<T>::call(handle, blasToCublas(op_a), blasToCublas(op_b), to_int(s.m), to_int(s.n),
                         to_int(s.k), blasToCublasCast(&alpha), blasToCublasCast(a.ptr()),
                         to_int(a.ld()), blasToCublasCast(b.ptr()), to_int(b.ld()),
                         blasToCublasCast(&beta), blasToCublasCast(c.ptr()), to_int(c.ld()));
}

template <class T>
void hemm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const T alpha,
          const Tile<const T, Device::GPU>& a, const Tile<const T, Device::GPU>& b, const T beta,
          const Tile<T, Device::GPU>& c) {
  using util::blasToCublas;
  using util::blasToCublasCast;
  auto s = getHemmSizes(side, a, b, c);
  gpublas::Hemm<T>::call(handle, blasToCublas(side), blasToCublas(uplo), to_int(s.m), to_int(s.n),
                         blasToCublasCast(&alpha), blasToCublasCast(a.ptr()), to_int(a.ld()),
                         blasToCublasCast(b.ptr()), to_int(b.ld()), blasToCublasCast(&beta),
                         blasToCublasCast(c.ptr()), to_int(c.ld()));
}

template <class T>
void her2k(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const T alpha,
           const matrix::Tile<const T, Device::GPU>& a, const Tile<const T, Device::GPU>& b,
           const BaseType<T> beta, const matrix::Tile<T, Device::GPU>& c) {
  using util::blasToCublas;
  using util::blasToCublasCast;
  auto s = getHer2kSizes(op, a, b, c);
  gpublas::Her2k<T>::call(handle, blasToCublas(uplo), blasToCublas(op), to_int(s.n), to_int(s.k),
                          blasToCublasCast(&alpha), blasToCublasCast(a.ptr()), to_int(a.ld()),
                          blasToCublasCast(b.ptr()), to_int(b.ld()), blasToCublasCast(&beta),
                          blasToCublasCast(c.ptr()), to_int(c.ld()));
}

template <class T>
void herk(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const matrix::Tile<const T, Device::GPU>& a, const BaseType<T> beta,
          const matrix::Tile<T, Device::GPU>& c) {
  using util::blasToCublas;
  using util::blasToCublasCast;
  auto s = getHerkSizes(op, a, c);
  gpublas::Herk<T>::call(handle, blasToCublas(uplo), blasToCublas(op), to_int(s.n), to_int(s.k),
                         blasToCublasCast(&alpha), blasToCublasCast(a.ptr()), to_int(a.ld()),
                         blasToCublasCast(&beta), blasToCublasCast(c.ptr()), to_int(c.ld()));
}

#ifdef DLAF_WITH_CUDA
template <class T>
void trmm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
  using util::blasToCublas;
  using util::blasToCublasCast;
  auto s = tile::internal::getTrmmSizes(side, a, b);

  gpublas::Trmm<T>::call(handle, blasToCublas(side), blasToCublas(uplo), blasToCublas(op),
                         blasToCublas(diag), to_int(s.m), to_int(s.n), blasToCublasCast(&alpha),
                         blasToCublasCast(a.ptr()), to_int(a.ld()), blasToCublasCast(b.ptr()),
                         to_int(b.ld()), blasToCublasCast(b.ptr()), to_int(b.ld()));
}

template <class T>
void trmm3(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
           const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
           const matrix::Tile<const T, Device::GPU>& b, const matrix::Tile<T, Device::GPU>& c) {
  using util::blasToCublas;
  using util::blasToCublasCast;
  auto s = tile::internal::getTrmm3Sizes(side, a, b, c);
  DLAF_ASSERT(b.ptr() == nullptr || b.ptr() != c.ptr(), b.ptr(), c.ptr());

  gpublas::Trmm<T>::call(handle, blasToCublas(side), blasToCublas(uplo), blasToCublas(op),
                         blasToCublas(diag), to_int(s.m), to_int(s.n), blasToCublasCast(&alpha),
                         blasToCublasCast(a.ptr()), to_int(a.ld()), blasToCublasCast(b.ptr()),
                         to_int(b.ld()), blasToCublasCast(c.ptr()), to_int(c.ld()));
}
#elif defined DLAF_WITH_HIP
template <class T>
void trmm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
}

template <class T>
void trmm3(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
           const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
           const matrix::Tile<const T, Device::GPU>& b, const matrix::Tile<T, Device::GPU>& c) {
}
#endif

template <class T>
void trsm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
  using util::blasToCublas;
  using util::blasToCublasCast;
  auto s = getTrsmSizes(side, a, b);
  auto a_ptr =
#ifdef DLAF_WITH_CUDA
      blasToCublasCast(a.ptr());
#elif defined(DLAF_WITH_HIP)
      // The hipblas API requires a non-const argument
      blasToCublasCast(const_cast<T*>(a.ptr()));
#endif
  gpublas::Trsm<T>::call(handle, blasToCublas(side), blasToCublas(uplo), blasToCublas(op),
                         blasToCublas(diag), to_int(s.m), to_int(s.n), blasToCublasCast(&alpha),
                         a_ptr, to_int(a.ld()), blasToCublasCast(b.ptr()), to_int(b.ld()));
}
#endif  // defined(DLAF_WITH_GPU)

DLAF_MAKE_CALLABLE_OBJECT(gemm);
DLAF_MAKE_CALLABLE_OBJECT(hemm);
DLAF_MAKE_CALLABLE_OBJECT(her2k);
DLAF_MAKE_CALLABLE_OBJECT(herk);
DLAF_MAKE_CALLABLE_OBJECT(trmm);
DLAF_MAKE_CALLABLE_OBJECT(trmm3);
DLAF_MAKE_CALLABLE_OBJECT(trsm);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, gemm, internal::gemm_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, hemm, internal::hemm_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, her2k,
                                     internal::her2k_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, herk, internal::herk_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, trmm, internal::trmm_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, trmm3,
                                     internal::trmm3_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(dlaf::internal::TransformDispatchType::Blas, trsm, internal::trsm_o)

#endif
}
}
