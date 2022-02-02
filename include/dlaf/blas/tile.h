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
#include "dlaf/sender/partial_transform.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"
#include "dlaf/util_blas.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>

#include "dlaf/cublas/error.h"
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

/// Performs a matrix-matrix multiplication involving a triangular matrix.
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

/// Performs a matrix-matrix multiplication, involving a triangular matrix.
template <class T>
void trmm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) noexcept {
  auto s = tile::internal::getTrmmSizes(side, a, b);
  blas::trmm(blas::Layout::ColMajor, side, uplo, op, diag, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}

/// Performs a triangular solve.
template <class T>
void trsm(const blas::Side side, const blas::Uplo uplo, const blas::Op op, const blas::Diag diag,
          const T alpha, const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) noexcept {
  auto s = tile::internal::getTrsmSizes(side, a, b);
  blas::trsm(blas::Layout::ColMajor, side, uplo, op, diag, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(),
             b.ld());
}

#ifdef DLAF_WITH_CUDA

#define DLAF_DECLARE_CUBLAS_OP(Name) \
  template <typename T>              \
  struct Cublas##Name

#define DLAF_DEFINE_CUBLAS_OP(Name, Type, f)                    \
  template <>                                                   \
  struct Cublas##Name<Type> {                                   \
    template <typename... Args>                                 \
    static void call(Args&&... args) {                          \
      DLAF_CUBLAS_CALL(cublas##f(std::forward<Args>(args)...)); \
    }                                                           \
  }

DLAF_DECLARE_CUBLAS_OP(Axpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, float, Saxpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, double, Daxpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, std::complex<float>, Caxpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, std::complex<double>, Zaxpy);

DLAF_DECLARE_CUBLAS_OP(Gemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, float, Sgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, double, Dgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, std::complex<float>, Cgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, std::complex<double>, Zgemm);

DLAF_DECLARE_CUBLAS_OP(Hemm);
DLAF_DEFINE_CUBLAS_OP(Hemm, float, Ssymm);
DLAF_DEFINE_CUBLAS_OP(Hemm, double, Dsymm);
DLAF_DEFINE_CUBLAS_OP(Hemm, std::complex<float>, Chemm);
DLAF_DEFINE_CUBLAS_OP(Hemm, std::complex<double>, Zhemm);

DLAF_DECLARE_CUBLAS_OP(Her2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, float, Ssyr2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, double, Dsyr2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, std::complex<float>, Cher2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, std::complex<double>, Zher2k);

DLAF_DECLARE_CUBLAS_OP(Herk);
DLAF_DEFINE_CUBLAS_OP(Herk, float, Ssyrk);
DLAF_DEFINE_CUBLAS_OP(Herk, double, Dsyrk);
DLAF_DEFINE_CUBLAS_OP(Herk, std::complex<float>, Cherk);
DLAF_DEFINE_CUBLAS_OP(Herk, std::complex<double>, Zherk);

DLAF_DECLARE_CUBLAS_OP(Trmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, float, Strmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, double, Dtrmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, std::complex<float>, Ctrmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, std::complex<double>, Ztrmm);

DLAF_DECLARE_CUBLAS_OP(Trsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, float, Strsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, double, Dtrsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, std::complex<float>, Ctrsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, std::complex<double>, Ztrsm);

template <class T>
void gemm(cublasHandle_t handle, const blas::Op op_a, const blas::Op op_b, const T alpha,
          const matrix::Tile<const T, Device::GPU>& a, const matrix::Tile<const T, Device::GPU>& b,
          const T beta, const matrix::Tile<T, Device::GPU>& c) {
  auto s = getGemmSizes(op_a, op_b, a, b, c);
  CublasGemm<T>::call(handle, util::blasToCublas(op_a), util::blasToCublas(op_b), to_int(s.m),
                      to_int(s.n), to_int(s.k), util::blasToCublasCast(&alpha),
                      util::blasToCublasCast(a.ptr()), to_int(a.ld()), util::blasToCublasCast(b.ptr()),
                      to_int(b.ld()), util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()),
                      to_int(c.ld()));
}

template <class T>
void hemm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const T alpha,
          const Tile<const T, Device::GPU>& a, const Tile<const T, Device::GPU>& b, const T beta,
          const Tile<T, Device::GPU>& c) {
  auto s = getHemmSizes(side, a, b, c);
  CublasHemm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo), to_int(s.m),
                      to_int(s.n), util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()),
                      to_int(a.ld()), util::blasToCublasCast(b.ptr()), to_int(b.ld()),
                      util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), to_int(c.ld()));
}

template <class T>
void her2k(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const T alpha,
           const matrix::Tile<const T, Device::GPU>& a, const Tile<const T, Device::GPU>& b,
           const BaseType<T> beta, const matrix::Tile<T, Device::GPU>& c) {
  auto s = getHer2kSizes(op, a, b, c);
  CublasHer2k<T>::call(handle, util::blasToCublas(uplo), util::blasToCublas(op), to_int(s.n),
                       to_int(s.k), util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()),
                       to_int(a.ld()), util::blasToCublasCast(b.ptr()), to_int(b.ld()),
                       util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), to_int(c.ld()));
}

template <class T>
void herk(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const matrix::Tile<const T, Device::GPU>& a, const BaseType<T> beta,
          const matrix::Tile<T, Device::GPU>& c) {
  auto s = getHerkSizes(op, a, c);
  CublasHerk<T>::call(handle, util::blasToCublas(uplo), util::blasToCublas(op), to_int(s.n), to_int(s.k),
                      util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), to_int(a.ld()),
                      util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), to_int(c.ld()));
}

template <class T>
void trmm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
  auto s = tile::internal::getTrmmSizes(side, a, b);
  CublasTrmm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo), util::blasToCublas(op),
                      util::blasToCublas(diag), to_int(s.m), to_int(s.n), util::blasToCublasCast(&alpha),
                      util::blasToCublasCast(a.ptr()), to_int(a.ld()), util::blasToCublasCast(b.ptr()),
                      to_int(b.ld()), util::blasToCublasCast(b.ptr()), to_int(b.ld()));
}

template <class T>
void trsm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
  auto s = getTrsmSizes(side, a, b);
  CublasTrsm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo), util::blasToCublas(op),
                      util::blasToCublas(diag), to_int(s.m), to_int(s.n), util::blasToCublasCast(&alpha),
                      util::blasToCublasCast(a.ptr()), to_int(a.ld()), util::blasToCublasCast(b.ptr()),
                      to_int(b.ld()));
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(gemm);
DLAF_MAKE_CALLABLE_OBJECT(hemm);
DLAF_MAKE_CALLABLE_OBJECT(her2k);
DLAF_MAKE_CALLABLE_OBJECT(herk);
DLAF_MAKE_CALLABLE_OBJECT(trmm);
DLAF_MAKE_CALLABLE_OBJECT(trsm);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(gemm, internal::gemm_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(hemm, internal::hemm_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(her2k, internal::her2k_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(herk, internal::herk_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(trmm, internal::trmm_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(trsm, internal::trsm_o)

#endif
}
}
