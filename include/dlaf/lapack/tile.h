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

/// @file

#include <lapack.hh>
// LAPACKPP includes complex.h which defines the macro I.
// This breaks pika.
#ifdef I
#undef I
#endif

#ifdef DLAF_WITH_CUDA
#include <cusolverDn.h>

#include <pika/modules/async_cuda.hpp>
#endif

#include "dlaf/common/assert.h"
#include "dlaf/common/callable_object.h"
#include "dlaf/lapack/enum_output.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"
#include "dlaf/util_lapack.h"
#include "dlaf/util_tile.h"

#ifdef DLAF_WITH_CUDA
#include "dlaf/cusolver/assert_info.h"
#include "dlaf/cusolver/error.h"
#include "dlaf/cusolver/hegst.h"
#include "dlaf/lapack/gpu/laset.h"
#include "dlaf/util_cublas.h"
#endif

namespace dlaf {
namespace tile {
using matrix::Tile;

// See LAPACK documentation for more details.

/// Copies all elements from Tile a to Tile b.
///
/// @pre @param a and @param b must have the same size (number of elements).
///
/// This overload blocks until completion of the algorithm.
template <class T>
void lacpy(const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) {
  DLAF_ASSERT_MODERATE(a.size() == b.size(), a, b);

  const SizeType m = a.size().rows();
  const SizeType n = a.size().cols();

  lapack::lacpy(blas::Uplo::General, m, n, a.ptr(), a.ld(), b.ptr(), b.ld());
}

/// Copies a 2D @param region from tile @param in starting at @param in_idx to tile @param out starting
/// at @param out_idx.
///
/// @pre @param region has to fit within @param in and @param out taking into account the starting
/// indices @param in_idx and @param out_idx.
template <class T>
void lacpy(TileElementSize region, TileElementIndex in_idx, const Tile<const T, Device::CPU>& in,
           TileElementIndex out_idx, const Tile<T, Device::CPU>& out) {
  DLAF_ASSERT_MODERATE(in_idx.isIn(in.size() - region + TileElementSize(1, 1)),
                       "Region goes out of bounds for `in`!", region, in_idx, in);
  DLAF_ASSERT_MODERATE(out_idx.isIn(out.size() - region + TileElementSize(1, 1)),
                       "Region goes out of bounds for `out`!", region, out_idx, out);

  lapack::lacpy(blas::Uplo::General, region.rows(), region.cols(), in.ptr(in_idx), in.ld(),
                out.ptr(out_idx), out.ld());
}

#ifdef DLAF_DOXYGEN

/// Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any
/// element, of a general rectangular matrix.
///
/// @pre a.size().isValid().
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
dlaf::BaseType<T> lange(const dlaf::internal::Policy<B>& p, const lapack::Norm norm,
                        const Tile<T, Device::CPU>& a);

/// \overload lange
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
dlaf::BaseType<T> lange(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload lange
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
dlaf::BaseType<T> lange(const dlaf::internal::Policy<B>& p);

/// Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any
/// element, of a triangular matrix.
///
/// @pre uplo != blas::Uplo::General,
/// @pre a.size().isValid(),
/// @pre a.size().rows() >= a.size().cols() if uplo == blas::Uplo::Lower,
/// @pre a.size().rows() <= a.size().cols() if uplo == blas::Uplo::Upper.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
dlaf::BaseType<T> lantr(const dlaf::internal::Policy<B>& p, const lapack::Norm norm,
                        const blas::Uplo uplo, const blas::Diag diag, const Tile<T, Device::CPU>& a);

/// \overload lantr
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
dlaf::BaseType<T> lantr(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload lantr
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
dlaf::BaseType<T> lantr(const dlaf::internal::Policy<B>& p);

/// Set off-diagonal (@param alpha) and diagonal (@param beta) elements of Tile @param tile.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void laset(const dlaf::internal::Policy<B>& p, const blas::Uplo uplo, T alpha, T beta,
           const Tile<T, Device::CPU>& tile);

/// \overload laset
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
void laset(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload laset
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
void laset(const dlaf::internal::Policy<B>& p);

/// Set zero all the elements of Tile @param tile.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void set0(const dlaf::internal::Policy<B>& p, const Tile<T, Device::CPU>& tile);

/// \overload set0
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
void set0(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload set0
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
void set0(const dlaf::internal::Policy<B>& p);

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
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void hegst(const dlaf::internal::Policy<B>&, const int itype, const blas::Uplo uplo, const Tile<T, D>& a,
           const Tile<T, D>& b);

/// \overload hegst
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto hegst(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload hegst
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto hegst(const dlaf::internal::Policy<B>& p);

/// Compute the cholesky decomposition of a (with return code).
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @returns info = 0 on success or info > 0 if the tile is not positive definite.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
auto potrfInfo(const dlaf::internal::Policy<B>&, const blas::Uplo uplo, const Tile<T, D>& a);

/// \overload potrfInfo
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto potrfInfo(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload potrfInfo
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto potrfInfo(const dlaf::internal::Policy<B>& p);

/// Compute the cholesky decomposition of a.
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @pre matrix @p a is square,
/// @pre matrix @p a is positive definite.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void potrf(const dlaf::internal::Policy<B>& p, const blas::Uplo uplo, const Tile<T, D>& a);

/// \overload potrf
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<pika::execution::experimental::is_sender_v<Sender>>>
auto potrf(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload potrf
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto potrf(const dlaf::internal::Policy<B>& p);

#else

namespace internal {

template <class T>
dlaf::BaseType<T> lange(const lapack::Norm norm, const Tile<T, Device::CPU>& a) noexcept {
  return lapack::lange(norm, a.size().rows(), a.size().cols(), a.ptr(), a.ld());
}

template <class T>
dlaf::BaseType<T> lantr(const lapack::Norm norm, const blas::Uplo uplo, const blas::Diag diag,
                        const Tile<T, Device::CPU>& a) noexcept {
  switch (uplo) {
    case blas::Uplo::Lower:
      DLAF_ASSERT(a.size().rows() >= a.size().cols(), a);
      break;
    case blas::Uplo::Upper:
      DLAF_ASSERT(a.size().rows() <= a.size().cols(), a);
      break;
    case blas::Uplo::General:
      DLAF_ASSERT(blas::Uplo::General == uplo, uplo);
      break;
  }
  return lapack::lantr(norm, uplo, diag, a.size().rows(), a.size().cols(), a.ptr(), a.ld());
}

template <class T>
void laset(const blas::Uplo uplo, T alpha, T beta, const Tile<T, Device::CPU>& tile) {
  const SizeType m = tile.size().rows();
  const SizeType n = tile.size().cols();

  lapack::laset(uplo, m, n, alpha, beta, tile.ptr(), tile.ld());
}

template <class T>
void set0(const Tile<T, Device::CPU>& tile) {
  tile::internal::laset(blas::Uplo::General, static_cast<T>(0.0), static_cast<T>(0.0), tile);
}

template <class T>
void hegst(const int itype, const blas::Uplo uplo, const Tile<T, Device::CPU>& a,
           const Tile<T, Device::CPU>& b) {
  DLAF_ASSERT(square_size(a), a);
  DLAF_ASSERT(square_size(b), b);
  DLAF_ASSERT(a.size() == b.size(), a, b);
  DLAF_ASSERT(itype >= 1 && itype <= 3, itype);

  auto info = lapack::hegst(itype, uplo, a.size().cols(), a.ptr(), a.ld(), b.ptr(), b.ld());

  DLAF_ASSERT(info == 0, info);
}

template <class T>
long long potrfInfo(const blas::Uplo uplo, const Tile<T, Device::CPU>& a) {
  DLAF_ASSERT(square_size(a), a);

  auto info = lapack::potrf(uplo, a.size().rows(), a.ptr(), a.ld());
  DLAF_ASSERT_HEAVY(info >= 0, info);

  return info;
}

template <class T>
void potrf(const blas::Uplo uplo, const Tile<T, Device::CPU>& a) noexcept {
  auto info = potrfInfo(uplo, a);

  DLAF_ASSERT(info == 0, info);
}

#ifdef DLAF_WITH_CUDA
namespace internal {
#define DLAF_DECLARE_CUSOLVER_OP(Name) \
  template <typename T>                \
  struct Cusolver##Name

#define DLAF_DEFINE_CUSOLVER_OP_BUFFER(Name, Type, f)                              \
  template <>                                                                      \
  struct Cusolver##Name<Type> {                                                    \
    template <typename... Args>                                                    \
    static void call(Args&&... args) {                                             \
      DLAF_CUSOLVER_CALL(cusolverDn##f(std::forward<Args>(args)...));              \
    }                                                                              \
    template <typename... Args>                                                    \
    static void callBufferSize(Args&&... args) {                                   \
      DLAF_CUSOLVER_CALL(cusolverDn##f##_bufferSize(std::forward<Args>(args)...)); \
    }                                                                              \
  }

DLAF_DECLARE_CUSOLVER_OP(Hegst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, float, Ssygst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, double, Dsygst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, std::complex<float>, Chegst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, std::complex<double>, Zhegst);

DLAF_DECLARE_CUSOLVER_OP(Potrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, float, Spotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, double, Dpotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, std::complex<float>, Cpotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, std::complex<double>, Zpotrf);
}

namespace internal {
template <class T>
class CusolverInfo {
  memory::MemoryView<T, Device::GPU> workspace_;
  memory::MemoryView<int, Device::GPU> info_;

public:
  CusolverInfo(int workspace_size) : workspace_(workspace_size), info_(1) {}

  T* workspace() {
    return workspace_();
  }
  int* info() {
    return info_();
  }
};

template <class F, class T>
void assertExtendInfo(F assertFunc, cusolverDnHandle_t handle, CusolverInfo<T>&& info) {
  cudaStream_t stream;
  DLAF_CUSOLVER_CALL(cusolverDnGetStream(handle, &stream));
  assertFunc(stream, info.info());
  // Extend info scope to the end of the kernel execution
  auto extend_info = [info = std::move(info)]() {};
  pika::cuda::experimental::detail::get_future_with_event(stream) |
      pika::execution::experimental::then(std::move(extend_info)) |
      pika::execution::experimental::start_detached();
}
}

template <class T>
dlaf::BaseType<T> lange(cusolverDnHandle_t handle, const lapack::Norm norm,
                        const Tile<T, Device::GPU>& a) {
  DLAF_STATIC_UNIMPLEMENTED(T);
  dlaf::internal::silenceUnusedWarningFor(handle, norm, a);
}

template <class T>
dlaf::BaseType<T> lantr(cusolverDnHandle_t handle, const lapack::Norm norm, const blas::Uplo uplo,
                        const blas::Diag diag, const Tile<T, Device::GPU>& a) {
  DLAF_STATIC_UNIMPLEMENTED(T);
  dlaf::internal::silenceUnusedWarningFor(handle, norm, uplo, diag, a);
}

template <class T>
void laset(const blas::Uplo uplo, T alpha, T beta, const Tile<T, Device::GPU>& tile,
           cudaStream_t stream) {
  const SizeType m = tile.size().rows();
  const SizeType n = tile.size().cols();

  gpulapack::laset(util::blasToCublas(uplo), m, n, alpha, beta, tile.ptr(), tile.ld(), stream);
}

template <class T>
void set0(const Tile<T, Device::GPU>& tile, cudaStream_t stream) {
  DLAF_CUDA_CALL(cudaMemset2DAsync(tile.ptr(), sizeof(T) * to_sizet(tile.ld()), 0,
                                   sizeof(T) * to_sizet(tile.size().rows()),
                                   to_sizet(tile.size().cols()), stream));
}

template <class T>
void hegst(cusolverDnHandle_t handle, const int itype, const blas::Uplo uplo,
           const matrix::Tile<T, Device::GPU>& a, const matrix::Tile<T, Device::GPU>& b) {
  DLAF_ASSERT(square_size(a), a);
  DLAF_ASSERT(square_size(b), b);
  DLAF_ASSERT(a.size() == b.size(), a, b);
  const auto n = a.size().rows();

  int workspace_size;
  internal::CusolverHegst<T>::callBufferSize(handle, itype, util::blasToCublas(uplo), to_int(n),
                                             util::blasToCublasCast(a.ptr()), to_int(a.ld()),
                                             util::blasToCublasCast(b.ptr()), to_int(b.ld()),
                                             &workspace_size);
  internal::CusolverInfo<T> info{std::max(1, workspace_size)};
  internal::CusolverHegst<T>::call(handle, itype, util::blasToCublas(uplo), to_int(n),
                                   util::blasToCublasCast(a.ptr()), to_int(a.ld()),
                                   util::blasToCublasCast(b.ptr()), to_int(b.ld()),
                                   util::blasToCublasCast(info.workspace()), info.info());

  assertExtendInfo(dlaf::cusolver::assertInfoHegst, handle, std::move(info));
}

template <class T>
internal::CusolverInfo<T> potrfInfo(cusolverDnHandle_t handle, const blas::Uplo uplo,
                                    const matrix::Tile<T, Device::GPU>& a) {
  DLAF_ASSERT(square_size(a), a);
  const auto n = a.size().rows();

  int workspace_size;
  internal::CusolverPotrf<T>::callBufferSize(handle, util::blasToCublas(uplo), to_int(n),
                                             util::blasToCublasCast(a.ptr()), to_int(a.ld()),
                                             &workspace_size);
  internal::CusolverInfo<T> info{workspace_size};
  internal::CusolverPotrf<T>::call(handle, util::blasToCublas(uplo), to_int(n),
                                   util::blasToCublasCast(a.ptr()), to_int(a.ld()),
                                   util::blasToCublasCast(info.workspace()), workspace_size,
                                   info.info());

  return info;
}

template <class T>
void potrf(cusolverDnHandle_t handle, const blas::Uplo uplo, const matrix::Tile<T, Device::GPU>& a) {
  auto info = potrfInfo(handle, uplo, a);
  assertExtendInfo(dlaf::cusolver::assertInfoHegst, handle, std::move(info));
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(lange);
DLAF_MAKE_CALLABLE_OBJECT(lantr);
DLAF_MAKE_CALLABLE_OBJECT(laset);
DLAF_MAKE_CALLABLE_OBJECT(set0);
DLAF_MAKE_CALLABLE_OBJECT(hegst);
DLAF_MAKE_CALLABLE_OBJECT(potrf);
DLAF_MAKE_CALLABLE_OBJECT(potrfInfo);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(lange, internal::lange_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(lantr, internal::lantr_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(laset, internal::laset_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(set0, internal::set0_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(hegst, internal::hegst_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(potrf, internal::potrf_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(potrfInfo, internal::potrfInfo_o)

#endif
}
}
