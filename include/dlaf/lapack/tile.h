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

#include <cstddef>

#include <lapack.hh>
// LAPACKPP includes complex.h which defines the macro I.
// This breaks pika.
#ifdef I
#undef I
#endif

#ifdef DLAF_WITH_GPU
#include <pika/cuda.hpp>
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

#ifdef DLAF_WITH_GPU
#include "dlaf/gpu/lapack/api.h"
#include "dlaf/gpu/lapack/assert_info.h"
#include "dlaf/gpu/lapack/error.h"
#include "dlaf/lapack/gpu/laset.h"
#include "dlaf/util_cublas.h"
#endif
// hegst functions get exposed in rocSOLVER
#ifdef DLAF_WITH_CUDA
#include "dlaf/gpu/cusolver/hegst.h"
#elif defined(DLAF_WITH_HIP)
#include "dlaf/gpu/blas/error.h"
#include "dlaf/util_rocblas.h"
#endif

namespace dlaf::tile {
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
                        const Tile<T, D>& a);

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
                        const blas::Uplo uplo, const blas::Diag diag, const Tile<T, D>& a);

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
           const Tile<T, D>& tile);

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
void set0(const dlaf::internal::Policy<B>& p, const Tile<T, D>& tile);

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

/// Computes the eigenvalues and eigenvectors of a real tridiagonal symmetric matrix using the divide &
/// conquer algorithm.
///
/// @param tridiag is `n x 2`. On entry stores the tridiagonal symmetric matrix: the diagonal in the
/// first column, the off-diagonal in the second column. The last entry of the second column is unused.
/// On exit stores the eigenvalues in the 1st column in ascending order.
///
/// @param evecs is `n x n`. On exit stores the eigenvectors of the tridiagonal symmetrix matrix. The
/// order of the eigenvectors follows that of the eigenvalues.
///
/// This overload blocks until completion of the algorithm.
template <Backend B, class T, Device D>
void stedc(const dlaf::internal::Policy<B>& p, const Tile<BaseType<T>, D>& tridiag,
           const Tile<T, D>& evecs);

/// \overload stedc
///
/// This overload takes a policy argument and a sender which must send all required arguments for the
/// algorithm. Returns a sender which signals a connected receiver when the algorithm is done.
template <Backend B, typename Sender,
          typename = std::enable_if_t<hpx::execution::experimental::is_sender_v<Sender>>>
auto stedc(const dlaf::internal::Policy<B>& p, Sender&& s);

/// \overload stedc
///
/// This overload partially applies the algorithm with a policy for later use with operator| with a
/// sender on the left-hand side.
template <Backend B>
auto stedc(const dlaf::internal::Policy<B>& p);

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

template <class T>
void stedc(const Tile<BaseType<T>, Device::CPU>& tridiag, const Tile<T, Device::CPU>& evecs) {
  DLAF_ASSERT(square_size(evecs), evecs);
  DLAF_ASSERT(tridiag.size().rows() == evecs.size().rows(), tridiag, evecs);
  DLAF_ASSERT(tridiag.size().cols() == 2, tridiag);

  // In lapackpp see `util.hh` and `job_comp2char()` and `enum class Job`
  // Note that `lapack::Job::Vec` corresponds to `compz=I` in the LAPACK

  // compz, n, D, E, Z, ldz
  lapack::stedc(lapack::Job::Vec, evecs.size().rows(), tridiag.ptr(),
                tridiag.ptr(TileElementIndex(0, 1)), evecs.ptr(), evecs.ld());
}

template <class T>
void scaleCol(T alpha, SizeType col, const Tile<T, Device::CPU>& tile) {
  DLAF_ASSERT(col >= 0, col);
  DLAF_ASSERT(tile.size().cols() > col, tile, col);
  blas::scal(tile.size().rows(), alpha, tile.ptr(TileElementIndex(0, col)), 1);
}

#ifdef DLAF_WITH_GPU
namespace internal {
#define DLAF_DECLARE_GPULAPACK_OP(Name) \
  template <typename T>                 \
  struct Cusolver##Name

DLAF_DECLARE_GPULAPACK_OP(Hegst);
DLAF_DECLARE_GPULAPACK_OP(Potrf);

#ifdef DLAF_WITH_CUDA

#define DLAF_DEFINE_CUSOLVER_OP_BUFFER(Name, Type, f)                                      \
  template <>                                                                              \
  struct Cusolver##Name<Type> {                                                            \
    template <typename... Args>                                                            \
    static void call(Args&&... args) {                                                     \
      DLAF_GPULAPACK_CHECK_ERROR(cusolverDn##f(std::forward<Args>(args)...));              \
    }                                                                                      \
    template <typename... Args>                                                            \
    static void callBufferSize(Args&&... args) {                                           \
      DLAF_GPULAPACK_CHECK_ERROR(cusolverDn##f##_bufferSize(std::forward<Args>(args)...)); \
    }                                                                                      \
  }

DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, float, Ssygst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, double, Dsygst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, std::complex<float>, Chegst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, std::complex<double>, Zhegst);

DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, float, Spotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, double, Dpotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, std::complex<float>, Cpotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, std::complex<double>, Zpotrf);

#elif defined(DLAF_WITH_HIP)

#define DLAF_GET_ROCSOLVER_WORKSPACE(f)                                                               \
  [&]() {                                                                                             \
    std::size_t workspace_size;                                                                       \
    DLAF_GPULAPACK_CHECK_ERROR(                                                                       \
        rocblas_start_device_memory_size_query(static_cast<rocblas_handle>(handle)));                 \
    DLAF_GPULAPACK_CHECK_ERROR(rocsolver_##f(handle, std::forward<Args>(args)...));                   \
    DLAF_GPULAPACK_CHECK_ERROR(                                                                       \
        rocblas_stop_device_memory_size_query(static_cast<rocblas_handle>(handle), &workspace_size)); \
    return ::dlaf::memory::MemoryView<std::byte, Device::GPU>(to_int(workspace_size));                \
  }();

inline void extendROCSolverWorkspace(cusolverDnHandle_t handle,
                                     ::dlaf::memory::MemoryView<std::byte, Device::GPU>&& workspace) {
  cudaStream_t stream;
  DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));
  auto f = [workspace = std::move(workspace)](cudaError_t status) { DLAF_GPU_CHECK_ERROR(status); };
  pika::cuda::experimental::detail::add_event_callback(std::move(f), stream);
}

#define DLAF_DEFINE_CUSOLVER_OP_BUFFER(Name, Type, f)                                 \
  template <>                                                                         \
  struct Cusolver##Name<Type> {                                                       \
    template <typename... Args>                                                       \
    static void call(cusolverDnHandle_t handle, Args&&... args) {                     \
      auto workspace = DLAF_GET_ROCSOLVER_WORKSPACE(f);                               \
      DLAF_GPULAPACK_CHECK_ERROR(                                                     \
          rocblas_set_workspace(handle, workspace(), to_sizet(workspace.size())));    \
      DLAF_GPULAPACK_CHECK_ERROR(rocsolver_##f(handle, std::forward<Args>(args)...)); \
      DLAF_GPULAPACK_CHECK_ERROR(rocblas_set_workspace(handle, nullptr, 0));          \
      extendROCSolverWorkspace(handle, std::move(workspace));                         \
    }                                                                                 \
  }

DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, float, ssygst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, double, dsygst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, std::complex<float>, chegst);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Hegst, std::complex<double>, zhegst);

DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, float, spotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, double, dpotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, std::complex<float>, cpotrf);
DLAF_DEFINE_CUSOLVER_OP_BUFFER(Potrf, std::complex<double>, zpotrf);

#endif
}

namespace internal {
template <class T>
class CusolverInfo {
#ifdef DLAF_WITH_CUDA
  memory::MemoryView<T, Device::GPU> workspace_;
#endif
  memory::MemoryView<int, Device::GPU> info_;

public:
  CusolverInfo(int workspace_size)
      :
#ifdef DLAF_WITH_CUDA
        workspace_(workspace_size),
#endif
        info_(1) {
  }
  CusolverInfo() : info_(1) {}

#ifdef DLAF_WITH_CUDA
  T* workspace() {
    return workspace_();
  }
#endif
  int* info() {
    return info_();
  }
};

template <class F, class T>
void assertExtendInfo(F assertFunc, cusolverDnHandle_t handle, CusolverInfo<T>&& info) {
  cudaStream_t stream;
  DLAF_GPULAPACK_CHECK_ERROR(cusolverDnGetStream(handle, &stream));
  assertFunc(stream, info.info());
  // Extend info scope to the end of the kernel execution
  auto extend_info = [info = std::move(info)](cudaError_t status) { DLAF_GPU_CHECK_ERROR(status); };
  pika::cuda::experimental::detail::add_event_callback(std::move(extend_info), stream);
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

  gpulapack::laset(uplo, m, n, alpha, beta, tile.ptr(), tile.ld(), stream);
}

template <class T>
void set0(const Tile<T, Device::GPU>& tile, cudaStream_t stream) {
  DLAF_GPU_CHECK_ERROR(cudaMemset2DAsync(tile.ptr(), sizeof(T) * to_sizet(tile.ld()), 0,
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

#ifdef DLAF_WITH_CUDA
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

  assertExtendInfo(dlaf::gpulapack::internal::assertInfoHegst, handle, std::move(info));
#elif defined(DLAF_WITH_HIP)
  internal::CusolverHegst<T>::call(handle, util::blasToRocblas(itype), util::blasToRocblas(uplo),
                                   to_int(n), util::blasToRocblasCast(a.ptr()), to_int(a.ld()),
                                   util::blasToRocblasCast(b.ptr()), to_int(b.ld()));
#endif
}

template <class T>
internal::CusolverInfo<T> potrfInfo(cusolverDnHandle_t handle, const blas::Uplo uplo,
                                    const matrix::Tile<T, Device::GPU>& a) {
  DLAF_ASSERT(square_size(a), a);
  const auto n = a.size().rows();

#ifdef DLAF_WITH_CUDA
  int workspace_size;
  internal::CusolverPotrf<T>::callBufferSize(handle, util::blasToCublas(uplo), to_int(n),
                                             util::blasToCublasCast(a.ptr()), to_int(a.ld()),
                                             &workspace_size);
  internal::CusolverInfo<T> info{workspace_size};
  internal::CusolverPotrf<T>::call(handle, util::blasToCublas(uplo), to_int(n),
                                   util::blasToCublasCast(a.ptr()), to_int(a.ld()),
                                   util::blasToCublasCast(info.workspace()), workspace_size,
                                   info.info());
#elif defined(DLAF_WITH_HIP)
  internal::CusolverInfo<T> info{};
  internal::CusolverPotrf<T>::call(handle, util::blasToRocblas(uplo), to_int(n),
                                   util::blasToRocblasCast(a.ptr()), to_int(a.ld()), info.info());
#endif

  return info;
}

template <class T>
void potrf(cusolverDnHandle_t handle, const blas::Uplo uplo, const matrix::Tile<T, Device::GPU>& a) {
  auto info = potrfInfo(handle, uplo, a);
  assertExtendInfo(dlaf::gpulapack::internal::assertInfoPotrf, handle, std::move(info));
}

template <class T>
void stedc(cusolverDnHandle_t, const Tile<BaseType<T>, Device::CPU>&, const Tile<T, Device::CPU>&) {
  DLAF_STATIC_UNIMPLEMENTED(T);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(lange);
DLAF_MAKE_CALLABLE_OBJECT(lantr);
DLAF_MAKE_CALLABLE_OBJECT(laset);
DLAF_MAKE_CALLABLE_OBJECT(set0);
DLAF_MAKE_CALLABLE_OBJECT(hegst);
DLAF_MAKE_CALLABLE_OBJECT(potrf);
DLAF_MAKE_CALLABLE_OBJECT(potrfInfo);
DLAF_MAKE_CALLABLE_OBJECT(stedc);
DLAF_MAKE_CALLABLE_OBJECT(scaleCol);
}

DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Lapack, lange,
                                     internal::lange_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Lapack, lantr,
                                     internal::lantr_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Plain, laset,
                                     internal::laset_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Plain, set0,
                                     internal::set0_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Lapack, hegst,
                                     internal::hegst_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Lapack, potrf,
                                     internal::potrf_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Lapack, potrfInfo,
                                     internal::potrfInfo_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Lapack, stedc,
                                     internal::stedc_o)
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(::dlaf::internal::TransformDispatchType::Lapack, scaleCol,
                                     internal::scaleCol_o)

#endif
}
