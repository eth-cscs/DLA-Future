//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <lapack.hh>
// LAPACKPP includes complex.h which defines the macro I.
// This breaks HPX.
#ifdef I
#undef I
#endif

#ifdef DLAF_WITH_CUDA
#include <cusolverDn.h>

#include <hpx/modules/async_cuda.hpp>
#endif

#include "dlaf/common/assert.h"
#include "dlaf/common/callable_object.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_tile.h"

#ifdef DLAF_WITH_CUDA
#include "dlaf/cusolver/error.h"
#include "dlaf/util_cublas.h"
#endif

namespace dlaf {
namespace tile {
using matrix::Tile;

// See LAPACK documentation for more details.

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
template <class T>
void hegst(const int itype, const blas::Uplo uplo, const Tile<T, Device::CPU>& a,
           const Tile<T, Device::CPU>& b) {
  DLAF_ASSERT(square_size(a), a);
  DLAF_ASSERT(itype >= 1 && itype <= 3, itype);

  auto info = lapack::hegst(itype, uplo, a.size().cols(), a.ptr(), a.ld(), b.ptr(), b.ld());

  DLAF_ASSERT(info == 0, info);
}

/// Copies all elements from Tile a to Tile b.
///
/// @pre @param a and @param b must have the same size (number of elements).
template <class T>
void lacpy(const Tile<const T, Device::CPU>& a, const Tile<T, Device::CPU>& b) {
  DLAF_ASSERT_MODERATE(a.size() == b.size(), a, b);

  const SizeType m = a.size().rows();
  const SizeType n = a.size().cols();

  lapack::lacpy(lapack::MatrixType::General, m, n, a.ptr(), a.ld(), b.ptr(), b.ld());
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

  lapack::lacpy(lapack::MatrixType::General, region.rows(), region.cols(), in.ptr(in_idx), in.ld(),
                out.ptr(out_idx), out.ld());
}

/// Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any
/// element, of a general rectangular matrix.
///
/// @pre a.size().isValid().
template <class T>
dlaf::BaseType<T> lange(const lapack::Norm norm, const Tile<T, Device::CPU>& a) noexcept {
  return lapack::lange(norm, a.size().rows(), a.size().cols(), a.ptr(), a.ld());
}

/// Compute the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any
/// element, of a triangular matrix.
///
/// @pre uplo != blas::Uplo::General,
/// @pre a.size().isValid(),
/// @pre a.size().rows() >= a.size().cols() if uplo == blas::Uplo::Lower,
/// @pre a.size().rows() <= a.size().cols() if uplo == blas::Uplo::Upper.
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

/// Compute the cholesky decomposition of a (with return code).
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @returns info = 0 on success or info > 0 if the tile is not positive definite.
template <class T>
long long potrfInfo(const blas::Uplo uplo, const Tile<T, Device::CPU>& a) {
  DLAF_ASSERT(square_size(a), a);

  auto info = lapack::potrf(uplo, a.size().rows(), a.ptr(), a.ld());
  DLAF_ASSERT_HEAVY(info >= 0, info);

  return info;
}

/// Compute the cholesky decomposition of a.
///
/// Only the upper or lower triangular elements are referenced according to @p uplo.
/// @pre matrix @p a is square,
/// @pre matrix @p a is positive definite.
template <class T>
void potrf(const blas::Uplo uplo, const Tile<T, Device::CPU>& a) noexcept {
  auto info = potrfInfo(uplo, a);

  DLAF_ASSERT(info == 0, info);
}

#ifdef DLAF_WITH_CUDA
namespace internal {
template <typename T>
struct CublasPotrf;

template <>
struct CublasPotrf<float> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnSpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(std::forward<Args>(args)...));
  }
};

template <>
struct CublasPotrf<double> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnDpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnDpotrf_bufferSize(std::forward<Args>(args)...));
  }
};

template <>
struct CublasPotrf<std::complex<float>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnCpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnCpotrf_bufferSize(std::forward<Args>(args)...));
  }
};

template <>
struct CublasPotrf<std::complex<double>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnZpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnZpotrf_bufferSize(std::forward<Args>(args)...));
  }
};
}

namespace internal {
template <class T>
class CusolverPotrfInfo {
  memory::MemoryView<T, Device::GPU> workspace_;
  memory::MemoryView<int, Device::GPU> info_;

public:
  CusolverPotrfInfo(int workspace_size) : workspace_(workspace_size), info_(1) {}

  T* workspace() {
    return workspace_();
  }
  int* info() {
    return info_();
  }
};
}

template <class T>
internal::CusolverPotrfInfo<T> potrfInfo(cusolverDnHandle_t handle, const blas::Uplo uplo,
                                         const matrix::Tile<T, Device::GPU>& a) {
  DLAF_ASSERT(square_size(a), a);
  const int n = a.size().rows();

  int workspace_size;
  internal::CublasPotrf<T>::callBufferSize(handle, util::blasToCublas(uplo), n,
                                           util::blasToCublasCast(a.ptr()), a.ld(), &workspace_size);
  internal::CusolverPotrfInfo<T> info{workspace_size};
  internal::CublasPotrf<T>::call(handle, util::blasToCublas(uplo), n, util::blasToCublasCast(a.ptr()),
                                 a.ld(), util::blasToCublasCast(info.workspace()), workspace_size,
                                 info.info());

  return info;
}

template <class T>
void potrf(cusolverDnHandle_t handle, const blas::Uplo uplo, const matrix::Tile<T, Device::GPU>& a) {
  auto info = potrfInfo(handle, uplo, a);

#ifdef DLAF_ASSERT_HEAVY_ENABLE
  // info does not yet have the correct result, so we manually insert a copy
  // operation from device to host which will happen once the potrf completes
  // (because we use the same stream), and a continuation after the copy to
  // check the value.
  cudaStream_t stream;
  DLAF_CUSOLVER_CALL(cusolverDnGetStream(handle, &stream));
  auto f = hpx::cuda::experimental::detail::get_future_with_event(stream);

  memory::MemoryView<int, Device::CPU> info_host(1);
  DLAF_CUDA_CALL(cudaMemcpyAsync(info_host(), info.info(), sizeof(int), cudaMemcpyDeviceToHost, stream));

  hpx::cuda::experimental::detail::get_future_with_event(stream)
      .then(hpx::launch::sync,
            [info = std::move(info), info_host = std::move(info_host)](hpx::future<void>&&) {
              DLAF_ASSERT_HEAVY(*(info_host()) == 0, *(info_host()));
            });
#endif
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(potrf);

}
}
