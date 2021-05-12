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

#include "blas.hh"

#include "dlaf/common/callable_object.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_blas.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <blas.hh>

#include "dlaf/cublas/error.h"
#include "dlaf/util_cublas.h"
#endif

namespace dlaf {
namespace tile {
using matrix::Tile;

// See BLAS documentation for more details.

/// Computes general matrix matrix multiplication.
template <class T>
void gemm(const blas::Op op_a, const blas::Op op_b, const T alpha, const Tile<const T, Device::CPU>& a,
          const Tile<const T, Device::CPU>& b, const T beta, const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getGemmSizes(op_a, op_b, a, b, c);
  blas::gemm(blas::Layout::ColMajor, op_a, op_b, s.m, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(),
             beta, c.ptr(), c.ld());
}

/// Computes matrix matrix multiplication where matrix @p a is hermitian (symmetric if T is real).
template <class T>
void hemm(const blas::Side side, const blas::Uplo uplo, const T alpha,
          const Tile<const T, Device::CPU>& a, const Tile<const T, Device::CPU>& b, const T beta,
          const Tile<T, Device::CPU>& c) {
  auto s = tile::internal::getHemmSizes(side, a, b, c);
  blas::hemm(blas::Layout::ColMajor, side, uplo, s.m, s.n, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
             c.ptr(), c.ld());
}

/// Performs a rank 2k update of hermitian (symmetric if T is real) tile a.
template <class T>
void her2k(const blas::Uplo uplo, const blas::Op op, const T alpha, const Tile<const T, Device::CPU>& a,
           const Tile<const T, Device::CPU>& b, const BaseType<T> beta, const Tile<T, Device::CPU>& c) {
  auto s = tile::internal::getHer2kSizes(op, a, b, c);
  blas::her2k(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), b.ptr(), b.ld(), beta,
              c.ptr(), c.ld());
}

/// Performs a rank k update of hermitian (symmetric if T is real) tile @p a.
template <class T>
void herk(const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const Tile<const T, Device::CPU>& a, const BaseType<T> beta,
          const Tile<T, Device::CPU>& c) noexcept {
  auto s = tile::internal::getHerkSizes(op, a, c);
  blas::herk(blas::Layout::ColMajor, uplo, op, s.n, s.k, alpha, a.ptr(), a.ld(), beta, c.ptr(), c.ld());
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
namespace internal {
template <typename T>
struct CublasGemm;

template <>
struct CublasGemm<float> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasSgemm(std::forward<Args>(args)...));
  }
};

template <>
struct CublasGemm<double> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasDgemm(std::forward<Args>(args)...));
  }
};

template <>
struct CublasGemm<std::complex<float>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasCgemm(std::forward<Args>(args)...));
  }
};

template <>
struct CublasGemm<std::complex<double>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasZgemm(std::forward<Args>(args)...));
  }
};

template <typename T>
struct CublasHerk;

template <>
struct CublasHerk<std::complex<float>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasCherk(std::forward<Args>(args)...));
  }
};

template <>
struct CublasHerk<std::complex<double>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasZherk(std::forward<Args>(args)...));
  }
};

template <typename T>
struct CublasTrsm;

template <>
struct CublasTrsm<float> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasStrsm(std::forward<Args>(args)...));
  }
};

template <>
struct CublasTrsm<double> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasDtrsm(std::forward<Args>(args)...));
  }
};

template <>
struct CublasTrsm<std::complex<float>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasCtrsm(std::forward<Args>(args)...));
  }
};

template <>
struct CublasTrsm<std::complex<double>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUBLAS_CALL(cublasZtrsm(std::forward<Args>(args)...));
  }
};
}

/// Computes general matrix matrix multiplication.
template <class T>
void gemm(cublasHandle_t handle, const blas::Op op_a, const blas::Op op_b, const T alpha,
          const matrix::Tile<const T, Device::GPU>& a, const matrix::Tile<const T, Device::GPU>& b,
          const T beta, const matrix::Tile<T, Device::GPU>& c) {
  auto s = tile::internal::getGemmSizes(op_a, op_b, a, b, c);
  internal::CublasGemm<T>::call(handle, util::blasToCublas(op_a), util::blasToCublas(op_b), s.m, s.n,
                                s.k, util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()),
                                a.ld(), util::blasToCublasCast(b.ptr()), b.ld(),
                                util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), c.ld());
}

/// Performs a rank k update of hermitian (symmetric if T is real) tile @p a.
template <class T>
void herk(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const matrix::Tile<const T, Device::GPU>& a, const BaseType<T> beta,
          const matrix::Tile<T, Device::GPU>& c) {
  auto s = tile::internal::getHerkSizes(op, a, c);
  internal::CublasHerk<T>::call(handle, util::blasToCublas(uplo), util::blasToCublas(op), s.n, s.k,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), c.ld());
}

/// Performs a triangular solve.
template <class T>
void trsm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
  auto s = tile::internal::getTrsmSizes(side, a, b);
  internal::CublasTrsm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo),
                                util::blasToCublas(op), util::blasToCublas(diag), s.m, s.n,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(b.ptr()), b.ld());
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(gemm);
DLAF_MAKE_CALLABLE_OBJECT(hemm);
DLAF_MAKE_CALLABLE_OBJECT(her2k);
DLAF_MAKE_CALLABLE_OBJECT(herk);
DLAF_MAKE_CALLABLE_OBJECT(trsm);

}
}
