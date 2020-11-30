//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#ifdef DLAF_WITH_CUDA

#include <cublas_v2.h>
#include <blas.hh>

#include "dlaf/cublas/error.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_cublas.h"

namespace dlaf {
namespace tile {
namespace internal {
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

}

template <class T>
void cublasTrsm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
                const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
                const matrix::Tile<T, Device::GPU>& b) {
  SizeType m = b.size().rows();
  SizeType n = b.size().cols();

  DLAF_ASSERT(a.size().rows() == a.size().cols(), "`a` is not square!", a);

  auto left_side = (side == blas::Side::Left ? m : n);
  DLAF_ASSERT(a.size().rows() == left_side, "`a` has an invalid size!", a, left_side);

  internal::CublasTrsm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo),
                                util::blasToCublas(op), util::blasToCublas(diag), m, n,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(b.ptr()), b.ld());
}

template <class T>
void cublasGemm(cublasHandle_t handle, const blas::Op op_a, const blas::Op op_b, const T alpha,
                const matrix::Tile<const T, Device::GPU>& a, const matrix::Tile<const T, Device::GPU>& b,
                const T beta, const matrix::Tile<T, Device::GPU>& c) {
  SizeType m;
  SizeType k;
  if (op_a == blas::Op::NoTrans) {
    m = a.size().rows();
    k = a.size().cols();
  }
  else {
    m = a.size().cols();
    k = a.size().rows();
  }
  SizeType k2;
  SizeType n;
  if (op_b == blas::Op::NoTrans) {
    k2 = b.size().rows();
    n = b.size().cols();
  }
  else {
    k2 = b.size().cols();
    n = b.size().rows();
  }

  DLAF_ASSERT(m == c.size().rows(), "`m` cannot be determined!", m, c);
  DLAF_ASSERT(n == c.size().cols(), "`n` cannot be determined!", n, c);
  DLAF_ASSERT(k == k2, "`k` cannot be determined!", k, k2);

  internal::CublasGemm<T>::call(handle, util::blasToCublas(op_a), util::blasToCublas(op_b), m, n, k,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(b.ptr()), b.ld(), util::blasToCublasCast(&beta),
                                util::blasToCublasCast(c.ptr()), c.ld());
}

template <class T>
void cublasHerk(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
                const matrix::Tile<const T, Device::GPU>& a, const BaseType<T> beta,
                const matrix::Tile<T, Device::GPU>& c) {
  SizeType n;
  SizeType k;
  if (op == blas::Op::NoTrans) {
    n = a.size().rows();
    k = a.size().cols();
  }
  else {
    n = a.size().cols();
    k = a.size().rows();
  }

  DLAF_ASSERT((!std::is_same<T, ComplexType<T>>::value || op != blas::Op::Trans),
              "op = Trans is not allowed for Complex values!");
  DLAF_ASSERT(c.size().rows() == c.size().cols(), "`c` is not square!", c);
  DLAF_ASSERT(c.size().rows() == n, "`c` has an invalid size!", c, n);

  internal::CublasHerk<T>::call(handle, util::blasToCublas(uplo), util::blasToCublas(op), n, k,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), c.ld());
}

}
}

#endif
