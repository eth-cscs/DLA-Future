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

#ifdef DLAF_WITH_CUDA

#include <cublas_v2.h>
#include <blas.hh>

#include "dlaf/common/callable_object.h"
#include "dlaf/cublas/error.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_blas.h"
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
void trsm(cublasHandle_t handle, const blas::Side side, const blas::Uplo uplo, const blas::Op op,
          const blas::Diag diag, const T alpha, const matrix::Tile<const T, Device::GPU>& a,
          const matrix::Tile<T, Device::GPU>& b) {
  auto s = tile::internal::getTrsmSizes(side, a, b);
  internal::CublasTrsm<T>::call(handle, util::blasToCublas(side), util::blasToCublas(uplo),
                                util::blasToCublas(op), util::blasToCublas(diag), s.m, s.n,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(b.ptr()), b.ld());
}

// TODO: This should be in a common location, not in the gpu header.
DLAF_MAKE_CALLABLE_OBJECT(trsm);

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

// TODO: This should be in a common location, not in the gpu header.
DLAF_MAKE_CALLABLE_OBJECT(gemm);

template <class T>
void herk(cublasHandle_t handle, const blas::Uplo uplo, const blas::Op op, const BaseType<T> alpha,
          const matrix::Tile<const T, Device::GPU>& a, const BaseType<T> beta,
          const matrix::Tile<T, Device::GPU>& c) {
  auto s = tile::internal::getHerkSizes(op, a, c);
  internal::CublasHerk<T>::call(handle, util::blasToCublas(uplo), util::blasToCublas(op), s.n, s.k,
                                util::blasToCublasCast(&alpha), util::blasToCublasCast(a.ptr()), a.ld(),
                                util::blasToCublasCast(&beta), util::blasToCublasCast(c.ptr()), c.ld());
}

// TODO: This should be in a common location, not in the gpu header.
DLAF_MAKE_CALLABLE_OBJECT(herk);
}
}

#endif
