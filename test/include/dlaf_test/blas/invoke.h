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

#include "dlaf/types.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "dlaf/cublas/error.h"
#include "dlaf/cuda/error.h"
#endif

namespace dlaf::test {
namespace internal {

template <Device D>
struct Invoke;

template <>
struct Invoke<Device::CPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    f(std::forward<Args>(args)...);
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct Invoke<Device::GPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    f(std::forward<Args>(args)..., nullptr);
    DLAF_CUDA_CALL(cudaDeviceSynchronize());
  }
};
#endif

template <Device D>
struct InvokeBlas;

template <>
struct InvokeBlas<Device::CPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    f(std::forward<Args>(args)...);
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct InvokeBlas<Device::GPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    cublasHandle_t handle;
    DLAF_CUBLAS_CALL(cublasCreate(&handle));
    f(handle, std::forward<Args>(args)...);
    DLAF_CUDA_CALL(cudaDeviceSynchronize());
    DLAF_CUBLAS_CALL(cublasDestroy(handle));
  }
};
#endif
}

/// Invokes a call in a generic way:
/// For CPU it calls f(args...),
/// For GPU it calls f(args..., stream) with the null stream and it synchronizes.
template <Device D, class F, class... Args>
void invoke(F&& f, Args&&... args) {
  internal::Invoke<D>::call(std::forward<F>(f), std::forward<Args>(args)...);
}

/// Invokes a Blas call in a generic way:
/// For CPU it calls f(args...),
/// For GPU it creates a cublas handle, it calls f(handle, args...) and it synchronizes.
template <Device D, class F, class... Args>
void invokeBlas(F&& f, Args&&... args) {
  internal::InvokeBlas<D>::call(std::forward<F>(f), std::forward<Args>(args)...);
}

}
