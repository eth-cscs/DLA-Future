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

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include "dlaf/types.h"

#ifdef DLAF_WITH_GPU
#include "dlaf/gpu/blas/api.h"
#include "dlaf/gpu/blas/error.h"
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

#ifdef DLAF_WITH_GPU
template <>
struct Invoke<Device::GPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    f(std::forward<Args>(args)..., nullptr);
    whip::device_synchronize();
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

#ifdef DLAF_WITH_GPU
template <>
struct InvokeBlas<Device::GPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    cublasHandle_t handle;
    DLAF_GPUBLAS_CHECK_ERROR(cublasCreate(&handle));
    f(handle, std::forward<Args>(args)...);
    whip::device_synchronize();
    DLAF_GPUBLAS_CHECK_ERROR(cublasDestroy(handle));
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
