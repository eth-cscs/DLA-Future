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

#include "dlaf/types.h"

#ifdef DLAF_WITH_CUDA
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "dlaf/cuda/error.h"
#include "dlaf/cusolver/error.h"
#endif

namespace dlaf {
namespace test {

namespace internal {
template <Device D>
struct InvokeLapack;

template <>
struct InvokeLapack<Device::CPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    f(std::forward<Args>(args)...);
  }

  template <class F, class... Args>
  static auto callInfo(F&& f, Args&&... args) {
    return f(std::forward<Args>(args)...);
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct InvokeLapack<Device::GPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    cusolverDnHandle_t handle;
    DLAF_CUSOLVER_CALL(cusolverDnCreate(&handle));
    f(handle, std::forward<Args>(args)...);
    DLAF_CUDA_CALL(cudaDeviceSynchronize());
    DLAF_CUSOLVER_CALL(cusolverDnDestroy(handle));
  }

  template <class F, class... Args>
  static int callInfo(F&& f, Args&&... args) {
    cusolverDnHandle_t handle;
    DLAF_CUSOLVER_CALL(cusolverDnCreate(&handle));
    auto result = f(handle, std::forward<Args>(args)...);
    int info_host;
    // The copy will happen on the same (default) stream as the potrf, and since
    // this is a blocking call, we can access info_host without further
    // synchronization.
    DLAF_CUDA_CALL(cudaMemcpy(&info_host, result.info(), sizeof(int), cudaMemcpyDeviceToHost));
    DLAF_CUSOLVER_CALL(cusolverDnDestroy(handle));

    return info_host;
  }
};
#endif
}

/// Invokes a Blas call in a generic way:
/// For CPU it calls f(args...),
/// For GPU it create a cublas handle, it calls f(handle, args...) and it sychronize.
template <Device D, class F, class... Args>
void invokeLapack(F&& f, Args&&... args) {
  internal::InvokeLapack<D>::call(std::forward<F>(f), std::forward<Args>(args)...);
}

template <Device D, class F, class... Args>
auto invokeLapackInfo(F&& f, Args&&... args) {
  return internal::InvokeLapack<D>::callInfo(std::forward<F>(f), std::forward<Args>(args)...);
}

}
}
