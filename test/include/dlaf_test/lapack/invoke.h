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

#ifdef DLAF_WITH_GPU
#include "dlaf/gpu/api.h"
#include "dlaf/gpu/error.h"
#include "dlaf/gpu/lapack/api.h"
#include "dlaf/gpu/lapack/error.h"
#endif

namespace dlaf::test {
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

#ifdef DLAF_WITH_GPU
template <>
struct InvokeLapack<Device::GPU> {
  template <class F, class... Args>
  static void call(F&& f, Args&&... args) {
    cusolverDnHandle_t handle;
    DLAF_GPULAPACK_CHECK_ERROR(cusolverDnCreate(&handle));
    f(handle, std::forward<Args>(args)...);
    DLAF_GPU_CHECK_ERROR(cudaDeviceSynchronize());
    DLAF_GPULAPACK_CHECK_ERROR(cusolverDnDestroy(handle));
  }

  template <class F, class... Args>
  static int callInfo(F&& f, Args&&... args) {
    cusolverDnHandle_t handle;
    DLAF_GPULAPACK_CHECK_ERROR(cusolverDnCreate(&handle));
    auto result = f(handle, std::forward<Args>(args)...);
    int info_host;
    // The copy will happen on the same (default) stream as the call to f, and
    // since this is a blocking call, we can access info_host without further
    // synchronization.
    DLAF_GPU_CHECK_ERROR(cudaMemcpy(&info_host, result.info(), sizeof(int), cudaMemcpyDeviceToHost));
    DLAF_GPULAPACK_CHECK_ERROR(cusolverDnDestroy(handle));

    return info_host;
  }
};
#endif
}

/// Invokes a Lapack call in a generic way:
/// For CPU it calls f(args...),
/// For GPU it creates a cuSolver handle, it calls f(handle, args...) and it synchronizes.
template <Device D, class F, class... Args>
void invokeLapack(F&& f, Args&&... args) {
  internal::InvokeLapack<D>::call(std::forward<F>(f), std::forward<Args>(args)...);
}

template <Device D, class F, class... Args>
auto invokeLapackInfo(F&& f, Args&&... args) {
  return internal::InvokeLapack<D>::callInfo(std::forward<F>(f), std::forward<Args>(args)...);
}

}
