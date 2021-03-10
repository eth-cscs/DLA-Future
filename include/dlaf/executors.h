//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/execution.hpp>
#include <hpx/thread.hpp>

#ifdef DLAF_WITH_CUDA
#include <hpx/modules/async_cuda.hpp>
#endif

#include <dlaf/init.h>
#include <dlaf/types.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#endif

namespace dlaf {
namespace internal {
template <Device S, Device D>
struct GetCopyExecutor {
  static auto call() {
    return hpx::execution::parallel_executor{&hpx::resource::get_thread_pool("default"),
                                             hpx::threads::thread_priority::normal};
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct GetCopyExecutor<Device::GPU, Device::CPU> {
  static auto call() {
    return dlaf::cuda::Executor{getNpCudaStreamPool()};
  }
};

template <>
struct GetCopyExecutor<Device::CPU, Device::GPU> {
  static auto call() {
    return dlaf::cuda::Executor{getNpCudaStreamPool()};
  }
};

template <>
struct GetCopyExecutor<Device::GPU, Device::GPU> {
  static auto call() {
    return dlaf::cuda::Executor{getNpCudaStreamPool()};
  }
};
#endif
}

/// Returns an MPI executor appropriate for use with the given backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
auto getMPIExecutor() {
  return hpx::execution::parallel_executor{&hpx::resource::get_thread_pool(
                                               hpx::resource::pool_exists("mpi") ? "mpi" : "default"),
                                           hpx::threads::thread_priority::high};
}

/// Returns a normal priority executor appropriate for use with the given
/// backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
auto getNpExecutor() {
  return hpx::execution::parallel_executor{&hpx::resource::get_thread_pool("default"),
                                           hpx::threads::thread_priority::normal};
}

#ifdef DLAF_WITH_CUDA
template <>
inline auto getNpExecutor<Backend::GPU>() {
  return dlaf::cublas::Executor{internal::getNpCudaStreamPool(), internal::getCublasHandlePool()};
}
#endif

/// Returns a high priority executor appropriate for use with the given
/// backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
auto getHpExecutor() {
  return hpx::execution::parallel_executor{&hpx::resource::get_thread_pool("default"),
                                           hpx::threads::thread_priority::high};
}

#ifdef DLAF_WITH_CUDA
template <>
inline auto getHpExecutor<Backend::GPU>() {
  return dlaf::cublas::Executor{internal::getHpCudaStreamPool(), internal::getCublasHandlePool()};
}
#endif

/// Returns an executor appropriate for copying from @tparam S to @tparam D.
///
/// @tparam S source device.
/// @tparam D destination device.
template <Device S, Device D>
decltype(auto) getCopyExecutor() {
  return internal::GetCopyExecutor<S, D>::call();
}
}
