//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020, ETH Zurich
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

#include <dlaf/communication/executor.h>
#include <dlaf/init.h>
#include <dlaf/types.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#endif

namespace dlaf {
namespace internal {
template <Backend B>
struct GetMPIExecutor {
  static auto call() {
    return hpx::execution::parallel_executor(&hpx::resource::get_thread_pool(
                                                 comm::internal::mpi_pool_exists() ? "mpi" : "default"),
                                             hpx::threads::thread_priority_high);
  }
};

template <Backend B>
struct GetNpExecutor {
  static auto call() {
    return hpx::execution::parallel_executor{hpx::threads::thread_priority::normal};
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct GetNpExecutor<Backend::GPU> {
  static auto call() {
    return dlaf::cublas::Executor{getNpCudaStreamPool(), getCublasHandlePool()};
  }
};
#endif

template <Backend B>
struct GetHpExecutor {
  static auto call() {
    return hpx::execution::parallel_executor{hpx::threads::thread_priority::high};
  }
};

#ifdef DLAF_WITH_CUDA
template <>
struct GetHpExecutor<Backend::GPU> {
  static auto call() {
    return dlaf::cublas::Executor{getHpCudaStreamPool(), getCublasHandlePool()};
  }
};
#endif

template <Device S, Device D>
struct GetCopyExecutor {
  static auto call() {
    return hpx::execution::parallel_executor{};
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

/// Returns an MPI executor approprate for use with the given backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
decltype(auto) getMPIExecutor() {
  return internal::GetMPIExecutor<B>::call();
}

/// Returns a normal priority executor approprate for use with the given
/// backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
decltype(auto) getNpExecutor() {
  return internal::GetNpExecutor<B>::call();
}

/// Returns a high priority executor approprate for use with the given
/// backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
decltype(auto) getHpExecutor() {
  return internal::GetHpExecutor<B>::call();
}

/// Returns an executor appropriate for copying from @tparam S to @tparam D.
///
/// @tparam S source device.
/// @tparam D destination device.
template <Device S, Device D>
decltype(auto) getCopyExecutor() {
  return internal::GetCopyExecutor<S, D>::call();
}
}
