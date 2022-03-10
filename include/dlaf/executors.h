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

#include <pika/execution.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/thread.hpp>

#ifdef DLAF_WITH_CUDA
#include <pika/modules/async_cuda.hpp>
#endif

#include <dlaf/communication/executor.h>
#include <dlaf/init.h>
#include <dlaf/types.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#include <dlaf/cusolver/executor.h>
#endif

namespace dlaf {
/// Returns an MPI executor appropriate for use with the given backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
auto getMPIExecutor() {
  return dlaf::comm::Executor{};
}

/// Returns a high priority executor appropriate for use with the given
/// backend.
///
/// @tparam B backend with which the executor should be used.
template <Backend B>
auto getHpExecutor() {
  return pika::execution::parallel_executor{&pika::resource::get_thread_pool("default"),
                                            pika::threads::thread_priority::high};
}

#ifdef DLAF_WITH_CUDA
template <>
inline auto getHpExecutor<Backend::GPU>() {
  return dlaf::cusolver::Executor{internal::getHpCudaStreamPool(), internal::getCublasHandlePool(),
                                  internal::getCusolverHandlePool()};
}
#endif

/// Returns an executor appropriate for copying from @tparam S to @tparam D.
///
/// @tparam S source device.
/// @tparam D destination device.
template <Device S, Device D, typename Dummy = void>
decltype(auto) getCopyExecutor() {
  if constexpr (S == Device::CPU && D == Device::CPU) {
    return pika::execution::parallel_executor{&pika::resource::get_thread_pool("default"),
                                              pika::threads::thread_priority::normal};
  }
  else {
    DLAF_STATIC_FAIL(Dummy,
                     "Do not use getCopyExecutor for copying anymore. Prefer sender adaptors with CopyBackend.");
  }
}
}
