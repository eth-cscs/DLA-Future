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
#include <hpx/functional.hpp>
#include <hpx/include/resource_partitioner.hpp>

#ifdef DLAF_WITH_CUDA
#include <hpx/modules/async_cuda.hpp>
#endif

#include <dlaf/types.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#endif

namespace dlaf {
namespace internal {
bool& initialized();

template <Backend D>
struct Init {
  // Initialization and finalization does nothing by default. Behaviour can be
  // overridden for backends.
  static void initialize(int, char**) {}
  static void finalize() {}
};

#ifdef DLAF_WITH_CUDA
void initializeNpCudaStreamPool();
void finalizeNpCudaStreamPool();
cuda::StreamPool getNpCudaStreamPool();

void initializeHpCudaStreamPool();
void finalizeHpCudaStreamPool();
cuda::StreamPool getHpCudaStreamPool();

void initializeCublasHandlePool();
void finalizeCublasHandlePool();
cublas::HandlePool getCublasHandlePool();

template <>
struct Init<Backend::GPU> {
  static void initialize(int, char**) {
    initializeNpCudaStreamPool();
    initializeHpCudaStreamPool();
    initializeCublasHandlePool();
    hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool(0));
  }

  static void finalize() {
    finalizeNpCudaStreamPool();
    finalizeHpCudaStreamPool();
    finalizeCublasHandlePool();
    hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool(0));
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

template <Device S, Device D>
decltype(auto) getCopyExecutor() {
  return GetCopyExecutor<S, D>::call();
}
}

inline void initialize(int argc, char** argv) {
  DLAF_ASSERT(!internal::initialized(), "");
  internal::Init<Backend::MC>::initialize(argc, argv);
#ifdef DLAF_WITH_CUDA
  internal::Init<Backend::GPU>::initialize(argc, argv);
#endif
  internal::initialized() = true;
}

inline void finalize() {
  DLAF_ASSERT(internal::initialized(), "");
  internal::Init<Backend::MC>::finalize();
#ifdef DLAF_WITH_CUDA
  internal::Init<Backend::GPU>::finalize();
#endif
  internal::initialized() = false;
}
}
