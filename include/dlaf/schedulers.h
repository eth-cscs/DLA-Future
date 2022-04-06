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
#include <pika/runtime.hpp>
#include <pika/thread.hpp>

#ifdef DLAF_WITH_CUDA
#include <pika/cuda.hpp>
#endif

#include <dlaf/init.h>
#include <dlaf/types.h>

namespace dlaf::internal {
template <Backend backend>
auto getBackendScheduler() {
  if constexpr (backend == Backend::MC) {
    return pika::execution::experimental::thread_pool_scheduler{
        &pika::resource::get_thread_pool("default")};
  }
#ifdef DLAF_WITH_CUDA
  else if constexpr (backend == Backend::GPU) {
    return pika::cuda::experimental::cuda_scheduler{internal::getCudaPool()};
  }
#endif
}

template <Backend backend>
auto getBackendScheduler(const pika::threads::thread_priority priority) {
  return pika::execution::experimental::with_priority(getBackendScheduler<backend>(), priority);
}
}
