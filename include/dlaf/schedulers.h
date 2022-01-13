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

#include <hpx/include/resource_partitioner.hpp>
#include <hpx/local/execution.hpp>
#include <hpx/local/thread.hpp>

#ifdef DLAF_WITH_CUDA
#include <hpx/modules/async_cuda.hpp>
#endif

#include <dlaf/init.h>
#include <dlaf/types.h>

namespace dlaf {
template <Backend backend>
auto getBackendScheduler() {
  if constexpr (backend == Backend::MC) {
    return hpx::execution::experimental::thread_pool_scheduler{
        &hpx::resource::get_thread_pool("default")};
  }
#ifdef DLAF_WITH_CUDA
  else if constexpr (backend == Backend::GPU) {
    return hpx::cuda::experimental::cuda_scheduler{internal::getCudaPool()};
  }
#endif
}

template <Backend backend>
auto getBackendScheduler(const hpx::threads::thread_priority priority) {
  return hpx::execution::experimental::with_priority(getBackendScheduler<backend>(), priority);
}
}
