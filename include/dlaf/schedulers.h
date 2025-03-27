//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <pika/execution.hpp>
#include <pika/runtime.hpp>
#include <pika/thread.hpp>

#ifdef DLAF_WITH_GPU
#include <pika/cuda.hpp>
#endif

#include <dlaf/common/assert.h>
#include <dlaf/init.h>
#include <dlaf/types.h>

namespace dlaf::internal {
template <Backend backend>
auto getBackendScheduler(
    const pika::execution::thread_priority priority = pika::execution::thread_priority::default_,
    const pika::execution::thread_stacksize stacksize = pika::execution::thread_stacksize::default_) {
  namespace ex = pika::execution::experimental;
  using pika::execution::thread_priority;
  using pika::execution::thread_stacksize;

  if constexpr (backend == Backend::MC) {
    return ex::with_stacksize(
        ex::with_priority(ex::thread_pool_scheduler{&pika::resource::get_thread_pool("default")},
                          priority),
        stacksize);
  }
#ifdef DLAF_WITH_GPU
  else if constexpr (backend == Backend::GPU) {
    silenceUnusedWarningFor(stacksize);
    namespace cu = pika::cuda::experimental;

    return ex::with_priority(cu::cuda_scheduler{internal::getGpuPool()}, priority);
  }
#endif
}
}  // namespace dlaf::internal
