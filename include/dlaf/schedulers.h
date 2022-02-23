//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/execution.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/runtime.hpp>
#include <pika/thread.hpp>

#include <dlaf/init.h>
#include <dlaf/types.h>

namespace dlaf::internal {
template <Backend backend>
auto getBackendScheduler() {
  if constexpr (backend == Backend::MC) {
    return pika::execution::experimental::thread_pool_scheduler{
        &pika::resource::get_thread_pool("default")};
  }
}

template <Backend backend>
auto getBackendScheduler(const pika::threads::thread_priority priority) {
  return pika::execution::experimental::with_priority(getBackendScheduler<backend>(), priority);
}
}
