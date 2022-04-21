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

#include <type_traits>
#include <utility>

#include <dlaf/types.h>

namespace dlaf {
namespace internal {
/// A policy class for use as a tag for dispatching algorithms to a particular
/// backend.
template <Backend B>
class Policy {
private:
  const pika::threads::thread_priority priority_ = pika::threads::thread_priority::normal;
  const pika::threads::thread_schedule_hint hint_{};

public:
  Policy() = default;
  explicit Policy(pika::threads::thread_priority priority, pika::threads::thread_schedule_hint hint = {})
      : priority_(priority), hint_(hint) {}
  Policy(Policy&&) = default;
  Policy(Policy const&) = default;
  Policy& operator=(Policy&&) = default;
  Policy& operator=(Policy const&) = default;

  pika::threads::thread_priority priority() const noexcept {
    return priority_;
  }

  pika::threads::thread_schedule_hint hint() const noexcept {
    return hint_;
  }
};
}
}
