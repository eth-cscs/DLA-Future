//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <pika/async_rw_mutex.hpp>
#include <pika/execution.hpp>
#include <dlaf/common/assert.h>

namespace dlaf::common {

/// Pipeline takes ownership of a given object and manages the access to this resource by serializing
/// calls. Anyone that requires access to the underlying resource will get a sender, which is the
/// way to register to the queue. All requests are serialized and served in the same order they arrive.
/// On destruction it does not wait for the queued requests for the resource and exits immediately.
///
/// The mechanism for auto-releasing the resource and passing it to the next user works thanks to the
/// internal wrapper object. The wrapper contains the real resource, and it will do what is
/// needed to unlock the next user as soon as the wrapper is destroyed.
template <class T>
class Pipeline {
  using AsyncRwMutex = pika::execution::experimental::async_rw_mutex<T>;

public:
  using Wrapper = typename AsyncRwMutex::readwrite_access_type;
  using Sender = pika::execution::experimental::unique_any_sender<Wrapper>;

  /// Create a Pipeline by moving in the resource (it takes the ownership).
  explicit Pipeline(T object) : pipeline(std::move(object)) {}
  Pipeline(Pipeline&&) = default;
  Pipeline& operator=(Pipeline&&) = default;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  /// Enqueue for the resource.
  ///
  /// @return a sender that will become ready as soon as the previous user releases the resource.
  Sender operator()() {
    DLAF_ASSERT(pipeline.has_value(), "");
    return pipeline->readwrite();
  }

  /// Check if the pipeline is valid.
  ///
  /// @return true if the pipeline hasn't been reset, otherwise false.
  bool valid() const noexcept {
    return pipeline.has_value();
  }

  /// Reset the pipeline.
  ///
  /// @post !valid()
  void reset() noexcept {
    pipeline.reset();
  }

private:
  std::optional<AsyncRwMutex> pipeline = std::nullopt;
};
}
