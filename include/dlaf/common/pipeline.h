//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <optional>
#include <utility>

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
  using ReadOnlyWrapper = typename AsyncRwMutex::read_access_type;
  using ReadWriteWrapper = typename AsyncRwMutex::readwrite_access_type;
  using ReadOnlySender = pika::execution::experimental::any_sender<ReadOnlyWrapper>;
  using ReadWriteSender = pika::execution::experimental::unique_any_sender<ReadWriteWrapper>;

  /// Create an invalid Pipeline.
  Pipeline() = default;

  /// Create a Pipeline by moving in the resource (it takes the ownership).
  explicit Pipeline(T object) : pipeline(std::move(object)) {}

  Pipeline(Pipeline&& other) noexcept
      : pipeline(std::exchange(other.pipeline, std::nullopt)),
        nested_sender(std::exchange(other.nested_sender, std::nullopt)) {}

  Pipeline& operator=(Pipeline&& other) noexcept {
    if (this != &other) {
      pipeline = std::exchange(other.pipeline, std::nullopt);
      nested_sender = std::exchange(other.nested_sender, std::nullopt);
    }

    return *this;
  }

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  ~Pipeline() {
    release_parent_pipeline();
  }

  /// Enqueue for exclusive read-write access to the resource.
  ///
  /// @return a sender that will become ready as soon as the previous user releases the resource.
  /// @pre valid()
  ReadWriteSender readwrite() {
    DLAF_ASSERT(valid(), "");
    return pipeline->readwrite();
  }

  /// Enqueue for shared read-only access to the resource.
  ///
  /// @return a sender that will become ready as soon as the previous user releases the resource.
  /// @pre valid()
  ReadOnlySender read() {
    DLAF_ASSERT(valid(), "");
    return pipeline->read();
  }

  /// Create a sub pipeline to the value contained in the current Pipeline
  ///
  /// All accesses to the sub pipeline are sequenced after previous accesses and before later accesses to
  /// the original pipeline, independently of when values are accessed in the sub pipeline.
  Pipeline sub_pipeline() {
    namespace ex = pika::execution::experimental;

    // Move value from pipeline into sub pipeline, then store a sender of the wrapper of the pipeline in
    // a sender which we will release when the sub pipeline is released. This ensures that all accesses
    // to the sub pipeline happen after previous accesses and before later accesses to the pipeline.
    Pipeline sub_pipeline(T{});
    sub_pipeline.nested_sender =
        ex::when_all(sub_pipeline.pipeline->readwrite(), this->pipeline->readwrite()) |
        ex::then([](auto sub_wrapper, auto wrapper) {
          sub_wrapper.get() = std::move(wrapper.get());

          return wrapper;
        }) |
        ex::ensure_started();

    return sub_pipeline;
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
    release_parent_pipeline();
    pipeline.reset();
  }

private:
  void release_parent_pipeline() {
    namespace ex = pika::execution::experimental;

    if (nested_sender) {
      DLAF_ASSERT(valid(), "");

      auto s =
          ex::when_all(pipeline->readwrite(), std::move(nested_sender.value())) |
          ex::then([](auto sub_wrapper, auto wrapper) { wrapper.get() = std::move(sub_wrapper.get()); });
      ex::start_detached(std::move(s));
      nested_sender.reset();
    }
  }

  std::optional<AsyncRwMutex> pipeline = std::nullopt;
  std::optional<pika::execution::experimental::unique_any_sender<ReadWriteWrapper>> nested_sender =
      std::nullopt;
};
}
