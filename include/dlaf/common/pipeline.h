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

/// @file

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

  // TODO
  Pipeline() {}
  /// Create a Pipeline by moving in the resource (it takes the ownership).
  explicit Pipeline(T object) : pipeline(std::move(object)) {}
  Pipeline(Pipeline&&) = default;
  Pipeline& operator=(Pipeline&&) = default;
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  ~Pipeline() {
    releaseParentPipeline();
  }

  /// Enqueue for the resource.
  ///
  /// @return a sender that will become ready as soon as the previous user releases the resource.
  /// @pre valid()
  ReadWriteSender operator()() {
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

  Pipeline subPipeline() {
    namespace ex = pika::execution::experimental;

    // TODO: Requires default constructibility. Is it a must?
    Pipeline sub_pipeline(T{});
    // Move communicator from pipeline into sub pipeline
    auto s = ex::when_all(sub_pipeline.pipeline->readwrite(), this->pipeline->readwrite()) |
             ex::then([](auto sub_comm_wrapper, auto comm_wrapper) {
               sub_comm_wrapper.get() = std::move(comm_wrapper.get());

               auto sub_comm_wrapper_local = std::move(sub_comm_wrapper);
               auto comm_wrapper_local = std::move(comm_wrapper);
             });
    ex::start_detached(std::move(s));

    // Access pipeline again, we'll move the communicator from the sub pipeline back into the pipeline on
    // destruction. This ensures that accesses to the parent pipeline happen only after the sub pipeline
    // has been released.
    sub_pipeline.nested_sender = this->pipeline->readwrite();

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
    releaseParentPipeline();
    pipeline.reset();
  }

private:
  void releaseParentPipeline() {
    namespace ex = pika::execution::experimental;

    if (nested_sender) {
      DLAF_ASSERT(valid(), "");

      auto s = ex::when_all(pipeline->readwrite(), std::move(nested_sender.value())) |
               ex::then([](auto sub_comm_wrapper, auto comm_wrapper) {
                 comm_wrapper.get() = std::move(sub_comm_wrapper.get());

                 auto sub_comm_wrapper_local = std::move(sub_comm_wrapper);
                 auto comm_wrapper_local = std::move(comm_wrapper);
               });
      ex::start_detached(std::move(s));
      nested_sender.reset();
    }
  }

  std::optional<AsyncRwMutex> pipeline = std::nullopt;
  std::optional<decltype(pipeline->readwrite())> nested_sender = std::nullopt;
};
}
