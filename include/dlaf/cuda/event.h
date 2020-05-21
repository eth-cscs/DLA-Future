//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <cuda_runtime.h>

#include <hpx/modules/execution_base.hpp>

#include "dlaf/cuda/error.h"

namespace dlaf {
namespace cuda {

// A Wrapper class for an event used for synchronization
class Event {
  bool is_moved;
  cudaEvent_t event_;

public:
  Event() noexcept {
    is_moved = false;
    // Create an event to query a CUBLAS kernel for completion. Timing is disabled for performance. [1]
    //
    // [1]: CUDA Runtime API, section 5.5 Event Management
    DLAF_CUDA_CALL(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  Event& operator=(Event&& e) noexcept {
    event_ = e.event_;
    e.is_moved = true;
    return *this;
  }

  Event(Event&& e) noexcept {
    *this = std::move(e);
  }

  ~Event() {
    if (!is_moved)
      DLAF_CUDA_CALL(cudaEventDestroy(event_));
  }

  // Insert the event into the stream
  void record(cudaStream_t stream) const noexcept {
    DLAF_CUDA_CALL(cudaEventRecord(event_, stream));
  }

  // Queries if the event has completed and yields the HPX task if is not.
  void query() const noexcept {
    hpx::util::yield_while([event = event_] {
      // Note that the call succeeds even if the event is associated to a device that is different from
      // the current device on the host thread. That could be the case if a previous task executing on
      // the same host thread set a different device. [1]
      //
      // [1]: CUDA Programming Guide, section 3.2.3.6, Stream and Event Behavior
      cudaError_t err = cudaEventQuery(event);

      // Note: Despite the name, `cudaErrorNotReady` is not considered an error. [1]
      //
      // [1]: CUDA Runtime API, section 5.33 Data types used by CUDA Runtime
      if (err == cudaErrorNotReady) {
        return true;
      }
      else if (err == cudaSuccess) {
        return false;
      }
      DLAF_CUDA_CALL(err);  // An error occured, report and terminate.
      return false;         // Unreachable!
    });
  }
};

}
}
