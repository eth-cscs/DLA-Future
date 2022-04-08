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

/// @file

#ifdef DLAF_WITH_CUDA

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <pika/runtime.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cusolver/error.h"

namespace dlaf {
namespace cusolver {
namespace internal {
class HandlePoolImpl {
  int device_;
  std::size_t num_worker_threads_ = pika::get_num_worker_threads();
  std::vector<cusolverDnHandle_t> handles_;

public:
  HandlePoolImpl(int device) : device_(device), handles_(num_worker_threads_) {
    DLAF_CUDA_CHECK_ERROR(cudaSetDevice(device_));

    for (auto& h : handles_) {
      DLAF_CUSOLVER_CHECK_ERROR(cusolverDnCreate(&h));
    }
  }

  HandlePoolImpl& operator=(HandlePoolImpl&&) = default;
  HandlePoolImpl(HandlePoolImpl&&) = default;
  HandlePoolImpl(const HandlePoolImpl&) = delete;
  HandlePoolImpl& operator=(const HandlePoolImpl&) = delete;

  ~HandlePoolImpl() {
    for (auto& h : handles_) {
      DLAF_CUSOLVER_CHECK_ERROR(cusolverDnDestroy(h));
    }
  }

  cusolverDnHandle_t getNextHandle(cudaStream_t stream) {
    cusolverDnHandle_t handle = handles_[pika::get_worker_thread_num()];
    DLAF_CUDA_CHECK_ERROR(cudaSetDevice(device_));
    DLAF_CUSOLVER_CHECK_ERROR(cusolverDnSetStream(handle, stream));
    return handle;
  }

  int getDevice() {
    return device_;
  }
};
}

/// A pool of cuSOLVER handles with reference semantics (copying points to the
/// same underlying cuSOLVER handles, last reference destroys the handles).
/// Allows access to cuSOLVER handles associated with a particular stream. The
/// user must ensure that the handle pool and the stream use the same device.
/// Each pika worker thread is assigned thread local cuSOLVER handle.
class HandlePool {
  std::shared_ptr<internal::HandlePoolImpl> handles_ptr_;

public:
  HandlePool(int device = 0) : handles_ptr_(std::make_shared<internal::HandlePoolImpl>(device)) {}

  cusolverDnHandle_t getNextHandle(cudaStream_t stream) {
    DLAF_ASSERT(bool(handles_ptr_), "");
    return handles_ptr_->getNextHandle(stream);
  }

  int getDevice() {
    DLAF_ASSERT(bool(handles_ptr_), "");
    return handles_ptr_->getDevice();
  }

  bool operator==(HandlePool const& rhs) const noexcept {
    return handles_ptr_ == rhs.handles_ptr_;
  }

  bool operator!=(HandlePool const& rhs) const noexcept {
    return !(*this == rhs);
  }
};
}
}

#endif
