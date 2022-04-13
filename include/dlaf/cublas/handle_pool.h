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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <pika/runtime.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cuda/error.h"
#include "dlaf/cuda/executor.h"

namespace dlaf {
namespace cublas {
namespace internal {
class HandlePoolImpl {
  int device_;
  std::size_t num_worker_threads_ = pika::get_num_worker_threads();
  std::vector<cublasHandle_t> handles_;
  cublasPointerMode_t ptr_mode_;

public:
  HandlePoolImpl(int device, cublasPointerMode_t ptr_mode)
      : device_(device), handles_(num_worker_threads_), ptr_mode_(ptr_mode) {
    DLAF_CUDA_CHECK_ERROR(cudaSetDevice(device_));

    for (auto& h : handles_) {
      DLAF_CUBLAS_CHECK_ERROR(cublasCreate(&h));
    }
  }

  HandlePoolImpl& operator=(HandlePoolImpl&&) = default;
  HandlePoolImpl(HandlePoolImpl&&) = default;
  HandlePoolImpl(const HandlePoolImpl&) = delete;
  HandlePoolImpl& operator=(const HandlePoolImpl&) = delete;

  ~HandlePoolImpl() {
    for (auto& h : handles_) {
      DLAF_CUBLAS_CHECK_ERROR(cublasDestroy(h));
    }
  }

  cublasHandle_t getNextHandle(cudaStream_t stream) {
    cublasHandle_t handle = handles_[pika::get_worker_thread_num()];
    DLAF_CUDA_CHECK_ERROR(cudaSetDevice(device_));
    DLAF_CUBLAS_CHECK_ERROR(cublasSetStream(handle, stream));
    DLAF_CUBLAS_CHECK_ERROR(cublasSetPointerMode(handle, ptr_mode_));
    return handle;
  }

  int getDevice() {
    return device_;
  }
};
}

/// A pool of cuBLAS handles with reference semantics (copying points to the
/// same underlying cuBLAS handles, last reference destroys the references).
/// Allows access to cuBLAS handles associated with a particular stream. The
/// user must ensure that the handle pool and the stream use the same device.
/// Each pika worker thread is assigned thread local cuBLAS handle.
class HandlePool {
  std::shared_ptr<internal::HandlePoolImpl> handles_ptr_;

public:
  HandlePool(int device = 0, cublasPointerMode_t ptr_mode = CUBLAS_POINTER_MODE_HOST)
      : handles_ptr_(std::make_shared<internal::HandlePoolImpl>(device, ptr_mode)) {}

  cublasHandle_t getNextHandle(cudaStream_t stream) {
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
