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

#include <cstdlib>
#include <memory>
#ifdef DLAF_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "dlaf/types.h"
#ifdef DLAF_WITH_CUDA
#include "dlaf/cuda/error.h"
#endif

namespace dlaf {
namespace memory {

/// The class @c MemoryChunk represents a layer of abstraction over the underlying host memory.

template <class T, Device device>
class MemoryChunk {
public:
  using ElementType = T;

  /// Creates a MemoryChunk object with size 0.
  MemoryChunk() : MemoryChunk(0) {}

  /// Creates a MemoryChunk object allocating the required memory.
  ///
  /// @param size The size of the memory to be allocated.
  ///
  /// Memory of @a size elements of type @c T are is allocated on the given device.
  MemoryChunk(SizeType size) : size_(size), ptr_(nullptr), allocated_(true) {
    DLAF_ASSERT(size >= 0, "");

    if (size == 0)
      return;

    std::size_t mem_size = static_cast<std::size_t>(size_) * sizeof(T);
#ifdef DLAF_WITH_CUDA
    if (device == Device::CPU) {
      DLAF_CUDA_CALL(cudaMallocHost(&ptr_, mem_size));
    }
    else {
      DLAF_CUDA_CALL(cudaMalloc(&ptr_, mem_size));
    }
#else
    if (device == Device::CPU) {
      ptr_ = static_cast<T*>(std::malloc(mem_size));
    }
    else {
      std::cout << "[ERROR] CUDA code was requested but the `DLAF_WITH_CUDA` flag was not passed!";
      std::terminate();
    }
#endif
  }

  /// Creates a MemoryChunk object from an existing memory allocation.
  ///
  /// @param ptr  The pointer to the already allocated memory,
  /// @param size The size (in number of elements of type @c T) of the existing allocation,
  /// @pre @p ptr+i can be deferenced for 0 < @c i < @p size.
  MemoryChunk(T* ptr, SizeType size) : size_(size), ptr_(size > 0 ? ptr : nullptr), allocated_(false) {
    DLAF_ASSERT_HEAVY(size == 0 ? ptr_ == nullptr : ptr_ != nullptr, "");
  }

  MemoryChunk(const MemoryChunk&) = delete;

  /// Move constructor.
  MemoryChunk(MemoryChunk&& rhs) : size_(rhs.size_), ptr_(rhs.ptr_), allocated_(rhs.allocated_) {
    rhs.ptr_ = nullptr;
    rhs.size_ = 0;
    rhs.allocated_ = false;
  }

  MemoryChunk& operator=(const MemoryChunk&) = delete;

  /// Move assignement.
  MemoryChunk& operator=(MemoryChunk&& rhs) {
    deallocate();

    size_ = rhs.size_;
    ptr_ = rhs.ptr_;
    allocated_ = rhs.allocated_;

    rhs.size_ = 0;
    rhs.ptr_ = nullptr;
    rhs.allocated_ = false;

    return *this;
  }

  /// Destructor. Memory is deallocated only if it was allocated at construction.
  ~MemoryChunk() {
    deallocate();
  }

  /// Returns a pointer to the underlying memory at a given index.
  ///
  /// @param index index of the position,
  /// @pre @p index < @p size.
  T* operator()(SizeType index) {
    DLAF_ASSERT_HEAVY(index < size_, "", index, size_);
    return ptr_ + index;
  }
  const T* operator()(SizeType index) const {
    DLAF_ASSERT_HEAVY(index < size_, "", index, size_);
    return ptr_ + index;
  }

  /// Returns a pointer to the underlying memory.
  /// If @p size == 0 a @c nullptr is returned.
  T* operator()() {
    return ptr_;
  }
  const T* operator()() const {
    return ptr_;
  }

  /// Returns the number of elements of type @c T allocated.
  SizeType size() const {
    return size_;
  }

private:
  void deallocate() {
    if (allocated_) {
#ifdef DLAF_WITH_CUDA
      if (device == Device::CPU) {
        DLAF_CUDA_CALL(cudaFreeHost(ptr_));
      }
      else {
        DLAF_CUDA_CALL(cudaFree(ptr_));
      }
#else
      if (device == Device::CPU) {
        std::free(ptr_);
      }
#endif
    }
  }

  SizeType size_;
  T* ptr_;
  bool allocated_;
};

}  // namespace memory
}  // namespace dlaf
