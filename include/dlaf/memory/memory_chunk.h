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

#include <cassert>
#include <cstdlib>
#include <memory>
#ifdef DLAF_WITH_CUDA
#include <cuda_runtime.h>
#endif
#include "dlaf/types.h"

namespace dlaf {
namespace memory {

/// The class @c MemoryChunk represents a layer of abstraction over the underlying host memory.

template <class T, Device device>
class MemoryChunk {
public:
  using ElementType = T;

  /// @brief Creates a MemoryChunk object with size 0.
  MemoryChunk() : MemoryChunk(0) {}

  /// @brief Creates a MemoryChunk object allocating the required memory.
  ///
  /// @param size The size of the memory to be allocated.
  ///
  /// Memory of @a size elements of type @c T are is allocated on the given device.
  MemoryChunk(std::size_t size) : size_(size), ptr_(nullptr), allocated_(true) {
    if (size == 0)
      return;

#ifdef DLAF_WITH_CUDA
    if (device == Device::CPU) {
      cudaMallocHost(&ptr_, size_ * sizeof(T));
    }
    else {
      cudaMalloc(&ptr_, size_ * sizeof(T));
    }
#else
    if (device == Device::CPU) {
      ptr_ = static_cast<T*>(std::malloc(size_ * sizeof(T)));
    }
    else {
      std::terminate();
    }
#endif
  }

  /// @brief Creates a MemoryChunk object from an existing memory allocation.
  ///
  /// @param ptr  The pointer to the already allocated memory.
  /// @param size The size (in number of elements of type @c T) of the existing allocation.
  /// @pre @p ptr+i can be deferenced for 0 < @c i < @p size
  MemoryChunk(T* ptr, std::size_t size)
      : size_(size), ptr_(size > 0 ? ptr : nullptr), allocated_(false) {
    assert(size == 0 ? ptr_ == nullptr : ptr_ != nullptr);
  }

  MemoryChunk(const MemoryChunk&) = delete;

  /// @brief Move constructor
  MemoryChunk(MemoryChunk&& rhs) : size_(rhs.size_), ptr_(rhs.ptr_), allocated_(rhs.allocated_) {
    rhs.ptr_ = nullptr;
    rhs.size_ = 0;
    rhs.allocated_ = false;
  }

  MemoryChunk& operator=(const MemoryChunk&) = delete;

  /// @brief Move assignement
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

  /// @brief Destructor. Memory is deallocated only if it was allocated at construction.
  ~MemoryChunk() {
    deallocate();
  }

  /// @brief Returns a pointer to the underlying memory at a given index.
  ///
  /// @param index index of the position
  /// @pre @p index < @p size
  T* operator()(size_t index) {
    assert(index < size_);
    return ptr_ + index;
  }
  const T* operator()(size_t index) const {
    assert(index < size_);
    return ptr_ + index;
  }

  /// @brief Returns a pointer to the underlying memory.
  /// If @p size == 0 a @c nullptr is returned.
  T* operator()() {
    return ptr_;
  }
  const T* operator()() const {
    return ptr_;
  }

  /// @brief Returns the number of elements of type @c T allocated.
  std::size_t size() const {
    return size_;
  }

private:
  void deallocate() {
    if (allocated_) {
#ifdef DLAF_WITH_CUDA
      if (device == Device::CPU) {
        cudaFreeHost(ptr_);
      }
      else {
        cudaFree(ptr_);
      }
#else
      if (device == Device::CPU) {
        std::free(ptr_);
      }
#endif
    }
  }

  std::size_t size_;
  T* ptr_;
  bool allocated_;
};

}  // namespace memory
}  // namespace dlaf
