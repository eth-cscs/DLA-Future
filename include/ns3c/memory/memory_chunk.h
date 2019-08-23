//
// NS3C
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
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif
#include "ns3c/types.h"

namespace ns3c {
namespace memory {

/// The class MemoryChunk represents a layer of abstraction over the underlying host memory.

template <class T, Device device>
class MemoryChunk {
public:
  using ElementType = T;

  /// @brief Creates a MemoryChunk object with size 0.
  MemoryChunk() : MemoryChunk(0) {}

  /// @brief Creates a MemoryChunk object allocating the required memory.
  ///
  /// @param size The size of the memory to be allocated.
  /// @pre size >= 0
  ///
  /// Memory of @a size elements of type @c T are is allocated on the given device.
  MemoryChunk(std::size_t size) : size_(size), ptr_(nullptr), allocated_(true) {
    assert(size_ >= 0);
    if (size == 0)
      return;

#ifdef WITH_CUDA
    if (device == Device::CPU) {
      cudaMallocHost(&ptr_, size_ * sizeof(T));
    }
    else {
      cudaMalloc(&ptr_, size_ * sizeof(T));
    }
#else
    if (device == Device::CPU) {
      ptr_ = (T*) std::malloc(size_ * sizeof(T));
    }
    else {
      std::terminate();
    }
#endif
  }

  /// @brief Creates a MemoryChunk object from an existing memory allocation.
  ///
  /// @param ptr  The pointer to the already allocated memory.
  /// @param size The size (in number of elements of type T) of the existing allocation.
  /// @pre size >= 0
  MemoryChunk(T* ptr, std::size_t size)
      : size_(size), ptr_(size > 0 ? ptr : nullptr), allocated_(false) {
    assert(size_ >= 0);
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
  /// @pre 0 <= index < size
  T* operator()(size_t index) {
    return ptr_ + index;
  }
  const T* operator()(size_t index) const {
    return ptr_ + index;
  }

  /// @brief Returns a pointer to the underlying memory.
  /// If size == 0 a nullptr is returned.
  T* operator()() {
    return ptr_;
  }
  const T* operator()() const {
    return ptr_;
  }

  std::size_t size() const {
    return size_;
  }

private:
  void deallocate() {
    if (allocated_) {
#ifdef WITH_CUDA
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
}  // namespace ns3c
