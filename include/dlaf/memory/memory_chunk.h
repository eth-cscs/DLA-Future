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

#include <cstdlib>
#include <memory>

#include <umpire/Allocator.hpp>

#include "dlaf/types.h"

namespace dlaf {
namespace memory {

namespace internal {
umpire::Allocator& getUmpireHostAllocator();
void initializeUmpireHostAllocator(std::size_t initial_bytes);
void finalizeUmpireHostAllocator();

#ifdef DLAF_WITH_GPU
void initializeUmpireDeviceAllocator(std::size_t initial_bytes);
void finalizeUmpireDeviceAllocator();
umpire::Allocator& getUmpireDeviceAllocator();
#endif
}

/// The class @c MemoryChunk represents a layer of abstraction over the underlying device memory.
template <class T, Device D>
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
    DLAF_ASSERT(size >= 0, size);

    if (size == 0)
      return;

    std::size_t mem_size = static_cast<std::size_t>(size_) * sizeof(T);
#ifdef DLAF_WITH_GPU
    if (D == Device::CPU) {
      ptr_ = static_cast<T*>(internal::getUmpireHostAllocator().allocate(mem_size));
    }
    else {
      ptr_ = static_cast<T*>(internal::getUmpireDeviceAllocator().allocate(mem_size));
    }
#else
    if (D == Device::CPU) {
      ptr_ = static_cast<T*>(internal::getUmpireHostAllocator().allocate(mem_size));
    }
    else {
      std::cout
          << "[ERROR] GPU memory was requested but the `DLAF_WITH_CUDA` or `DLAF_WITH_HIP` flags were not passed!";
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
    DLAF_ASSERT_HEAVY(size == 0 ? ptr_ == nullptr : ptr_ != nullptr, size);
  }

  MemoryChunk(const MemoryChunk&) = delete;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
  /// Move constructor.
  MemoryChunk(MemoryChunk&& rhs) noexcept
      : size_(rhs.size_), ptr_(rhs.ptr_), allocated_(rhs.allocated_) {
    rhs.ptr_ = nullptr;
    rhs.size_ = 0;
    rhs.allocated_ = false;
  }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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
    DLAF_ASSERT_HEAVY(index < size_, index, size_);
    return ptr_ + index;
  }
  const T* operator()(SizeType index) const {
    DLAF_ASSERT_HEAVY(index < size_, index, size_);
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
#ifdef DLAF_WITH_GPU
      if (D == Device::CPU) {
        internal::getUmpireHostAllocator().deallocate(ptr_);
      }
      else {
        internal::getUmpireDeviceAllocator().deallocate(ptr_);
      }
#else
      if (D == Device::CPU) {
        internal::getUmpireHostAllocator().deallocate(ptr_);
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
