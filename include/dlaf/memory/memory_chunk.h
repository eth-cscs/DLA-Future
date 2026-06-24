//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>

#include <umpire/Allocator.hpp>

#include <dlaf/memory/allocation_types.h>
#include <dlaf/memory/memory_type.h>
#include <dlaf/types.h>

namespace dlaf::memory {
namespace internal {
umpire::Allocator& getUmpireHostAllocator();
void initializeUmpireHostAllocator(std::size_t initial_block_bytes, std::size_t next_block_bytes,
                                   std::size_t alignment_bytes, double coalesce_free_ratio,
                                   double coalesce_reallocation_ratio);
void finalizeUmpireHostAllocator();

#ifdef DLAF_WITH_GPU
void initializeUmpireDeviceAllocator(std::size_t initial_block_bytes, std::size_t next_block_bytes,
                                     std::size_t alignment_bytes, double coalesce_free_ratio,
                                     double coalesce_reallocation_ratio);
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
  MemoryChunk() : MemoryChunk(0, AllocateOn::Construction) {}

  /// Creates a MemoryChunk object allocating the required memory.
  ///
  /// @param size The size of the memory to be allocated.
  ///
  /// Memory of @a size elements of type @c T are is allocated on the given device.
  MemoryChunk(SizeType size, AllocateOn allocate_on)
      : size_(size), ptr_(nullptr), status_(initial_allocation_status(size)) {
    DLAF_ASSERT(size >= 0, size);

    if (size > 0 && allocate_on == AllocateOn::Construction) {
      allocate();
    }
  }

  /// Creates a MemoryChunk object from an existing memory allocation.
  ///
  /// @param ptr  The pointer to the already allocated memory,
  /// @param size The size (in number of elements of type @c T) of the existing allocation,
  /// @pre @p ptr+i can be dereferenced for 0 <= @c i < @p size.
  MemoryChunk(T* ptr, SizeType size)
      : size_(size), ptr_(size > 0 ? ptr : nullptr),
        status_(internal::AllocationStatus::ExternallyManaged) {
    DLAF_ASSERT_HEAVY(size == 0 ? ptr_ == nullptr : ptr_ != nullptr, size);

    using dlaf::memory::internal::get_memory_type;
    using dlaf::memory::internal::MemoryType;
    auto memory_type = get_memory_type(ptr);
    switch (memory_type) {
      case MemoryType::Host:
        DLAF_ASSERT(D == Device::CPU, "Using a host pointer to construct a device MemoryChunk");
        break;
      case MemoryType::Device:
        DLAF_ASSERT(D == Device::GPU, "Using a device pointer to construct a host MemoryChunk");
        break;
      case MemoryType::Managed:
        break;
      case MemoryType::Unified:
        break;
    }
  }

  MemoryChunk(const MemoryChunk&) = delete;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
  /// Move constructor.
  MemoryChunk(MemoryChunk&& rhs) noexcept
      : size_(rhs.size_), ptr_(rhs.ptr_), status_(rhs.status_.load(std::memory_order_relaxed)) {
    rhs.ptr_ = nullptr;
    rhs.size_ = 0;
    rhs.status_.store(internal::AllocationStatus::Empty, std::memory_order_relaxed);
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
    status_ = rhs.status_.load(std::memory_order_relaxed);

    rhs.size_ = 0;
    rhs.ptr_ = nullptr;
    rhs.status_.store(internal::AllocationStatus::Empty, std::memory_order_relaxed);

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
    return (*this)() + index;
  }
  const T* operator()(SizeType index) const {
    DLAF_ASSERT_HEAVY(index < size_, index, size_);
    return (*this)() + index;
  }

  /// Returns a pointer to the underlying memory.
  /// If @p size == 0 a @c nullptr is returned.
  T* operator()() {
    using internal::AllocationStatus;
    if (status_.load(std::memory_order_acquire) == AllocationStatus::WaitAllocation) {
      allocate();
    }
    DLAF_ASSERT_HEAVY(status_.load(std::memory_order_acquire) != AllocationStatus::WaitAllocation,
                      "DLAF Internal allocation error");
    DLAF_ASSERT_HEAVY(size_ == 0 || ptr_ != nullptr, size_);
    return ptr_;
  }
  const T* operator()() const {
    using internal::AllocationStatus;
    DLAF_ASSERT(status_.load(std::memory_order_acquire) != AllocationStatus::WaitAllocation,
                "DLAF Internal allocation error");
    DLAF_ASSERT_HEAVY(size_ == 0 || ptr_ != nullptr, size_);
    return ptr_;
  }

  /// Returns the number of elements of type @c T allocated.
  SizeType size() const {
    return size_;
  }

private:
  static internal::AllocationStatus initial_allocation_status(SizeType size) noexcept {
    using internal::AllocationStatus;
    if (size == 0)
      return AllocationStatus::Empty;
    return AllocationStatus::WaitAllocation;
  }

  void allocate() {
    using internal::AllocationStatus;
    std::lock_guard lock(allocation_mutex);
    if (status_.load(std::memory_order_acquire) == AllocationStatus::WaitAllocation) {
      std::size_t mem_size = static_cast<std::size_t>(size_) * sizeof(T);
      if (D == Device::CPU) {
        ptr_ = static_cast<T*>(internal::getUmpireHostAllocator().allocate(mem_size));
      }
#ifdef DLAF_WITH_GPU
      else {
        ptr_ = static_cast<T*>(internal::getUmpireDeviceAllocator().allocate(mem_size));
      }
#else
      else {
        std::cout
            << "[ERROR] GPU memory was requested but the `DLAF_WITH_CUDA` or `DLAF_WITH_HIP` flags were not passed!";
        std::terminate();
      }
#endif
      status_.store(AllocationStatus::Allocated, std::memory_order_release);
    }
  }

  void deallocate() {
    using internal::AllocationStatus;
    if (status_.load(std::memory_order_acquire) == AllocationStatus::Allocated) {
      if (D == Device::CPU) {
        internal::getUmpireHostAllocator().deallocate(ptr_);
      }
#ifdef DLAF_WITH_GPU
      else {
        internal::getUmpireDeviceAllocator().deallocate(ptr_);
      }
#endif
    }
  }

  SizeType size_;
  T* ptr_;
  std::atomic<internal::AllocationStatus> status_;
  std::mutex allocation_mutex;
};

}
