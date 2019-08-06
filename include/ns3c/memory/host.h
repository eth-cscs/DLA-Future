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

#include <cstdlib>
#include <memory>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ns3c {
namespace memory {

  /// The class Host represents a layer of abstraction over the underlying host memory.
        
template <class T, typename Allocator = std::allocator<T>>
class Host {
public:
  using ElementType = T;
  using AllocatorType = Allocator;

  /// Creates a host memory, allocating the required memory
  ///
  /// \param size The size of the memory to be allocated
  /// \param alloc (optional) Custom allocator
  ///
  /// Memory of \a size elements of type \c T are is allocated on the host. 
  /// If \a alloc is not provided, \c std::allocator is used by default
  Host(std::size_t size, Allocator alloc = Allocator()) : size_(size), ptr_(nullptr), alloc_(alloc) {
#ifdef WITH_CUDA
    cudaMallocHost(&ptr_, size_ * sizeof(T));
#else
    ptr_ = alloc_.allocate(size_);
#endif
  }
  
  /// Creates a host memory object from a memory pointer. Doesn't allocate memory
  ///
  /// \param ptr A pointer to an already allocated memory of type T
  ///
  Host(T* ptr) : size_(0), ptr_(ptr) {}

  /// Copy constructor
  Host(const Host& rhs) : size_(rhs.size_), ptr_(nullptr), alloc_(rhs.alloc_) {
#ifdef WITH_CUDA
    cudaMallocHost(&ptr_, size_ * sizeof(T));
#else
    ptr_ = alloc_.allocate(size_);
#endif
    std::copy(rhs(0), rhs(size_), ptr_);
  }

  /// Move constructor
  Host(Host&& rhs) : size_(rhs.size_), ptr_(rhs.ptr_), alloc_(rhs.alloc_) {
    rhs.ptr_ = nullptr;
  }

  /// Move assignement
  Host& operator=(Host&& rhs) {
    if (this != &rhs) {
      if (this->size > 0) {
#ifdef WITH_CUDA
        cudaFreeHost(this->ptr_);
#else
        alloc_.deallocate(this->ptr_, this->size_);
#endif
      }
      this->ptr_ = rhs.ptr_;
      this->alloc_ = rhs.alloc_;
      this->size_ = rhs.size_;
      rhs.mem_ = nullptr;
    }
  }

  /// Destructor. Memory is deallocated only if it was allocated at construction
  ~Host() {
    if (size_ > 0) {
#ifdef WITH_CUDA
      cudaFreeHost(ptr_);
#else
      alloc_.deallocate(ptr_, size_);
#endif
    }
  }
  
  /// Returns a pointer to the underlying memory at a given index
  ///
  /// \param index index of the position
  T* operator()(size_t index) {
    return ptr_ + index;
  }

  const T* operator()(size_t index) const {
    return ptr_ + index;
  }

  /// Returns a pointer to the underlying memory
  T* operator()() {
    return ptr_;
  }

  const T* operator()() const {
    return ptr_;
  }

  /// Returns the allocator
  const AllocatorType& get_allocator() {
    return alloc_;
  }

private:
  size_t size_;
  T* ptr_;
  AllocatorType alloc_;
};

}  // namespace memory
}  // namespace ns3c
