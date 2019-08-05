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

template <class T, typename Allocator = std::allocator<T>>
class Host {
  public:
  using ElementType = T;
  using AllocatorType = Allocator;

  // normal constructor, optional allocator
  Host(std::size_t size, Allocator alloc = Allocator()) : size_(size), ptr_(nullptr), alloc_(alloc) {
#ifdef WITH_CUDA
    cudaMallocHost(&ptr_, size_ * sizeof(T));
#else
    ptr_ = alloc_.allocate(size_);
#endif
  }

  // construct from pointer, do not initialize allocator
  Host(T* ptr) : size_(0), ptr_(ptr) {}

  // copy constructor
  Host(const Host& rhs) : size_(rhs.size_), ptr_(nullptr), alloc_(rhs.alloc_) {
#ifdef WITH_CUDA
    cudaMallocHost(&ptr_, size_ * sizeof(T));
#else
    ptr_ = alloc_.allocate(size_);
#endif
    std::copy(rhs(0), rhs(size_), ptr_);
  }

  // move constructor
  Host(Host&& rhs) : size_(rhs.size_), ptr_(rhs.ptr_), alloc_(rhs.alloc_) {
    rhs.ptr_ = nullptr;
  }

  // move assignement
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

  ~Host() {
    if (size_ > 0) {
#ifdef WITH_CUDA
      cudaFreeHost(ptr_);
#else
      alloc_.deallocate(ptr_, size_);
#endif
    }
  }

  T* operator()(size_t index) {
    return ptr_ + index;
  }

  const T* operator()(size_t index) const {
    return ptr_ + index;
  }

  T* operator()() {
    return ptr_;
  }

  const T* operator()() const {
    return ptr_;
  }

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
