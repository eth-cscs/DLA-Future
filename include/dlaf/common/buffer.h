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
#include <type_traits>

namespace dlaf {
namespace common {

template <class T, class Enable = void>
struct Buffer;

/// buffer by reference
template <class T>
struct Buffer<T*, void> {
  Buffer(T* ptr, std::size_t n) : Buffer(ptr, 1, n, 0) {}
  Buffer(T* ptr, std::size_t num_blocks, std::size_t blocksize, std::size_t stride)
      : ptr_(ptr), nblocks_(num_blocks), blocksize_(blocksize), stride_(stride) {
    assert(nblocks_ == 1 ? stride_ == 0 : stride_ > blocksize_);
  }

  T* ptr() {
    return ptr_;
  }

  const T* ptr() const {
    return ptr_;
  }

  std::size_t nblocks() const {
    return nblocks_;
  }

  std::size_t blocksize() const {
    return blocksize_;
  }

  std::size_t stride() const {
    return stride_;
  }

protected:
  T* ptr_;
  std::size_t nblocks_;
  std::size_t blocksize_;
  std::size_t stride_;
};

/// helper class for bounded contiguous array
template <class T, std::size_t N>
struct Buffer<T[N], std::enable_if_t<std::is_array<T[N]>::value>> : Buffer<std::decay_t<T[N]>> {
  Buffer(T array[N]) : Buffer<T*>(array, std::extent<T[N]>::value) {}
};

// API for algorithms
template <class Buffer>
auto get_pointer(Buffer& buffer) -> decltype(buffer.ptr()) {
  return buffer.ptr();
}

template <class Buffer>
auto get_num_blocks(const Buffer& buffer) -> decltype(buffer.nblocks()) {
  return buffer.nblocks();
}

template <class Buffer>
auto get_blocksize(const Buffer& buffer) -> decltype(buffer.blocksize()) {
  return buffer.blocksize();
}

template <class Buffer>
auto get_stride(const Buffer& buffer) -> decltype(buffer.stride()) {
  return buffer.stride();
}

template <class Buffer>
struct buffer_traits;

template <class T>
struct buffer_traits<Buffer<T*>> {
  using element_t = T;
};

template <class T, std::size_t N>
struct buffer_traits<Buffer<T[N]>> : buffer_traits<Buffer<std::decay_t<T[N]>>> {};

}
}

template <class T, class... Ts>
auto make_buffer(T data, Ts... args) {
  return dlaf::common::Buffer<T>{data, static_cast<std::size_t>(args)...};
}
