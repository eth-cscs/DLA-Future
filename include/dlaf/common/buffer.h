//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

namespace dlaf {
namespace common {

template <class Buffer, class = void>
struct is_buffer : std::false_type {};

template <class Buffer>
struct is_buffer<
    Buffer,
    std::enable_if_t<
        std::is_pointer<decltype(buffer_pointer(std::declval<const Buffer&>()))>::value &&
        std::is_same<decltype(buffer_nblocks(std::declval<const Buffer&>())), std::size_t>::value &&
        std::is_same<decltype(buffer_blocksize(std::declval<const Buffer&>())), std::size_t>::value &&
        std::is_same<decltype(buffer_stride(std::declval<const Buffer&>())), std::size_t>::value &&
        std::is_same<decltype(buffer_count(std::declval<const Buffer&>())), std::size_t>::value &&
        std::is_same<decltype(buffer_iscontiguous(std::declval<const Buffer&>())), bool>::value>>
    : std::true_type {};

/// @brief Traits for Buffer concept
///
/// buffer_traits<Buffer>::element_t represents the data type for the element of the buffer
template <class Buffer>
struct buffer_traits;

// Forward declaration of Buffer class
template <class T>
struct BufferBasic;

/// @brief fallback function for creating a BufferBasic
///
/// If you want to create a BufferBasic, it is preferred to use make_buffer()
/// Override this in the same namespace of a type for which you want to provide this concept.
template <class T, class... Ts>
auto create_buffer(T* data, Ts&&... args) noexcept {
  return BufferBasic<T>(data, static_cast<std::size_t>(args)...);
}

/// @brief Generic API for creating a buffer
///
/// This is an helper function that given a Buffer, returns exactly it as it is
/// It allows to use the make_buffer function for dealing both with common::BufferBasic
/// and anything that can be used to create a Buffer, without code duplication in user code
template <class Buffer, std::enable_if_t<is_buffer<Buffer>::value, int> = 0>
auto make_buffer(Buffer&& buffer) noexcept {
  return std::forward<Buffer>(buffer);
}

/// @brief Generic API for creating a buffer
///
/// Use this function to create a Buffer from the given parameters
template <class T, class... Ts>
auto make_buffer(T&& data, Ts&&... args) noexcept {
  return create_buffer(std::forward<T>(data), std::forward<Ts>(args)...);
}

// API for algorithms
/// @brief Return the pointer to data of the given Buffer
template <class Buffer>
auto buffer_pointer(const Buffer& buffer) noexcept -> decltype(buffer.data()) {
  return buffer.data();
}

/// @brief Return the number of blocks in the given Buffer
template <class Buffer>
auto buffer_nblocks(const Buffer& buffer) noexcept -> decltype(buffer.nblocks()) {
  return buffer.nblocks();
}

/// @brief Return the block size in the given Buffer
template <class Buffer>
auto buffer_blocksize(const Buffer& buffer) noexcept -> decltype(buffer.blocksize()) {
  return buffer.blocksize();
}

/// @brief Return the stride (in elements) of the given Buffer
template <class Buffer>
auto buffer_stride(const Buffer& buffer) noexcept -> decltype(buffer.stride()) {
  return buffer.stride();
}

/// @brief Return the number of elements stored in the given Buffer
template <class Buffer>
auto buffer_count(const Buffer& buffer) noexcept -> decltype(buffer.count()) {
  return buffer.count();
}

/// @brief Return true if the buffer is contiguous
template <class Buffer>
auto buffer_iscontiguous(const Buffer& buffer) noexcept -> decltype(buffer.is_contiguous()) {
  return buffer.is_contiguous();
}

/// @brief Generic API for copying a Buffer
///
/// Use this function to copy data from one Buffer to another
template <class BufferIn, class BufferOut>
void copy(const BufferIn& src, const BufferOut& dest) {
  static_assert(not std::is_const<typename buffer_traits<BufferOut>::element_t>::value,
                "Cannot copy to a Buffer of const data");

  if (buffer_iscontiguous(dest) && buffer_iscontiguous(src)) {
    assert(buffer_blocksize(src) == buffer_blocksize(dest));

    std::copy(buffer_pointer(src), buffer_pointer(src) + buffer_blocksize(src), buffer_pointer(dest));
  }
  else {
    if (buffer_iscontiguous(dest)) {
      assert(buffer_nblocks(src) * buffer_blocksize(src) == buffer_blocksize(dest));

      for (std::size_t i_block = 0; i_block < buffer_nblocks(src); ++i_block) {
        auto ptr_block_start = buffer_pointer(src) + i_block * buffer_stride(src);
        auto dest_block_start = buffer_pointer(dest) + i_block * buffer_blocksize(src);
        std::copy(ptr_block_start, ptr_block_start + buffer_blocksize(src), dest_block_start);
      }
    }
    else if (buffer_iscontiguous(src)) {
      assert(buffer_blocksize(src) == buffer_nblocks(dest) * buffer_blocksize(dest));

      for (std::size_t i_block = 0; i_block < buffer_nblocks(dest); ++i_block) {
        auto ptr_block_start = buffer_pointer(src) + i_block * buffer_blocksize(dest);
        auto dest_block_start = buffer_pointer(dest) + i_block * buffer_stride(dest);
        std::copy(ptr_block_start, ptr_block_start + buffer_blocksize(dest), dest_block_start);
      }
    }
    else {
      assert(buffer_count(src) == buffer_count(dest));

      for (std::size_t i = 0; i < buffer_count(src); ++i) {
        auto i_src = i / buffer_blocksize(src) * buffer_stride(src) + i % buffer_blocksize(src);
        auto i_dest = i / buffer_blocksize(dest) * buffer_stride(dest) + i % buffer_blocksize(dest);

        buffer_pointer(dest)[i_dest] = buffer_pointer(src)[i_src];
      }
    }
  }
}
}
}
