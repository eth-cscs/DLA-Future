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

#include <cassert>
#include <type_traits>

namespace dlaf {
namespace common {

/// @brief Buffer concept
///
/// This concept allows to access an underlying memory buffer containing data.
/// Data is viewed as a structure of optionally strided blocks of blocksize elements each.
///
/// It has reference semantic and it does not own the underlying memory buffer.
template <class T, class Enable = void>
struct Buffer;

template <class T>
struct Buffer<T*, void> {
  /// @brief Create a buffer pointing to contiguous data
  ///
  /// Create a Buffer pointing to @p n contiguous elements of type @p T starting at @p ptr
  /// @param ptr  pointer to the first element of the underlying contiguous buffer
  /// @param n    number of elements
  Buffer(T* ptr, std::size_t n) noexcept : Buffer(ptr, 1, n, 0) {}

  /// @brief Create a Buffer pointing to data with given structure
  ///
  /// Create a Buffer pointing to structured data.
  /// @param ptr        pointer to the first element of the underlying buffer
  /// @param num_blocks number of blocks
  /// @param blocksize  number of contiguous elements of type @p T in each block
  /// @param stride     stride (in elements) between starts of adjacent blocks
  /// @pre num_blocks != 0
  /// @pre stride == 0 if num_blocks == 1
  /// @pre stride >= blocksize if num_blocks > 1
  Buffer(T* ptr, std::size_t num_blocks, std::size_t blocksize, std::size_t stride) noexcept
      : data_(ptr), nblocks_(num_blocks), blocksize_(blocksize), stride_(stride) {
    assert(nblocks_ != 0);
    assert(nullptr != data_);
    assert(nblocks_ == 1 ? stride_ == 0 : stride_ >= blocksize_);
  }

  T* data() noexcept {
    return data_;
  }

  const T* data() const noexcept {
    return data_;
  }

  std::size_t nblocks() const noexcept {
    return nblocks_;
  }

  std::size_t blocksize() const noexcept {
    return blocksize_;
  }

  std::size_t stride() const noexcept {
    return stride_;
  }

protected:
  T* data_;
  std::size_t nblocks_;
  std::size_t blocksize_;
  std::size_t stride_;
};

/// @brief Helper class for creatig a buffer from a bounded array
template <class T, std::size_t N>
struct Buffer<T[N], std::enable_if_t<std::is_array<T[N]>::value>> : Buffer<std::decay_t<T[N]>> {
  /// @brief Create a Buffer from a given bounded C-array
  Buffer(T array[N]) noexcept : Buffer<T*>(array, std::extent<T[N]>::value) {}
};

/// @brief fallback function for creating a buffer
///
/// If you want to create a dlaf::common::Buffer, it is preferred to use make_buffer()\n
/// \n
/// Override this in the same namespace of a type for which you want to provide this concept.
template <class T, class... Ts>
auto create_buffer(T data, Ts... args) noexcept {
  return dlaf::common::Buffer<T>{data, static_cast<std::size_t>(args)...};
}

/// @brief Generic API for creating a buffer
///
/// Use this function to create a dlaf::common::Buffer from the given parameters
template <class T, class... Ts>
auto make_buffer(T data, Ts... args) noexcept {
  return create_buffer(data, static_cast<std::size_t>(args)...);
}

// API for algorithms
/// @brief Return the pointer to data of the given Buffer
template <class Buffer>
auto get_pointer(Buffer& buffer) noexcept -> decltype(buffer.data()) {
  return buffer.data();
}

/// @brief Return the number of blocks in the given Buffer
template <class Buffer>
auto get_num_blocks(const Buffer& buffer) noexcept -> decltype(buffer.nblocks()) {
  return buffer.nblocks();
}

/// @brief Return the block size in the given Buffer
template <class Buffer>
auto get_blocksize(const Buffer& buffer) noexcept -> decltype(buffer.blocksize()) {
  return buffer.blocksize();
}

/// @brief Return the stride (in elements) of the given Buffer
template <class Buffer>
auto get_stride(const Buffer& buffer) noexcept -> decltype(buffer.stride()) {
  return buffer.stride();
}

/// @brief Traits for Buffer concept
///
/// buffer_traits<Buffer>::element_t represents the data type for the element of the buffer
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
