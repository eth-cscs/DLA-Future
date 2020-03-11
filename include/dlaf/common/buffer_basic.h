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

#include "dlaf/common/buffer.h"

namespace dlaf {
namespace common {

template <class T>
struct BufferBasic;

/// @brief Buffer concept
///
/// This concept allows to access an underlying memory buffer containing data.
/// Data is viewed as a structure of optionally strided blocks of blocksize elements each.
///
/// It has reference semantic and it does not own the underlying memory buffer.
template <class T>
struct BufferBasic {
  /// @brief Create a buffer pointing to contiguous data
  ///
  /// Create a Buffer pointing to @p n contiguous elements of type @p T starting at @p ptr
  /// @param ptr  pointer to the first element of the underlying contiguous buffer
  /// @param n    number of elements
  BufferBasic(T* ptr, std::size_t n) noexcept : BufferBasic(ptr, 1, n, 0) {}

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
  BufferBasic(T* ptr, std::size_t num_blocks, std::size_t blocksize, std::size_t stride) noexcept
      : data_(ptr), nblocks_(num_blocks), blocksize_(blocksize), stride_(num_blocks == 1 ? 0 : stride) {
    assert(nblocks_ != 0);
    assert(nullptr != data_);
    assert(nblocks_ == 1 ? stride_ == 0 : stride_ >= blocksize_);

    if (blocksize_ == stride_) {
      blocksize_ = num_blocks * blocksize_;
      nblocks_ = 1;
      stride_ = 0;
    }
  }

  template <class Buffer, class T_ = T, class = std::enable_if_t<std::is_const<T_>::value>>
  BufferBasic(const Buffer& buffer) {
    data_ = buffer_pointer(buffer);
    nblocks_ = buffer_nblocks(buffer);
    blocksize_ = buffer_blocksize(buffer);
    stride_ = buffer_stride(buffer);
  }

  T* data() const noexcept {
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

  std::size_t count() const noexcept {
    return is_contiguous() ? blocksize() * nblocks() : (nblocks() * blocksize());
  }

  bool is_contiguous() const noexcept {
    return nblocks_ == 1 || stride_ == blocksize_;
  }

protected:
  BufferBasic() {
    data_ = nullptr;
    nblocks_ = 0;
    blocksize_ = 0;
    stride_ = 0;
  }

  T* data_;
  std::size_t nblocks_;
  std::size_t blocksize_;
  std::size_t stride_;
};

/// @brief Helper class for creatig a buffer from a bounded array
template <class T, std::size_t N>
struct BufferBasic<T[N]> : BufferBasic<T> {
  /// @brief Create a Buffer from a given bounded C-array
  BufferBasic(T array[N]) noexcept : BufferBasic<T>(array, std::extent<T[N]>::value) {}
};

/// @brief Helper class for creating a buffer that owns the underlying data
template <class T>
struct BufferWithMemory : public BufferBasic<T> {
  static_assert(not std::is_const<T>::value,
                "It is not worth to create a readonly buffer that no one can write to");

  BufferWithMemory() = default;

  BufferWithMemory(std::unique_ptr<T[]>&& memory, const std::size_t N)
      : BufferBasic<T>(memory.get(), N), memory_(std::move(memory)) {}

  BufferWithMemory(const std::size_t N) : BufferWithMemory(std::make_unique<T[]>(N), N) {}

protected:
  std::unique_ptr<T[]> memory_;
};

static_assert(std::is_convertible<BufferBasic<int>, BufferBasic<const int>>::value, "");
static_assert(not std::is_convertible<BufferBasic<const int>, BufferBasic<int>>::value, "");

template <class T>
struct buffer_traits<BufferBasic<T>> {
  using element_t = T;
};

template <class T, std::size_t N>
struct buffer_traits<BufferBasic<T[N]>> : buffer_traits<BufferBasic<T>> {};

template <class T>
struct buffer_traits<BufferWithMemory<T>> : buffer_traits<BufferBasic<T>> {};

/// Create a contiguous temporary BufferBasic
///
/// Create a temporary buffer that allows to store contiguously all the elements of the given buffer
template <class BufferIn>
auto create_temporary_buffer(const BufferIn& input) {
  using DataT = std::remove_const_t<typename common::buffer_traits<BufferIn>::element_t>;

  assert(buffer_count(input) > 0);

  return common::BufferWithMemory<DataT>(buffer_count(input));
}

}
}
