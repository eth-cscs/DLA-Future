//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include <dlaf/common/assert.h>
#include <dlaf/common/data.h>
#include <dlaf/types.h>

namespace dlaf {
namespace common {

/// Interface for the Data concept.
///
/// DataDescriptor is the interface implementing the Data concept (see include/dlaf/common/data.h).
/// Data is viewed as a structure of (optionally) strided blocks of blocksize elements each.
///
/// It has reference semantic and it does not own the underlying memory data.
template <class T>
struct DataDescriptor {
  /// Default constructor "null pointer".
  ///
  /// Create a DataDescriptor that does not point to anything.
  DataDescriptor() {
    data_ = nullptr;
    nblocks_ = 0;
    blocksize_ = 0;
    stride_ = 0;
  }

  /// Create a DataDescriptor pointing to contiguous data.
  ///
  /// Create a Data pointing to @p n contiguous elements of type @p T starting at @p ptr.
  /// @param ptr  pointer to the first element of the underlying contiguous data,
  /// @param n    number of elements.
  DataDescriptor(T* ptr, SizeType n) noexcept : DataDescriptor(ptr, 1, n, 0) {}

  /// Create a DataDescriptor pointing to data with given structure.
  ///
  /// Create a Data pointing to structured data.
  /// It ensures that the given structure is respected, but it is not guaranteed that it will
  /// be used as it is (e.g. if an equivalent view is available, structure parameters can be altered).
  /// @param ptr        pointer to the first element of the underlying data,
  /// @param num_blocks number of blocks,
  /// @param blocksize  number of contiguous elements of type @p T in each block,
  /// @param stride     stride (in elements) between starts of adjacent blocks,
  /// @pre num_blocks != 0,
  /// @pre stride == 0 if num_blocks == 1,
  /// @pre stride >= blocksize if num_blocks > 1.
  DataDescriptor(T* ptr, SizeType num_blocks, SizeType blocksize, SizeType stride) noexcept
      : data_(ptr), nblocks_(num_blocks), blocksize_(blocksize), stride_(num_blocks == 1 ? 0 : stride) {
    DLAF_ASSERT(num_blocks >= 0, num_blocks);
    DLAF_ASSERT(blocksize >= 0, blocksize);
    DLAF_ASSERT(stride >= 0, stride);

    if (blocksize_ == stride_) {
      blocksize_ = num_blocks * blocksize_;
      nblocks_ = 1;
      stride_ = 0;
    }
  }

  /// Conversion constructor from any Data concept.
  ///
  /// Create a DataDescriptor from any given Data concept.
  /// It allows to use the DataDescriptor as "base-class" for any implementation of Data concept.
  template <class Data, class T_ = T, class = std::enable_if_t<std::is_const_v<T_>>>
  DataDescriptor(const Data& data) {
    data_ = data_pointer(data);
    nblocks_ = data_nblocks(data);
    blocksize_ = data_blocksize(data);
    stride_ = data_stride(data);
  }

  /// Pointer to data.
  T* data() const noexcept {
    return data_;
  }

  /// Number of blocks.
  ///
  /// 1 in case of contiguous data (single block).
  SizeType nblocks() const noexcept {
    return nblocks_;
  }

  /// Number of elements in each block.
  ///
  /// For contiguous block, the return value equals to count().
  SizeType blocksize() const noexcept {
    return blocksize_;
  }

  /// Number of elements between the start of each block.
  ///
  /// 0 in case of contiguous data (single block).
  SizeType stride() const noexcept {
    return stride_;
  }

  /// Number of valid elements.
  SizeType count() const noexcept {
    return is_contiguous() ? blocksize() : (nblocks() * blocksize());
  }

  /// Return if the underlying data can be viewed as contiguous or not.
  bool is_contiguous() const noexcept {
    return nblocks_ == 1 || stride_ == blocksize_;
  }

  bool operator==(const DataDescriptor& rhs) const noexcept {
    return this == &rhs || (this->data_ == rhs.data_ && this->nblocks_ == rhs.nblocks_ &&
                            this->blocksize_ == rhs.blocksize_ && this->stride_ == rhs.stride_);
  }

  bool operator!=(const DataDescriptor& rhs) const noexcept {
    return !(*this == rhs);
  }

protected:
  T* data_;
  SizeType nblocks_;
  SizeType blocksize_;
  SizeType stride_;
};

/// Helper class for creating a DataDescriptor from a bounded C-array.
template <class T, std::size_t N>
struct DataDescriptor<T[N]> : DataDescriptor<T> {
  /// Create a Data from a given bounded C-array.
  DataDescriptor(T array[N]) noexcept : DataDescriptor<T>(array, std::extent_v<T[N]>) {}
};

/// Buffer of memory implementing the Data concept.
///
/// Simple memory buffer implementing the Data concept.
/// It owns the underlying memory.
template <class T>
struct Buffer : public DataDescriptor<T> {
  static_assert(not std::is_const_v<T>,
                "It is not worth to create a readonly data that no one can write to");

  /// Default constructor for not allocated buffer.
  Buffer() = default;

  /// Create a Buffer with given externally allocated memory.
  ///
  /// Acquire ownership of an externally allocated std::unique_ptr.
  Buffer(std::unique_ptr<T[]>&& memory, const SizeType N)
      : DataDescriptor<T>(memory.get(), N), memory_(std::move(memory)) {}

  /// Create a Buffer internally allocating the memory.
  ///
  /// Internally allocates the memory for @param N contiguous elements.
  Buffer(const SizeType N) : Buffer(std::make_unique<T[]>(static_cast<std::size_t>(N)), N) {}

  /// Return true if it is allocated, false otherwise
  operator bool() const {
    return static_cast<bool>(memory_);
  }

protected:
  std::unique_ptr<T[]> memory_;
};

static_assert(std::is_convertible_v<DataDescriptor<int>, DataDescriptor<const int>>, "");
static_assert(not std::is_convertible_v<DataDescriptor<const int>, DataDescriptor<int>>, "");

template <class T>
struct data_traits<DataDescriptor<T>> {
  using element_t = T;
};

template <class T, std::size_t N>
struct data_traits<DataDescriptor<T[N]>> : data_traits<DataDescriptor<T>> {};

template <class T>
struct data_traits<Buffer<T>> : data_traits<DataDescriptor<T>> {};

/// Create a contiguous temporary DataDescriptor.
///
/// Create a temporary data that allows to store contiguously all the elements of the given data.
template <class DataIn>
auto create_temporary_buffer(const DataIn& input) {
  using DataT = std::remove_const_t<typename common::data_traits<DataIn>::element_t>;

  DLAF_ASSERT_HEAVY(data_count(input) > 0, data_count(input));

  return common::Buffer<DataT>(data_count(input));
}

/// Helper function ensuring to work with a contiguous Data, allocating a temporary buffer if needed
///
/// @param support_buffer changed just if the @p input buffer is not contiguous
/// @return the given @p input if it is already contiguous, otherwise returns a newly allocated buffer
template <class DataIn, class Buffer>
auto make_contiguous(DataIn input, Buffer& support_buffer) {
  using DataT = typename common::data_traits<DataIn>::element_t;

  if (!input.is_contiguous()) {
    support_buffer = common::create_temporary_buffer(input);
    return common::DataDescriptor<DataT>(support_buffer);
  }

  return input;
}
}
}
