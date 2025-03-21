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
#include <memory>
#include <type_traits>
#include <utility>

#include <dlaf/common/assert.h>
#include <dlaf/types.h>

namespace dlaf {
namespace common {

#ifdef DLAF_DOXYGEN

/// Traits for verifying if the given type is an implementation of the Data concept.
///
/// Derive from @p std::true_type or @p std::false_type.
template <class Data>
struct is_data {};

#else
template <class Data, class = void>
struct is_data : std::false_type {};

template <class Data>
struct is_data<
    Data,
    std::enable_if_t<std::is_pointer_v<decltype(data_pointer(std::declval<const Data&>()))> &&
                     std::is_same_v<decltype(data_nblocks(std::declval<const Data&>())), SizeType> &&
                     std::is_same_v<decltype(data_blocksize(std::declval<const Data&>())), SizeType> &&
                     std::is_same_v<decltype(data_stride(std::declval<const Data&>())), SizeType> &&
                     std::is_same_v<decltype(data_count(std::declval<const Data&>())), SizeType> &&
                     std::is_same_v<decltype(data_iscontiguous(std::declval<const Data&>())), bool>>>
    : std::true_type {};
#endif

template <class T>
inline constexpr bool is_data_v = is_data<T>::value;

/// Traits for accessing properties of the given Data concept.
///
/// data_traits<Data>::element_t is the data type of the elements.
template <class Data>
struct data_traits;

// Forward declaration of DataDescriptor class
template <class T>
struct DataDescriptor;

/// Fallback function for creating a DataDescriptor.
///
/// If you want to create a DataDescriptor, it is preferred to use common::make_data().
/// Override this in the same namespace of the type for which you want to provide this concept.
template <class T, class... Ts>
auto create_data(T* data, Ts&&... args) noexcept {
  return DataDescriptor<T>(data, args...);
}

/// Generic API for creating a Data.
///
/// This is an helper function that given a Data, returns exactly it as it is
/// It allows to use the make_data function for dealing both with common::DataDescriptor
/// and anything that can be converted to a Data, without code duplication in user code.
template <class Data, std::enable_if_t<is_data_v<Data>, int> = 0>
auto make_data(Data&& data) noexcept {
  return std::forward<Data>(data);
}

/// Generic API for creating a Data.
///
/// Use this function to create a Data from the given parameters
/// This is the entry point for anything that can be converted to a DataDescriptor
/// It exploits ADL to call the specialized create_data() function.
///
/// Note:
/// exploiting ADL requires that at least one parameter is specific and inside the
/// namespace where the create_data() funtion is.
template <class T, class... Ts>
auto make_data(T&& data, Ts&&... args) noexcept {
  return create_data(std::forward<T>(data), std::forward<Ts>(args)...);
}

// API for algorithms.
/// Return the pointer to data of the given Data.
template <class Data>
auto data_pointer(const Data& data) noexcept -> decltype(data.data()) {
  return data.data();
}

/// Return the number of blocks in the given Data.
template <class Data>
auto data_nblocks(const Data& data) noexcept -> decltype(data.nblocks()) {
  return data.nblocks();
}

/// Return the block size in the given Data.
template <class Data>
auto data_blocksize(const Data& data) noexcept -> decltype(data.blocksize()) {
  return data.blocksize();
}

/// Return the stride (in elements) of the given Data.
template <class Data>
auto data_stride(const Data& data) noexcept -> decltype(data.stride()) {
  return data.stride();
}

/// Return the number of elements stored in the given Data.
template <class Data>
auto data_count(const Data& data) noexcept -> decltype(data.count()) {
  return data.count();
}

/// Return true if Data is contiguous.
template <class Data>
auto data_iscontiguous(const Data& data) noexcept -> decltype(data.is_contiguous()) {
  return data.is_contiguous();
}

/// Generic API for copying Data.
///
/// Use this function to copy values from one Data to another.
template <class DataIn, class DataOut>
void copy(const DataIn& src, const DataOut& dest) {
  static_assert(not std::is_const_v<typename data_traits<DataOut>::element_t>,
                "Cannot copy to a const Data");

  if (data_iscontiguous(dest) && data_iscontiguous(src)) {
    DLAF_ASSERT_HEAVY(data_blocksize(src) == data_blocksize(dest), data_blocksize(src),
                      data_blocksize(dest));

    std::copy(data_pointer(src), data_pointer(src) + data_blocksize(src), data_pointer(dest));
  }
  else {
    if (data_iscontiguous(dest)) {
      DLAF_ASSERT_HEAVY(data_nblocks(src) * data_blocksize(src) == data_blocksize(dest),
                        data_nblocks(src), data_blocksize(src), data_blocksize(dest));

      for (SizeType i_block = 0; i_block < data_nblocks(src); ++i_block) {
        auto ptr_block_start = data_pointer(src) + i_block * data_stride(src);
        auto dest_block_start = data_pointer(dest) + i_block * data_blocksize(src);
        std::copy(ptr_block_start, ptr_block_start + data_blocksize(src), dest_block_start);
      }
    }
    else if (data_iscontiguous(src)) {
      DLAF_ASSERT_HEAVY(data_blocksize(src) == data_nblocks(dest) * data_blocksize(dest),
                        data_blocksize(src), data_nblocks(dest), data_blocksize(dest));

      for (SizeType i_block = 0; i_block < data_nblocks(dest); ++i_block) {
        auto ptr_block_start = data_pointer(src) + i_block * data_blocksize(dest);
        auto dest_block_start = data_pointer(dest) + i_block * data_stride(dest);
        std::copy(ptr_block_start, ptr_block_start + data_blocksize(dest), dest_block_start);
      }
    }
    else {
      DLAF_ASSERT_HEAVY(data_count(src) == data_count(dest), data_count(src), data_count(dest));

      for (SizeType i = 0; i < data_count(src); ++i) {
        auto i_src = i / data_blocksize(src) * data_stride(src) + i % data_blocksize(src);
        auto i_dest = i / data_blocksize(dest) * data_stride(dest) + i % data_blocksize(dest);

        data_pointer(dest)[i_dest] = data_pointer(src)[i_src];
      }
    }
  }
}
}
}
