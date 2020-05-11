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

#include <cstdlib>
#include <memory>

#include "memory_chunk.h"
#include "dlaf/common/assert.h"
#include "dlaf/types.h"

namespace dlaf {
namespace memory {

/// @brief The class @c MemoryView represents a layer of abstraction over the underlying host memory.
///
/// Two levels of constness exist for @c MemoryView analogously to pointer semantics:
/// the constness of the view and the constness of the data referenced by the view.
/// Implicit conversion is allowed from views of non-const elements to views of const elements.
template <class T, Device device>
class MemoryView {
public:
  using ElementType = std::remove_const_t<T>;
  friend MemoryView<const ElementType, device>;

  /// @brief Creates a MemoryView of size 0.
  MemoryView() : memory_(std::make_shared<MemoryChunk<ElementType, device>>()), offset_(0), size_(0) {}

  /// @brief Creates a MemoryView object allocating the required memory.
  ///
  /// @param size The size of the memory to be allocated.
  ///
  /// Memory of @p size elements of type @c T is allocated on the given device.
  template <class U = T,
            class = typename std::enable_if_t<!std::is_const<U>::value && std::is_same<T, U>::value>>
  explicit MemoryView(std::size_t size)
      : memory_(std::make_shared<MemoryChunk<ElementType, device>>(size)), offset_(0), size_(size) {}

  /// @brief Creates a MemoryView object from an existing memory allocation.
  ///
  /// @param ptr  The pointer to the already allocated memory.
  /// @param size The size (in number of elements of type @c T) of the existing allocation.
  /// @pre @p ptr+i can be deferenced for 0 < @c i < @p size
  MemoryView(T* ptr, std::size_t size)
      : memory_(std::make_shared<MemoryChunk<ElementType, device>>(const_cast<ElementType*>(ptr), size)),
        offset_(0), size_(size) {}

  MemoryView(const MemoryView&) = default;
  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  MemoryView(const MemoryView<ElementType, device>& rhs)
      : memory_(rhs.memory_), offset_(rhs.offset_), size_(rhs.size_) {}

  MemoryView(MemoryView&& rhs) : memory_(rhs.memory_), offset_(rhs.offset_), size_(rhs.size_) {
    rhs.memory_ = std::make_shared<MemoryChunk<ElementType, device>>();
    rhs.size_ = 0;
    rhs.offset_ = 0;
  }
  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  MemoryView(MemoryView<ElementType, device>&& rhs)
      : memory_(rhs.memory_), offset_(rhs.offset_), size_(rhs.size_) {
    rhs.memory_ = std::make_shared<MemoryChunk<ElementType, device>>();
    rhs.size_ = 0;
    rhs.offset_ = 0;
  }

  /// @brief Creates a MemoryView object which is a subview of another MemoryView.
  ///
  /// @param memory_view The starting MemoryView object.
  /// @param offset      The index of the first element of the subview.
  /// @param size        The size (in number of elements of type @c T) of the subview.
  /// @pre subview should not exceeds the limits of @p memory_view
  MemoryView(const MemoryView& memory_view, std::size_t offset, std::size_t size)
      : memory_(size > 0 ? memory_view.memory_ : std::make_shared<MemoryChunk<ElementType, device>>()),
        offset_(size > 0 ? offset + memory_view.offset_ : 0), size_(size) {
    DLAF_ASSERT((offset + size <= memory_view.size_),
                "Sub MemoryView exceeds the limits of the base MemoryView");
  }
  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  MemoryView(const MemoryView<ElementType, device>& memory_view, std::size_t offset, std::size_t size)
      : memory_(size > 0 ? memory_view.memory_ : std::make_shared<MemoryChunk<ElementType, device>>()),
        offset_(size > 0 ? offset + memory_view.offset_ : 0), size_(size) {
    DLAF_ASSERT((offset + size <= memory_view.size_),
                "Sub MemoryView exceeds the limits of the base MemoryView");
  }

  MemoryView& operator=(const MemoryView&) = default;
  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  MemoryView& operator=(const MemoryView<ElementType, device>& rhs) {
    memory_ = rhs.memory_;
    offset_ = rhs.offset_;
    size_ = rhs.size_;

    return *this;
  }

  MemoryView& operator=(MemoryView&& rhs) {
    memory_ = std::move(rhs.memory_);
    offset_ = rhs.offset_;
    size_ = rhs.size_;

    rhs.memory_ = std::make_shared<MemoryChunk<ElementType, device>>();
    rhs.size_ = 0;
    rhs.offset_ = 0;

    return *this;
  }
  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  MemoryView& operator=(MemoryView<ElementType, device>&& rhs) {
    memory_ = std::move(rhs.memory_);
    offset_ = rhs.offset_;
    size_ = rhs.size_;

    rhs.memory_ = std::make_shared<MemoryChunk<ElementType, device>>();
    rhs.size_ = 0;
    rhs.offset_ = 0;

    return *this;
  }

  /// @brief Returns a pointer to the underlying memory at a given index.
  ///
  /// @param index index of the position
  /// @pre @p index < @p size
  T* operator()(size_t index) const {
    DLAF_ASSERT_HEAVY((index < size_));
    return memory_->operator()(offset_ + index);
  }

  /// @brief Returns a pointer to the underlying memory.
  /// If @p size == 0 a @c nullptr is returned.
  T* operator()() const {
    return size_ == 0 ? nullptr : memory_->operator()(offset_);
  }

  /// @brief Returns the number of elements accessible from the MemoryView.
  std::size_t size() const {
    return size_;
  }

private:
  std::shared_ptr<MemoryChunk<ElementType, device>> memory_;
  std::size_t offset_;
  std::size_t size_;
};

/// ---- ETI

#define DLAF_MEMVIEW_ETI(KWORD, DATATYPE, DEVICE) KWORD template class MemoryView<DATATYPE, DEVICE>;

DLAF_MEMVIEW_ETI(extern, float, Device::CPU)
DLAF_MEMVIEW_ETI(extern, double, Device::CPU)
DLAF_MEMVIEW_ETI(extern, std::complex<float>, Device::CPU)
DLAF_MEMVIEW_ETI(extern, std::complex<double>, Device::CPU)

// DLAF_MEMVIEW_ETI(extern, float, Device::GPU)
// DLAF_MEMVIEW_ETI(extern, double, Device::GPU)
// DLAF_MEMVIEW_ETI(extern, std::complex<float>, Device::GPU)
// DLAF_MEMVIEW_ETI(extern, std::complex<double>, Device::GPU)

}
}
