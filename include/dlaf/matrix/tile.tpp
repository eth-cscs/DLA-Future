//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

namespace dlaf {
namespace matrix {

template <class T, Device device>
Tile<const T, device>::Tile(const TileElementSize& size,
                            memory::MemoryView<ElementType, device>&& memory_view, SizeType ld) noexcept
    : size_(size), memory_view_(std::move(memory_view)), ld_(ld) {
  DLAF_ASSERT(size.isValid(), size);
  DLAF_ASSERT(ld_ >= std::max<SizeType>(1, size_.rows()), ld, size_.rows());
  DLAF_ASSERT(size.isEmpty() || size_.rows() + ld_ * (size_.cols() - 1) <= memory_view_.size(), size,
              size_, ld_, memory_view_.size());
}

template <class T, Device device>
Tile<const T, device>::Tile(Tile&& rhs) noexcept
    : size_(rhs.size_), memory_view_(std::move(rhs.memory_view_)), ld_(rhs.ld_), p_(std::move(rhs.p_)) {
  rhs.setDefaultSizes();
}

template <class T, Device device>
Tile<const T, device>::~Tile() {
  if (p_) {
    if (std::uncaught_exception())
      p_->set_exception(std::make_exception_ptr(ContinuationException{}));
    else
      p_->set_value(Tile<ElementType, device>(size_, std::move(memory_view_), ld_));
  }
}

template <class T, Device device>
Tile<const T, device>& Tile<const T, device>::operator=(Tile<const T, device>&& rhs) noexcept {
  size_ = rhs.size_;
  memory_view_ = std::move(rhs.memory_view_);
  ld_ = rhs.ld_;
  p_ = std::move(rhs.p_);
  rhs.setDefaultSizes();

  return *this;
}

template <class T, Device device>
void Tile<const T, device>::setDefaultSizes() noexcept {
  size_ = {0, 0};
  ld_ = 1;
}

}
}
