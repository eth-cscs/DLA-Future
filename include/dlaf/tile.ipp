//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

template <class T, Device device>
Tile<T, device>::Tile(TileElementSize size, memory::MemoryView<ElementType, device> memory_view,
                      SizeType ld)
    : size_(size), memory_view_(memory_view), ld_(ld) {
  using util::size_t::sum;
  using util::size_t::mul;
  if (size_.rows() < 0 || size_.cols() < 0)
    throw std::invalid_argument("Error: Invalid Tile sizes");
  if (ld_ < size_.rows() || ld_ < 1)
    throw std::invalid_argument("Error: Invalid Tile leading dimension");
  if (sum(size_.rows(), mul(ld_, (size_.cols() - 1))) > memory_view_.size())
    throw std::invalid_argument("Error: Tile exceeds the MemoryView limits");
}

template <class T, Device device>
Tile<T, device>::Tile(Tile&& rhs) noexcept
    : size_(rhs.size_), memory_view_(std::move(rhs.memory_view_)), ld_(rhs.ld_), p_(std::move(rhs.p_)) {
  rhs.size_ = {0, 0};
  rhs.ld_ = 1;
}

template <class T, Device device>
template <class U, class>
Tile<T, device>::Tile(Tile<ElementType, device>&& rhs) noexcept
    : size_(rhs.size_), memory_view_(std::move(rhs.memory_view_)), ld_(rhs.ld_), p_(std::move(rhs.p_)) {
  rhs.size_ = {0, 0};
  rhs.ld_ = 1;
}

template <class T, Device device>
Tile<T, device>::~Tile() {
  if (p_) {
    p_->set_value(Tile<ElementType, device>(size_, memory_view_, ld_));
  }
}

template <class T, Device device>
Tile<T, device>& Tile<T, device>::operator=(Tile<T, device>&& rhs) noexcept {
  size_ = rhs.size_;
  memory_view_ = std::move(rhs.memory_view_);
  ld_ = rhs.ld_;
  p_ = std::move(rhs.p_);
  rhs.size_ = {0, 0};
  rhs.ld_ = 1;

  return *this;
}

template <class T, Device device>
template <class U, class>
Tile<T, device>& Tile<T, device>::operator=(Tile<Tile<T, device>::ElementType, device>&& rhs) noexcept {
  size_ = rhs.size_;
  memory_view_ = std::move(rhs.memory_view_);
  ld_ = rhs.ld_;
  p_ = std::move(rhs.p_);
  rhs.size_ = {0, 0};
  rhs.ld_ = 1;

  return *this;
}
