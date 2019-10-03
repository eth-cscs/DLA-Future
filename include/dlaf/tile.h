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

#include <hpx/hpx.hpp>
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"

namespace dlaf {

/// @brief The Tile object aims to provide an effective way to access the memory as a two dimensional
/// array. It does not allocate any memory, but it references the memory given by a @c MemoryView object.
/// It represents the building block of the Matrix object and of linear algebra algorithms.
///
/// Two levels of constness exist for @c Tile analogously to pointer semantics:
/// the constness of the tile and the constness of the data referenced by the tile.
/// Implicit conversion is allowed from tiles of non-const elements to tiles of const elements.
///
/// Note: The constructor of tiles of const elements, requires a MemoryView of non-const memory, however
/// the tile of const elements ensure that the memory will not be modified.
template <class T, Device device>
class Tile {
public:
  using ElementType = std::remove_const_t<T>;
  friend Tile<const ElementType, device>;

  /// @brief Constructs a (@p size.rows() x @p size.cols()) Tile.
  /// @throw std::invalid_argument if @p size.row() < 0, @p size.cols() < 0 or @p ld < max(1, @p size.rows()).
  /// @throw std::invalid_argument if memory_view does not contain enough elements.
  /// The (i, j)-th element of the Tile is stored in the (i+ld*j)-th element of memory_view.
  Tile(TileElementSize size, memory::MemoryView<ElementType, device> memory_view, SizeType ld);

  Tile(const Tile&) = delete;

  Tile(Tile&& rhs) noexcept;

  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  Tile(Tile<ElementType, device>&& rhs) noexcept;

  /// @brief Destroys the Tile.
  /// If a promise was set using @c setPromise its value is set to a Tile
  /// which has the same size and which references the same memory as @p *this.
  ~Tile();

  Tile& operator=(const Tile&) = delete;

  Tile& operator=(Tile&& rhs) noexcept;

  template <class U = T,
            class = typename std::enable_if_t<std::is_const<U>::value && std::is_same<T, U>::value>>
  Tile& operator=(Tile<ElementType, device>&& rhs) noexcept;

  /// @brief Returns the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(size()) == true.
  T& operator()(TileElementIndex index) const noexcept {
    return *ptr(index);
  }

  /// @brief Returns the pointer to the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  /// @pre index.isValid() == true.
  /// @pre index.isIn(size()) == true.
  T* ptr(TileElementIndex index) const noexcept {
    using util::size_t::sum;
    using util::size_t::mul;
    assert(index.isValid());
    assert(index.isIn(size_));

    return memory_view_(sum(index.row(), mul(ld_, index.col())));
  }

  /// @brief Returns the size of the Tile.
  TileElementSize size() const noexcept {
    return size_;
  }
  /// @brief Returns the leading dimension.
  SizeType ld() const noexcept {
    return ld_;
  }

  /// @brief Sets the promise to which this Tile will be moved on destruction.
  /// @c setPromise can be called only once per object.
  /// @throw std::logic_error if @c setPromise was already called.
  template <class U = T>
  std::enable_if_t<!std::is_const<U>::value && std::is_same<T, U>::value, Tile&> setPromise(
      hpx::promise<Tile<T, device>>&& p) {
    if (p_)
      throw std::logic_error("setPromise has been already used on this object!");
    p_ = std::make_unique<hpx::promise<Tile<T, device>>>(std::move(p));
    return *this;
  }

private:
  TileElementSize size_;
  memory::MemoryView<ElementType, device> memory_view_;
  SizeType ld_;

  std::unique_ptr<hpx::promise<Tile<ElementType, device>>> p_;
};

#include <dlaf/tile.ipp>
}
