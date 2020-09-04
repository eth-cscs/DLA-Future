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

#include <exception>
#include <ostream>

#include <hpx/local/future.hpp>

#include "dlaf/common/data_descriptor.h"
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"

namespace dlaf {

/// Exception used to notify a continuation task that an exception has been thrown in a dependency task.
///
/// It is mainly used to enable exception propagation in the automatic-continuation mechanism.
struct ContinuationException final : public std::runtime_error {
  ContinuationException()
      : std::runtime_error("An exception has been thrown during the execution of the previous task.") {}
};

template <class T, Device device>
class Tile;

template <class T, Device device>
class Tile<const T, device>;

/// The Tile object aims to provide an effective way to access the memory as a two dimensional
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
class Tile<const T, device> {
  friend Tile<T, device>;

  template <class PT>
  using promise_t = hpx::lcos::local::promise<PT>;

public:
  using ElementType = T;

  /// Constructs a (@p size.rows() x @p size.cols()) Tile.
  ///
  /// @pre size.isValid(),
  /// @pre ld >= max(1, @p size.rows()),
  /// @pre memory_view contains enough elements.
  /// The (i, j)-th element of the Tile is stored in the (i+ld*j)-th element of memory_view.
  Tile(const TileElementSize& size, memory::MemoryView<ElementType, device>&& memory_view,
       SizeType ld) noexcept;

  Tile(const Tile&) = delete;

  Tile(Tile&& rhs) noexcept;

  /// Destroys the Tile.
  ///
  /// If a promise was set using @c setPromise its value is set to a Tile
  /// which has the same size and which references the same memory as @p *this.
  ~Tile();

  Tile& operator=(const Tile&) = delete;

  Tile& operator=(Tile&& rhs) noexcept;

  /// Returns the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  const T& operator()(const TileElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Returns the base pointer.
  const T* ptr() const noexcept {
    return memory_view_();
  }

  /// Returns the pointer to the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  const T* ptr(const TileElementIndex& index) const noexcept {
    using util::size_t::sum;
    using util::size_t::mul;
    DLAF_ASSERT_HEAVY(index.isIn(size_), "");

    return memory_view_(sum(index.row(), mul(ld_, index.col())));
  }

  /// Returns the size of the Tile.
  const TileElementSize& size() const noexcept {
    return size_;
  }
  /// Returns the leading dimension.
  SizeType ld() const noexcept {
    return ld_;
  }

  /// Prints information about the tile.
  friend std::ostream& operator<<(std::ostream& out, const Tile& tile) {
    return out << "size=" << tile.size() << ", ld=" << tile.ld();
  }

private:
  /// Sets size to {0, 0} and ld to 1.
  void setDefaultSizes() noexcept;

  TileElementSize size_;
  memory::MemoryView<ElementType, device> memory_view_;
  SizeType ld_;

  std::unique_ptr<promise_t<Tile<ElementType, device>>> p_;
};

template <class T, Device device>
class Tile : public Tile<const T, device> {
  template <class PT>
  using promise_t = hpx::lcos::local::promise<PT>;

  friend Tile<const T, device>;

public:
  using ElementType = T;

  /// Constructs a (@p size.rows() x @p size.cols()) Tile.
  ///
  /// @pre size.isValid(),
  /// @pre ld >= max(1, @p size.rows()),
  /// @pre memory_view contains enough elements.
  /// The (i, j)-th element of the Tile is stored in the (i+ld*j)-th element of memory_view.
  Tile(const TileElementSize& size, memory::MemoryView<ElementType, device>&& memory_view,
       SizeType ld) noexcept
      : Tile<const T, device>(size, std::move(memory_view), ld) {}

  Tile(const Tile&) = delete;

  Tile(Tile&& rhs) = default;

  Tile& operator=(const Tile&) = delete;

  Tile& operator=(Tile&& rhs) = default;

  /// Returns the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  T& operator()(const TileElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Returns the base pointer.
  T* ptr() const noexcept {
    return memory_view_();
  }

  /// Returns the pointer to the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  T* ptr(const TileElementIndex& index) const noexcept {
    using util::size_t::sum;
    using util::size_t::mul;
    DLAF_ASSERT_HEAVY(index.isIn(size_), "");

    return memory_view_(sum(index.row(), mul(ld_, index.col())));
  }

  /// Sets the promise to which this Tile will be moved on destruction.
  ///
  /// @c setPromise can be called only once per object.
  Tile& setPromise(promise_t<Tile<T, device>>&& p) {
    DLAF_ASSERT(!p_, "setPromise has been already used on this object!");
    p_ = std::make_unique<promise_t<Tile<T, device>>>(std::move(p));
    return *this;
  }

private:
  using Tile<const T, device>::size_;
  using Tile<const T, device>::memory_view_;
  using Tile<const T, device>::ld_;
  using Tile<const T, device>::p_;
};

/// Create a common::Buffer from a Tile.
template <class T, Device device>
auto create_data(const Tile<T, device>& tile) {
  return common::DataDescriptor<T>(tile.ptr({0, 0}), to_sizet(tile.size().cols()),
                                   to_sizet(tile.size().rows()), to_sizet(tile.ld()));
}

#include <dlaf/tile.tpp>

/// ---- ETI

#define DLAF_TILE_ETI(KWORD, DATATYPE, DEVICE) \
  KWORD template class Tile<DATATYPE, DEVICE>; \
  KWORD template class Tile<const DATATYPE, DEVICE>;

DLAF_TILE_ETI(extern, float, Device::CPU)
DLAF_TILE_ETI(extern, double, Device::CPU)
DLAF_TILE_ETI(extern, std::complex<float>, Device::CPU)
DLAF_TILE_ETI(extern, std::complex<double>, Device::CPU)

// DLAF_TILE_ETI(extern, float, Device::GPU)
// DLAF_TILE_ETI(extern, double, Device::GPU)
// DLAF_TILE_ETI(extern, std::complex<float>, Device::GPU)
// DLAF_TILE_ETI(extern, std::complex<double>, Device::GPU)

}
