//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <exception>
#include <ostream>
#include <type_traits>

#include <hpx/functional.hpp>
#include <hpx/local/future.hpp>
#include <hpx/tuple.hpp>

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

namespace matrix {
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
    DLAF_ASSERT_HEAVY(index.isIn(size_), index, size_);
    return memory_view_(index.row() + static_cast<SizeType>(ld_) * index.col());
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
    DLAF_ASSERT_HEAVY(index.isIn(size_), index, size_);
    return memory_view_(index.row() + ld_ * index.col());
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
  return common::DataDescriptor<T>(tile.ptr({0, 0}), tile.size().cols(), tile.size().rows(), tile.ld());
}

namespace internal {
/// Gets the value from future<Tile>, and forwards all other types unchanged.
template <typename T>
struct UnwrapFuture {
  template <typename U>
  static decltype(auto) call(U&& u) {
    return std::forward<U>(u);
  }
};

template <typename T, Device D>
struct UnwrapFuture<hpx::future<Tile<T, D>>> {
  template <typename U>
  static auto call(U&& u) {
    auto t = u.get();
    return t;
  }
};

/// Callable object used for the unwrapExtendTiles function below.
template <typename F>
class UnwrapExtendTiles {
  template <typename... Ts>
  auto callHelper(std::true_type, Ts&&... ts) {
    // Extract values from futures (not shared_futures).
    auto t = hpx::make_tuple<>(UnwrapFuture<std::decay_t<Ts>>::call(std::forward<Ts>(ts))...);

    // Call f with all futures (not just future<Tile>) unwrapped.
    hpx::invoke_fused(hpx::util::unwrapping(f), t);

    // Finally, we extend the lifetime of read-write tiles directly and
    // read-only tiles wrapped in shared_futures by returning them here in a
    // tuple.
    return t;
  }

  template <typename... Ts>
  auto callHelper(std::false_type, Ts&&... ts) {
    // Extract values from futures (not shared_futures).
    auto t = hpx::make_tuple<>(UnwrapFuture<std::decay_t<Ts>>::call(std::forward<Ts>(ts))...);

    // Call f with all futures (not just future<Tile>) unwrapped.
    auto&& r = hpx::invoke_fused(hpx::util::unwrapping(f), t);

    // Finally, we extend the lifetime of read-write tiles directly and
    // read-only tiles wrapped in shared_futures by returning them here in a
    // tuple.
    return hpx::make_tuple<>(std::forward<decltype(r)>(r), std::move(t));
  }

public:
  template <typename F_,
            typename = std::enable_if_t<!std::is_same<UnwrapExtendTiles, std::decay_t<F_>>::value>>
  UnwrapExtendTiles(F_&& f_) : f(std::forward<F_>(f_)) {}
  UnwrapExtendTiles(UnwrapExtendTiles&&) = default;
  UnwrapExtendTiles& operator=(UnwrapExtendTiles&&) = default;
  UnwrapExtendTiles(UnwrapExtendTiles const&) = default;
  UnwrapExtendTiles& operator=(UnwrapExtendTiles const&) = default;

  // We use trailing decltype for SFINAE. This ensures that this does not
  // become a candidate when F is not callable with the given arguments.
  template <typename... Ts>
  auto operator()(Ts&&... ts)
      -> decltype(callHelper(std::is_void<decltype(hpx::invoke(hpx::util::unwrapping(std::declval<F>()),
                                                               std::declval<Ts>()...))>{},
                             std::forward<Ts>(ts)...)) {
    return callHelper(std::is_void<decltype(hpx::invoke(hpx::util::unwrapping(std::declval<F>()),
                                                        std::declval<Ts>()...))>{},
                      std::forward<Ts>(ts)...);
  }

private:
  F f;
};
}

/// Custom version of hpx::util::unwrapping for tile lifetime management.
///
/// Unwraps and forwards all arguments to the function f, but also returns all
/// arguments as they are with the exception of future<Tile> arguments.
/// future<Tile> arguments are returned unwrapped (as getting the value from the
/// future leaves the future empty).  The return value of f is ignored. This
/// wrapper is useful for extending the lifetimes of tiles with custom executors
/// such as the CUDA/cuBLAS executors, where f returns immediately, but the
/// tiles must be kept alive until the completion of the operation. The wrapper
/// can be used with "normal" blocking host-side operations as well.
///
/// The wrapper returns a tuple of the input arguments for void functions, and
/// a tuple of the result and a tuple of the input arguments for non-void
/// functions. getUnwrapReturnValue should be used to extract the return value of
/// the wrapped function.
template <typename F>
auto unwrapExtendTiles(F&& f) {
  return internal::UnwrapExtendTiles<std::decay_t<F>>{std::forward<F>(f)};
}

/// Access the return value of the function wrapped by unwrapExtendTiles.
///
/// Because of the lifetime management that uwnrapExtendTiles does it will
/// return a tuple where the first element is the return value of the wrapped
/// function and the second element contains the arguments that have their
/// lifetime extended. This helper function extracts the return value of the
/// wrapped function. When the return type of the wrapped function is void, this
/// also returns void.
template <typename... Ts>
void getUnwrapReturnValue(hpx::future<hpx::tuple<Ts...>>&&) {}

template <typename R, typename... Ts>
auto getUnwrapReturnValue(hpx::future<hpx::tuple<R, hpx::tuple<Ts...>>>&& f) {
  auto split_f = hpx::split_future(std::move(f));
  return std::move(hpx::get<0>(split_f));
}

/// ---- ETI

#define DLAF_TILE_ETI(KWORD, DATATYPE, DEVICE) \
  KWORD template class Tile<DATATYPE, DEVICE>; \
  KWORD template class Tile<const DATATYPE, DEVICE>;

DLAF_TILE_ETI(extern, float, Device::CPU)
DLAF_TILE_ETI(extern, double, Device::CPU)
DLAF_TILE_ETI(extern, std::complex<float>, Device::CPU)
DLAF_TILE_ETI(extern, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_CUDA)
DLAF_TILE_ETI(extern, float, Device::GPU)
DLAF_TILE_ETI(extern, double, Device::GPU)
DLAF_TILE_ETI(extern, std::complex<float>, Device::GPU)
DLAF_TILE_ETI(extern, std::complex<double>, Device::GPU)
#endif
}
}

#include <dlaf/matrix/tile.tpp>
