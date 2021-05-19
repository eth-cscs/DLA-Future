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
/// Contains the information to create a subtile.
struct SubTileSpec {
  TileElementIndex origin;
  TileElementSize size;
};

// forward declarations
template <class T, Device device>
class Tile;

template <class T, Device device>
class Tile<const T, device>;

namespace internal {
template <class T, Device D>
hpx::shared_future<Tile<T, D>> splitTileInsertFutureInChain(hpx::future<Tile<T, D>>& tile);

template <class T, Device D>
hpx::future<Tile<T, D>> createSubTile(const hpx::shared_future<Tile<T, D>>& tile,
                                      const SubTileSpec& spec);
}

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
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;

  template <class PT>
  using promise_t = hpx::lcos::local::promise<PT>;

  friend TileType;
  friend hpx::future<Tile<const T, device>> internal::createSubTile<>(
      const hpx::shared_future<Tile<const T, device>>& tile, const SubTileSpec& spec);

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
    return memory_view_(linearIndex(index));
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
  void setDefaultSize() noexcept {
    size_ = {0, 0};
    ld_ = 1;
  }

  SizeType linearIndex(const TileElementIndex& index) const noexcept {
    DLAF_ASSERT_HEAVY(index.isIn(size_), index, size_);
    return index.row() + ld_ * index.col();
  }

  static SizeType linearSize(const TileElementSize& size, SizeType ld) noexcept {
    if (size.isEmpty())
      return 0;
    return size.rows() + ld * (size.cols() - 1);
  }

  // Creates an untracked subtile.
  // Dependencies are not influenced by the new created object therefore race-conditions
  // might happen if used improperly.
  Tile(const Tile& tile, const SubTileSpec& spec) noexcept;

  // Creates a read-only subtile keeping the dependencies.
  // It calls tile.get(), therefore it should be used when tile is guaranteed to be ready:
  // e.g. in dataflow, .then, ...
  Tile(hpx::shared_future<ConstTileType> tile, const SubTileSpec& spec)
      : Tile<const T, device>(tile.get(), spec) {
    sfc_ = std::move(tile);
  }

  TileElementSize size_;
  memory::MemoryView<ElementType, device> memory_view_;
  SizeType ld_;

  // With C++17 the following objects can be joined in a variant.
  std::unique_ptr<promise_t<TileType>> p_;
  hpx::shared_future<TileType> sf_;
  hpx::shared_future<ConstTileType> sfc_;
};

template <class T, Device device>
Tile<const T, device>::Tile(const TileElementSize& size,
                            memory::MemoryView<ElementType, device>&& memory_view, SizeType ld) noexcept
    : size_(size), memory_view_(std::move(memory_view)), ld_(ld) {
  DLAF_ASSERT(size_.isValid(), size_);
  DLAF_ASSERT(ld_ >= std::max<SizeType>(1, size_.rows()), ld, size_.rows());
  DLAF_ASSERT(size_.isEmpty() || linearSize(size_, ld_) <= memory_view_.size(), size_, ld_,
              memory_view_.size());
}

template <class T, Device device>
Tile<const T, device>::Tile(Tile&& rhs) noexcept
    : size_(rhs.size_), memory_view_(std::move(rhs.memory_view_)), ld_(rhs.ld_), p_(std::move(rhs.p_)),
      sf_(std::move(rhs.sf_)), sfc_(std::move(rhs.sfc_)) {
  rhs.setDefaultSize();
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
  sf_ = std::move(rhs.sf_);
  sfc_ = std::move(rhs.sfc_);
  rhs.setDefaultSize();

  return *this;
}

template <class T, Device device>
Tile<const T, device>::Tile(const Tile<const T, device>& tile, const SubTileSpec& spec) noexcept
    : Tile<const T, device>(  //
          spec.size,
          memory::MemoryView<T, device>(tile.memory_view_,
                                        spec.size.isEmpty() ? 0 : tile.linearIndex(spec.origin),
                                        tile.linearSize(spec.size, tile.ld())),
          tile.ld()) {
  DLAF_ASSERT(spec.origin.isValid(), spec.origin);
  DLAF_ASSERT(spec.origin.isInOrOn(tile.size()), spec.origin, tile.size());
  DLAF_ASSERT(spec.size.isValid(), spec.size);
  DLAF_ASSERT((spec.origin + spec.size).isInOrOn(tile.size_), spec.origin, spec.size, tile.size_);
}

template <class T, Device device>
class Tile : public Tile<const T, device> {
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;

  template <class PT>
  using promise_t = hpx::lcos::local::promise<PT>;

  friend ConstTileType;
  friend hpx::future<Tile<T, device>> internal::createSubTile<>(
      const hpx::shared_future<Tile<T, device>>& tile, const SubTileSpec& spec);
  friend hpx::shared_future<Tile<T, device>> internal::splitTileInsertFutureInChain<>(
      hpx::future<Tile<T, device>>& tile);

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
    return memory_view_(linearIndex(index));
  }

  /// Sets the promise to which this Tile will be moved on destruction.
  ///
  /// @c setPromise can be called only once per object.
  /// @pre this Tile should not be a subtile.
  Tile& setPromise(promise_t<TileType>&& p) {
    DLAF_ASSERT(!p_, "setPromise has been already used on this object!");
    DLAF_ASSERT_HEAVY(!sf_.valid(), "setPromise cannot be used on subtiles!");
    DLAF_ASSERT_HEAVY(!sfc_.valid(), "setPromise cannot be used on subtiles!");
    p_ = std::make_unique<promise_t<TileType>>(std::move(p));
    return *this;
  }

private:
  // Creates a writable subtile keeping the dependencies.
  // It calls old_tile.get(), therefore it should be used when old_tile is guaranteed to be ready:
  // e.g. in dataflow, .then, ...
  Tile(hpx::shared_future<TileType> tile, const SubTileSpec& spec) : ConstTileType(tile.get(), spec) {
    sf_ = std::move(tile);
  }

  using ConstTileType::linearIndex;
  using ConstTileType::size_;
  using ConstTileType::memory_view_;
  using ConstTileType::ld_;
  using ConstTileType::p_;
  using ConstTileType::sf_;
  using ConstTileType::sfc_;
};

/// Create a common::Buffer from a Tile.
template <class T, Device device>
auto create_data(const Tile<T, device>& tile) {
  return common::DataDescriptor<T>(tile.ptr({0, 0}), tile.size().cols(), tile.size().rows(), tile.ld());
}

namespace internal {
template <class T, Device D>
hpx::future<Tile<T, D>> createSubTile(const hpx::shared_future<Tile<T, D>>& tile,
                                      const SubTileSpec& spec) {
  return hpx::dataflow(
      hpx::launch::sync, [](auto tile, auto spec) { return Tile<T, D>(tile, spec); }, tile, spec);
}

template <class T, Device D>
hpx::shared_future<Tile<T, D>> splitTileInsertFutureInChain(hpx::future<Tile<T, D>>& tile) {
  // Insert a Tile in the tile dependency chains. 3 different cases are supported:
  // 1)  F1(P2)  F2(P3) ...      =>  F1(PN)  FN(P2)  F2(P3) ...
  // 2)  F1(SF(P2))  F2(P3) ...  =>  F1(PN)  FN(SF(P2))  F2(P3) ...
  // 3)  F1()                    =>  F1(PN)  FN()
  // where Pi, Fi is a promise future pair (Pi sets Fi),
  // F1(P2) means that the Tile in F1 will set promise P2.
  // and F1(SF(P2)) means that the shared future which will set promise P2 will be released.
  // On input tile is F1(*), on output tile is FN(*).
  // The shared future F1(PN) is returned and will be used to create subtiles.
  using hpx::lcos::local::promise;

  // 1. Create a new promise + tile pair PN, FN
  promise<Tile<T, D>> p;
  auto tmp_tile = p.get_future();
  // 2. Break the dependency chain inserting PN and storing P2 or SF(P2):  F1(PN)  FN()  F2(P3)
  auto swap_promise = [promise = std::move(p)](auto tile) mutable {
    // sfc_ should not be valid here, as it should be set only for const Tiles.
    DLAF_ASSERT_HEAVY(!tile.sfc_.valid(), "Internal Dependency Error");
    // Similarly p_ and sf_ should not be set at the same time.
    DLAF_ASSERT_HEAVY(!(tile.p_ && tile.sf_.valid()), "Internal Dependency Error");

    auto p = std::move(tile.p_);
    auto sf = std::move(tile.sf_);

    tile.setPromise(std::move(promise));
    // Note: C++17 std::variant can be used.
    return hpx::make_tuple(std::move(tile), std::make_tuple(std::move(p), std::move(sf)));
  };
  auto tmp =
      hpx::split_future(tile.then(hpx::launch::sync, hpx::util::unwrapping(std::move(swap_promise))));
  // old_tile = F1(PN) and will be used to create the subtiles
  hpx::shared_future<Tile<T, D>> old_tile = std::move(hpx::get<0>(tmp));
  // 3. Set P2 or SF(P2) into FN to restore the chain:  F1(PN)  FN(*) ...
  auto set_promise_or_shfuture = [](auto tile, auto p_sf_tuple) {
    auto& p = std::get<0>(p_sf_tuple);
    auto& sf = std::get<1>(p_sf_tuple);
    if (p)
      tile.setPromise(std::move(*p));
    else if (sf.valid())
      tile.sf_ = std::move(sf);

    return tile;
  };
  // tile = FN(*) (out argument) can be used to access the full tile after the subtiles tasks completed.
  tile = hpx::dataflow(hpx::launch::sync, hpx::util::unwrapping(set_promise_or_shfuture), tmp_tile,
                       std::move(hpx::get<1>(tmp)));

  return old_tile;
}
}

/// Create a read-only subtile of a given read-only tile.
///
/// The returned subtile will get ready, at the same time as @p tile gets ready.
/// The next dependency in the dependency chain will become ready only when @p tile
/// and the returned subtile go out of scope.
template <class T, Device D>
hpx::shared_future<Tile<const T, D>> splitTile(const hpx::shared_future<Tile<const T, D>>& tile,
                                               const SubTileSpec& spec) {
  return internal::createSubTile(tile, spec);
}

/// Create read-only subtiles of a given read-only tile.
///
/// The returned subtiles will get ready, at the same time as @p tile gets ready.
/// The next dependency in the dependency chain will become ready only when @p tile
/// and all the returned subtiles go out of scope.
template <class T, Device D>
std::vector<hpx::shared_future<Tile<const T, D>>> splitTile(
    const hpx::shared_future<Tile<const T, D>>& tile, const std::vector<SubTileSpec>& specs) {
  std::vector<hpx::shared_future<Tile<const T, D>>> ret;
  ret.reserve(specs.size());
  for (const auto& spec : specs) {
    ret.emplace_back(internal::createSubTile(tile, spec));
  }

  return ret;
}

/// Create a writeable subtile of a given tile.
///
/// The returned subtile will get ready, when the original tile was supposed to get ready.
/// @p tile is replaced with the (full) tile which will get ready when the subtile goes out of scope.
/// The next dependency in the dependency chain will become ready only when @p tile goes out of scope.
template <class T, Device D>
hpx::future<Tile<T, D>> splitTile(hpx::future<Tile<T, D>>& tile, const SubTileSpec& spec) {
  auto old_tile = internal::splitTileInsertFutureInChain(tile);
  // tile is now the new element of the dependency chain which will be ready
  // when the subtile will go out of scope.

  return internal::createSubTile(old_tile, spec);
}

/// Create a writeable subtile of a given tile.
///
/// All the returned subtiles will get ready, when the original tile was supposed to get ready
/// (therefore the returned subtiles should be disjoint, otherwise race conditions might occour).
/// @p tile is replaced with the (full) tile which will get ready when the all the subtiles go out of scope.
/// The next dependency in the dependency chain will become ready only when @p tile goes out of scope.
/// @pre The subtiles described with specs should be disjoint
///      (i.e. two different subtile cannot access the same element).
template <class T, Device D>
std::vector<hpx::future<Tile<T, D>>> splitTileDisjoint(hpx::future<Tile<T, D>>& tile,
                                                       const std::vector<SubTileSpec>& specs) {
  if (specs.size() == 0)
    return {};

#ifdef DLAF_ASSERT_MODERATE_ENABLE
  // Check if subtiles overlap.
  auto overlap = [](const auto& spec1, const auto& spec2) {
    // no overlap if either of the sizes is empty.
    if (spec1.size.isEmpty() || spec2.size.isEmpty())
      return false;

    const auto& start1 = spec1.origin;
    const auto end1 = start1 + spec1.size;
    const auto& start2 = spec2.origin;
    const auto end2 = start2 + spec2.size;

    // no overlap if rows do not overlap.
    if (end1.row() <= start2.row() || end2.row() <= start1.row())
      return false;
    // no overlap if cols do not overlap.
    if (end1.col() <= start2.col() || end2.col() <= start1.col())
      return false;

    return true;
  };
  for (auto it = std::cbegin(specs); it < std::cend(specs); ++it) {
    // no overlap possible if size is empty.
    if (it->size.isEmpty())
      continue;

    for (auto it2 = std::cbegin(specs); it2 < it; ++it2) {
      DLAF_ASSERT_MODERATE(!overlap(*it, *it2), it->origin, it->size, it2->origin, it2->size);
    }
  }
#endif

  auto old_tile = internal::splitTileInsertFutureInChain(tile);
  // tile is now the new element of the dependency chain which will be ready
  // when all subtiles will go out of scope.

  std::vector<hpx::future<Tile<T, D>>> ret;
  ret.reserve(specs.size());
  for (const auto& spec : specs) {
    ret.emplace_back(internal::createSubTile(old_tile, spec));
  }

  return ret;
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

template <class R, class... Ts>
auto getUnwrapRetValAndArgs(hpx::future<hpx::tuple<R, hpx::tuple<Ts...>>>&& f) {
  auto wrapped_res = hpx::split_future(std::move(f));
  auto ret_value = std::move(hpx::get<0>(wrapped_res));
  auto args = hpx::split_future(std::move(hpx::get<1>(wrapped_res)));
  return hpx::make_tuple(std::move(ret_value), std::move(args));
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
