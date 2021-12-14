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

#include <hpx/local/functional.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/tuple.hpp>
#include <hpx/local/unwrap.hpp>

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
namespace internal {

template <class T, Device device>
class TileData {
public:
  TileData(const TileElementSize& size, memory::MemoryView<T, device>&& memory_view,
           SizeType ld) noexcept
      : size_(size), memory_view_(std::move(memory_view)), ld_(ld) {
    DLAF_ASSERT(size_.isValid(), size_);
    DLAF_ASSERT(ld_ >= std::max<SizeType>(1, size_.rows()), ld, size_.rows());
    DLAF_ASSERT(size_.isEmpty() || linearSize(size_, ld_) <= memory_view_.size(), size_, ld_,
                memory_view_.size());
  }

  TileData(const TileData& rhs) = delete;

  TileData(TileData&& rhs) noexcept
      : size_(rhs.size_), memory_view_(std::move(rhs.memory_view_)), ld_(rhs.ld_) {
    rhs.setDefaultSize();
  }

  TileData& operator=(const TileData&) = delete;

  TileData& operator=(TileData&& rhs) {
    size_ = rhs.size_;
    memory_view_ = std::move(rhs.memory_view_);
    ld_ = rhs.ld_;
    rhs.setDefaultSize();

    return *this;
  }

  const auto& memoryView() const noexcept {
    return memory_view_;
  }

  T* ptr() const noexcept {
    return memory_view_();
  }

  T* ptr(const TileElementIndex& index) const noexcept {
    return memory_view_(linearIndex(index));
  }

  const TileElementSize& size() const noexcept {
    return size_;
  }
  SizeType ld() const noexcept {
    return ld_;
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

private:
  /// Sets size to {0, 0} and ld to 1.
  void setDefaultSize() noexcept {
    size_ = {0, 0};
    ld_ = 1;
  }

  TileElementSize size_;
  memory::MemoryView<T, device> memory_view_;
  SizeType ld_;
};
}

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
public:
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;
  using TileDataType = internal::TileData<T, device>;
  using TilePromise = hpx::lcos::local::promise<TileDataType>;

  friend TileType;
  friend hpx::future<Tile<const T, device>> internal::createSubTile<>(
      const hpx::shared_future<Tile<const T, device>>& tile, const SubTileSpec& spec);

  using ElementType = T;

  /// Constructs a (@p size.rows() x @p size.cols()) Tile.
  ///
  /// @pre size.isValid(),
  /// @pre ld >= max(1, @p size.rows()),
  /// @pre memory_view contains enough elements.
  /// The (i, j)-th element of the Tile is stored in the (i+ld*j)-th element of memory_view.
  Tile(const TileElementSize& size, memory::MemoryView<ElementType, device>&& memory_view,
       SizeType ld) noexcept
      : data_(size, std::move(memory_view), ld) {}

  Tile(TileDataType&& data) noexcept : data_(std::move(data)) {}

  Tile(const Tile&) = delete;

  Tile(Tile&& rhs) noexcept : data_(std::move(rhs.data_)), dep_tracker_(std::move(rhs.dep_tracker_)) {
    rhs.dep_tracker_ = std::monostate();
  };

  /// Destroys the Tile.
  ///
  /// If a promise was set using @c setPromise its value is set to a Tile
  /// which has the same size and which references the same memory as @p *this.
  ~Tile();

  Tile& operator=(const Tile&) = delete;

  Tile& operator=(Tile&& rhs) noexcept {
    data_ = std::move(rhs.data_);
    dep_tracker_ = std::move(rhs.dep_tracker_);
    rhs.dep_tracker_ = std::monostate();

    return *this;
  }

  /// Returns the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  const T& operator()(const TileElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Returns the base pointer.
  const T* ptr() const noexcept {
    return data_.ptr();
  }

  /// Returns the pointer to the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  const T* ptr(const TileElementIndex& index) const noexcept {
    return data_.ptr(index);
  }

  /// Returns the size of the Tile.
  const TileElementSize& size() const noexcept {
    return data_.size();
  }
  /// Returns the leading dimension.
  SizeType ld() const noexcept {
    return data_.ld();
  }

  /// Prints information about the tile.
  friend std::ostream& operator<<(std::ostream& out, const Tile& tile) {
    return out << "size=" << tile.size() << ", ld=" << tile.ld();
  }

private:
  static memory::MemoryView<T, device> createMemoryViewForSubtile(const Tile<const T, device>& tile,
                                                                  const SubTileSpec& spec) {
    DLAF_ASSERT(spec.origin.isValid(), spec.origin);
    DLAF_ASSERT(spec.origin.isInOrOn(tile.size()), spec.origin, tile.size());
    DLAF_ASSERT(spec.size.isValid(), spec.size);
    DLAF_ASSERT((spec.origin + spec.size).isInOrOn(tile.size()), spec.origin, spec.size, tile.size());

    return memory::MemoryView<T, device>(tile.data_.memoryView(),
                                         spec.size.isEmpty() ? 0 : tile.data_.linearIndex(spec.origin),
                                         tile.data_.linearSize(spec.size, tile.ld()));
  };

  // Creates an untracked subtile.
  // Dependencies are not influenced by the new created object therefore race-conditions
  // might happen if used improperly.
  Tile(const Tile& tile, const SubTileSpec& spec) noexcept;

  // Creates a read-only subtile keeping the dependencies.
  // It calls tile.get(), therefore it should be used when tile is guaranteed to be ready:
  // e.g. in dataflow, .then, ...
  Tile(hpx::shared_future<ConstTileType> tile, const SubTileSpec& spec)
      : Tile<const T, device>(tile.get(), spec) {
    dep_tracker_ = std::move(tile);
  }

  TileDataType data_;
  std::variant<std::monostate, TilePromise, hpx::shared_future<TileType>,
               hpx::shared_future<ConstTileType>>
      dep_tracker_;
};

template <class T, Device device>
Tile<const T, device>::~Tile() {
  if (std::holds_alternative<TilePromise>(dep_tracker_)) {
    auto& p_ = std::get<TilePromise>(dep_tracker_);
    if (std::uncaught_exceptions() > 0)
      p_.set_exception(std::make_exception_ptr(ContinuationException{}));
    else
      p_.set_value(std::move(this->data_));
  }
}

template <class T, Device device>
Tile<const T, device>::Tile(const Tile<const T, device>& tile, const SubTileSpec& spec) noexcept
    : Tile<const T, device>(spec.size, Tile::createMemoryViewForSubtile(tile, spec), tile.ld()) {}

template <class T, Device device>
class Tile : public Tile<const T, device> {
public:
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;
  using TileDataType = internal::TileData<T, device>;
  using TilePromise = hpx::lcos::local::promise<TileDataType>;

  friend ConstTileType;
  friend hpx::future<Tile<T, device>> internal::createSubTile<>(
      const hpx::shared_future<Tile<T, device>>& tile, const SubTileSpec& spec);
  friend hpx::shared_future<Tile<T, device>> internal::splitTileInsertFutureInChain<>(
      hpx::future<Tile<T, device>>& tile);

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

  Tile(TileDataType&& data) noexcept : ConstTileType(std::move(data)) {}

  Tile(const Tile&) = delete;
  Tile(Tile&& rhs) = default;

  Tile& operator=(const Tile&) = delete;
  Tile& operator=(Tile&&) = default;

  /// Returns the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  T& operator()(const TileElementIndex& index) const noexcept {
    return *ptr(index);
  }

  /// Returns the base pointer.
  T* ptr() const noexcept {
    return data_.ptr();
  }

  /// Returns the pointer to the (i, j)-th element,
  /// where @p i := @p index.row and @p j := @p index.col.
  ///
  /// @pre index.isIn(size()).
  T* ptr(const TileElementIndex& index) const noexcept {
    return data_.ptr(index);
  }

  /// Sets the promise to which this Tile will be moved on destruction.
  ///
  /// @c setPromise can be called only once per object.
  /// @pre this Tile should not be a subtile.
  Tile& setPromise(TilePromise&& p) {
    DLAF_ASSERT(!std::holds_alternative<TilePromise>(dep_tracker_),
                "setPromise has been already used on this object!");
    DLAF_ASSERT(std::holds_alternative<std::monostate>(dep_tracker_),
                "setPromise cannot be used on subtiles!");
    dep_tracker_ = std::move(p);
    return *this;
  }

private:
  // Creates a writable subtile keeping the dependencies.
  // It calls old_tile.get(), therefore it should be used when old_tile is guaranteed to be ready:
  // e.g. in dataflow, .then, ...
  Tile(hpx::shared_future<TileType> tile, const SubTileSpec& spec) : ConstTileType(tile.get(), spec) {
    dep_tracker_ = std::move(tile);
  }

  using ConstTileType::data_;
  using ConstTileType::dep_tracker_;
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
  using TileType = Tile<T, D>;
  using PromiseType = hpx::lcos::local::promise<typename TileType::TileDataType>;

  // 1. Create a new promise + tile pair PN, FN
  PromiseType p;
  auto tmp_tile = p.get_future();
  // 2. Break the dependency chain inserting PN and storing P2 or SF(P2):  F1(PN)  FN()  F2(P3)
  auto swap_promise = [promise = std::move(p)](auto tile) mutable {
    // The dep_tracker cannot be a const Tile (can happen only for const Tiles).
    DLAF_ASSERT_HEAVY(!std::holds_alternative < hpx::shared_future<Tile<const T, D>>(tile.dep_tracker_),
                      "Internal Dependency Error");

    auto dep_tracker = std::move(tile.dep_tracker_);
    tile.dep_tracker_ = std::move(promise);

    return hpx::make_tuple(std::move(tile), std::move(dep_tracker));
  };
  auto tmp = hpx::split_future(tile.then(hpx::launch::sync, hpx::unwrapping(std::move(swap_promise))));
  // old_tile = F1(PN) and will be used to create the subtiles
  hpx::shared_future<TileType> old_tile = std::move(hpx::get<0>(tmp));
  // 3. Set P2 or SF(P2) into FN to restore the chain:  F1(PN)  FN(*) ...
  auto set_promise_or_shfuture = [](auto tile_data, auto dep_tracker) {
    TileType tile(std::move(tile_data));
    tile.dep_tracker_ = std::move(dep_tracker);
    return tile;
  };
  // tile = FN(*) (out argument) can be used to access the full tile after the subtiles tasks completed.
  tile = hpx::dataflow(hpx::launch::sync, hpx::unwrapping(set_promise_or_shfuture), tmp_tile,
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
  static auto call(U u) {
    return u.get();
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
    hpx::invoke_fused(hpx::unwrapping(f), t);

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
    auto&& r = hpx::invoke_fused(hpx::unwrapping(f), t);

    // Finally, we extend the lifetime of read-write tiles directly and
    // read-only tiles wrapped in shared_futures by returning them here in a
    // tuple.
    return hpx::make_tuple<>(std::forward<decltype(r)>(r), std::move(t));
  }

public:
  template <typename F_,
            typename = std::enable_if_t<!std::is_same_v<UnwrapExtendTiles, std::decay_t<F_>>>>
  UnwrapExtendTiles(F_&& f_) : f(std::forward<F_>(f_)) {}
  UnwrapExtendTiles(UnwrapExtendTiles&&) = default;
  UnwrapExtendTiles& operator=(UnwrapExtendTiles&&) = default;
  UnwrapExtendTiles(UnwrapExtendTiles const&) = default;
  UnwrapExtendTiles& operator=(UnwrapExtendTiles const&) = default;

  // We use trailing decltype for SFINAE. This ensures that this does not
  // become a candidate when F is not callable with the given arguments.
  template <typename... Ts>
  auto operator()(Ts&&... ts) -> decltype(callHelper(
      std::is_void<decltype(hpx::invoke(hpx::unwrapping(std::declval<F>()), std::declval<Ts>()...))>{},
      std::forward<Ts>(ts)...)) {
    return callHelper(std::is_void<decltype(
                          hpx::invoke(hpx::unwrapping(std::declval<F>()), std::declval<Ts>()...))>{},
                      std::forward<Ts>(ts)...);
  }

private:
  F f;
};
}

/// Custom version of hpx::unwrapping for tile lifetime management.
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

/// Access return value and arguments of the function wrapped by unwrapExtendTiles
///
/// This helper function works as the similar getUnwrapReturnValue, but instead of returning just
/// the return value of the function, it returns also the tuple with all the parameters used in the
/// call.
template <class R, class... Ts>
auto getUnwrapRetValAndArgs(hpx::future<hpx::tuple<R, hpx::tuple<Ts...>>>&& f) {
  auto wrapped_res = hpx::split_future(std::move(f));
  auto ret_value = std::move(hpx::get<0>(wrapped_res));
  auto args = hpx::split_future(std::move(hpx::get<1>(wrapped_res)));
  return std::make_pair(std::move(ret_value), std::move(args));
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
