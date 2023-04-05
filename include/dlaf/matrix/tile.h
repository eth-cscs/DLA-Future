//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <exception>
#include <ostream>
#include <tuple>
#include <type_traits>

#include <pika/execution.hpp>
#include <pika/functional.hpp>
#include <pika/synchronization/async_rw_mutex.hpp>

#include "dlaf/common/data_descriptor.h"
#include "dlaf/matrix/index.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"

namespace dlaf::matrix {
namespace internal {

template <class T, Device D>
class TileData {
public:
  TileData(const TileElementSize& size, memory::MemoryView<T, D> memory_view, SizeType ld) noexcept
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
  memory::MemoryView<T, D> memory_view_;
  SizeType ld_;
};
}

/// Contains the information to create a subtile.
struct SubTileSpec {
  TileElementIndex origin;
  TileElementSize size;
};

// forward declarations
template <class T, Device D>
class Tile;

template <class T, Device D>
class Tile<const T, D>;

// TODO: Should these be public?
template <class T, Device D>
using TileAsyncRwMutex =
    pika::execution::experimental::async_rw_mutex<Tile<T, D>, const Tile<const T, D>>;

template <class T, Device D>
using TileAsyncRwMutexReadWriteWrapper =
    pika::execution::experimental::detail::async_rw_mutex_access_wrapper<
        Tile<T, D>, const Tile<const T, D>,
        pika::execution::experimental::detail::async_rw_mutex_access_type::readwrite>;
// TODO: This could in theory look like this. However, it seems like we end
// up with recursive definitions and incomplete types if we try. Can that
// be avoided?
// typename TileAsyncRwMutex<T, D>::readwrite_access_type;

template <class T, Device D>
using TileAsyncRwMutexReadOnlyWrapper =
    pika::execution::experimental::detail::async_rw_mutex_access_wrapper<
        Tile<T, D>, const Tile<const T, D>,
        pika::execution::experimental::detail::async_rw_mutex_access_type::read>;

// TODO: Replace by above everywhere.
template <class T, Device D>
using tile_async_rw_mutex_wrapper_type =
    pika::execution::experimental::detail::async_rw_mutex_access_wrapper<
        Tile<T, D>, const Tile<const T, D>,
        pika::execution::experimental::detail::async_rw_mutex_access_type::readwrite>;

template <class T, Device D>
using tile_async_ro_mutex_wrapper_type =
    pika::execution::experimental::detail::async_rw_mutex_access_wrapper<
        Tile<T, D>, const Tile<const T, D>,
        pika::execution::experimental::detail::async_rw_mutex_access_type::read>;

template <class T, Device D>
using ReadWriteTileSender = pika::execution::experimental::unique_any_sender<Tile<T, D>>;

template <class T, Device D>
using ReadOnlyTileSender =
    pika::execution::experimental::any_sender<TileAsyncRwMutexReadOnlyWrapper<T, D>>;

namespace internal {
// TODO: Do these need to be free functions or could they be hidden inside Tile?
// They're already friends.
template <class T, Device D>
Tile<T, D> createSubTileAsyncRwMutex(TileAsyncRwMutexReadOnlyWrapper<T, D> tile_wrapper,
                                     const SubTileSpec& spec);

template <class T, Device D>
Tile<T, D> createTileAsyncRwMutex(TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper);

template <class T, Device D>
Tile<T, D> createSubTileAsyncRwMutex(TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper,
                                     const SubTileSpec& spec);

template <class T, Device D>
Tile<T, D> createSubTileAsyncRwMutex(Tile<T, D> tile, const SubTileSpec& spec);
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
template <class T, Device D>
class Tile<const T, D> {
public:
  using TileType = Tile<T, D>;
  using ConstTileType = Tile<const T, D>;
  using TileDataType = internal::TileData<T, D>;

  friend TileType;
  using ElementType = T;
  static constexpr Device device = D;

  /// Constructs a (@p size.rows() x @p size.cols()) Tile.
  ///
  /// @pre size.isValid(),
  /// @pre ld >= max(1, @p size.rows()),
  /// @pre memory_view contains enough elements.
  /// The (i, j)-th element of the Tile is stored in the (i+ld*j)-th element of memory_view.
  Tile(const TileElementSize& size, memory::MemoryView<ElementType, D>&& memory_view,
       SizeType ld) noexcept
      : data_(size, std::move(memory_view), ld) {}

  Tile(TileDataType&& data) noexcept : data_(std::move(data)) {}

  Tile(const Tile&) = delete;

  Tile(Tile&& rhs) noexcept : data_(std::move(rhs.data_)), dep_tracker_(std::move(rhs.dep_tracker_)) {
    rhs.dep_tracker_ = std::monostate();
  };

  /// Destroys the Tile.
  ///
  /// If the tile holds a tracker, the tracker is released.
  ~Tile() = default;

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

  bool is_contiguous() const noexcept {
    return data_.ld() == data_.size().rows();
  }

  /// Prints information about the tile.
  friend std::ostream& operator<<(std::ostream& out, const Tile& tile) {
    return out << "size=" << tile.size() << ", ld=" << tile.ld();
  }

private:
  static memory::MemoryView<T, D> createMemoryViewForSubtile(const Tile<const T, D>& tile,
                                                             const SubTileSpec& spec) {
    DLAF_ASSERT(spec.origin.isValid(), spec.origin);
    DLAF_ASSERT(spec.origin.isInOrOn(tile.size()), spec.origin, tile.size());
    DLAF_ASSERT(spec.size.isValid(), spec.size);
    DLAF_ASSERT((spec.origin + spec.size).isInOrOn(tile.size()), spec.origin, spec.size, tile.size());

    return memory::MemoryView<T, D>(tile.data_.memoryView(),
                                    spec.size.isEmpty() ? 0 : tile.data_.linearIndex(spec.origin),
                                    tile.data_.linearSize(spec.size, tile.ld()));
  };

  Tile(const TileElementSize& size, const memory::MemoryView<ElementType, D>& memory_view,
       SizeType ld) noexcept
      : data_(size, memory_view, ld) {}

  // Creates an untracked subtile.
  // Dependencies are not influenced by the new created object therefore race-conditions
  // might happen if used improperly.
  Tile(const Tile& tile, const SubTileSpec& spec) noexcept;

  TileDataType data_;
  std::variant<std::monostate, TileAsyncRwMutexReadOnlyWrapper<T, D>,
               TileAsyncRwMutexReadWriteWrapper<T, D>>
      dep_tracker_;
};

template <class T, Device D>
Tile<const T, D>::Tile(const Tile<const T, D>& tile, const SubTileSpec& spec) noexcept
    : Tile<const T, D>(spec.size, Tile::createMemoryViewForSubtile(tile, spec), tile.ld()) {}

template <class T, Device D>
class Tile : public Tile<const T, D> {
public:
  using TileType = Tile<T, D>;
  using ConstTileType = Tile<const T, D>;
  using TileDataType = internal::TileData<T, D>;

  friend ConstTileType;
  friend Tile<T, D> internal::createSubTileAsyncRwMutex<>(
      TileAsyncRwMutexReadOnlyWrapper<T, D> tile_wrapper, const SubTileSpec& spec);
  friend Tile<T, D> internal::createTileAsyncRwMutex<>(
      TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper);
  friend Tile<T, D> internal::createSubTileAsyncRwMutex<>(
      TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper, const SubTileSpec& spec);
  friend Tile<T, D> internal::createSubTileAsyncRwMutex<>(Tile<T, D> tile, const SubTileSpec& spec);

  using ElementType = T;

  /// Constructs a (@p size.rows() x @p size.cols()) Tile.
  ///
  /// @pre size.isValid(),
  /// @pre ld >= max(1, @p size.rows()),
  /// @pre memory_view contains enough elements.
  /// The (i, j)-th element of the Tile is stored in the (i+ld*j)-th element of memory_view.
  Tile(const TileElementSize& size, memory::MemoryView<ElementType, D>&& memory_view,
       SizeType ld) noexcept
      : Tile<const T, D>(size, std::move(memory_view), ld) {}

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

private:
  Tile(TileAsyncRwMutexReadOnlyWrapper<T, D> tile_wrapper, const SubTileSpec& spec)
      : Tile<const T, D>(tile_wrapper.get(), spec) {
    dep_tracker_ = std::move(tile_wrapper);
  }

  Tile(TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper)
      : ConstTileType(tile_wrapper.get().size(), tile_wrapper.get().data_.memoryView(),
                      tile_wrapper.get().ld()) {
    dep_tracker_ = std::move(tile_wrapper);
  }

  Tile(TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper, const SubTileSpec& spec)
      : ConstTileType(tile_wrapper.get(), spec) {
    dep_tracker_ = std::move(tile_wrapper);
  }

  Tile(Tile<T, D> tile, const SubTileSpec& spec) : ConstTileType(tile, spec) {
    dep_tracker_ = std::move(tile.dep_tracker_);
  }

  using ConstTileType::data_;
  using ConstTileType::dep_tracker_;
};

/// Create a common::Buffer from a Tile.
template <class T, Device D>
auto create_data(const Tile<T, D>& tile) {
  return common::DataDescriptor<T>(tile.ptr({0, 0}), tile.size().cols(), tile.size().rows(), tile.ld());
}

namespace internal {
template <class T, Device D>
Tile<T, D> createSubTileAsyncRwMutex(TileAsyncRwMutexReadOnlyWrapper<T, D> tile_wrapper,
                                     const SubTileSpec& spec) {
  return {std::move(tile_wrapper), spec};
}

template <class T, Device D>
Tile<T, D> createTileAsyncRwMutex(TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper) {
  return {std::move(tile_wrapper)};
}

template <class T, Device D>
Tile<T, D> createSubTileAsyncRwMutex(TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper,
                                     const SubTileSpec& spec) {
  return {std::move(tile_wrapper), spec};
}

template <class T, Device D>
Tile<T, D> createSubTileAsyncRwMutex(Tile<T, D> tile, const SubTileSpec& spec) {
  return {std::move(tile), spec};
}
}

/// Create a read-only subtile of a given read-only tile.
///
/// The returned subtile will get ready, at the same time as @p tile gets ready.
/// The next dependency in the dependency chain will become ready only when @p tile
/// and the returned subtile go out of scope.
template <class T, Device D>
ReadOnlyTileSender<T, D> subTileSender(ReadOnlyTileSender<T, D> tile, const SubTileSpec& spec) {
  return std::move(tile) |
         pika::execution::experimental::let_value(
             [spec](TileAsyncRwMutexReadOnlyWrapper<T, D>& tile_wrapper) {
               TileAsyncRwMutex<T, D> tile_manager{
                   internal::createSubTileAsyncRwMutex<T, D>(std::move(tile_wrapper), spec)};
               return tile_manager.read();
             }) |
         pika::execution::experimental::split();
}

// TODO: Docs.
// TODO: shareReadWriteTile?
template <class T, Device D>
ReadOnlyTileSender<T, D> shareReadwriteTile(ReadWriteTileSender<T, D>&& tile) {
  return std::move(tile) | pika::execution::experimental::let_value([](Tile<T, D>& tile) {
           TileAsyncRwMutex<T, D> tile_manager{std::move(tile)};
           return tile_manager.read();
         }) |
         pika::execution::experimental::split();
}

// TODO: Docs.
template <class T, Device D>
std::vector<ReadOnlyTileSender<T, D>> subTileSenders(ReadOnlyTileSender<T, D> tile,
                                                     const std::vector<SubTileSpec>& specs) {
  std::vector<ReadOnlyTileSender<T, D>> senders;
  senders.reserve(specs.size());

  for (const auto& spec : specs) {
    senders.push_back(subTileSender(tile, spec));
  }

  return senders;
}

// TODO: Docs.
// TODO: This is the version which takes a ReadWriteWrapper. However, is this
// actually used anywhere? Most use cases should send the Tile without the
// wrapper.
template <class T, Device D>
ReadWriteTileSender<T, D> subTileSender(
    pika::execution::experimental::unique_any_sender<TileAsyncRwMutexReadWriteWrapper<T, D>>&& tile,
    const SubTileSpec& spec) {
  return std::move(tile) |
         pika::execution::experimental::then(
             [spec](TileAsyncRwMutexReadWriteWrapper<T, D> tile_wrapper) {
               return internal::createSubTileAsyncRwMutex<T, D>(std::move(tile_wrapper), spec);
             });
}

// TODO: Docs.
template <class T, Device D>
ReadWriteTileSender<T, D> subTileSender(ReadWriteTileSender<T, D>&& tile, const SubTileSpec& spec) {
  return std::move(tile) | pika::execution::experimental::then([spec](Tile<T, D> tile_wrapper) {
           return internal::createSubTileAsyncRwMutex<T, D>(std::move(tile_wrapper), spec);
         });
}

template <class T, Device D>
std::vector<ReadWriteTileSender<T, D>> subTileSenders(ReadWriteTileSender<T, D>&&,
                                                      const std::vector<SubTileSpec>&) {
  DLAF_UNIMPLEMENTED("read-write version of subTileSenders (with disjoint subtiles)");
  return {};
}

// TODO: Add the subTile/splitTile helpers for accessing a set of disjoint specs.

/// ---- ETI

#define DLAF_TILE_ETI(KWORD, DATATYPE, DEVICE) \
  KWORD template class Tile<DATATYPE, DEVICE>; \
  KWORD template class Tile<const DATATYPE, DEVICE>;

DLAF_TILE_ETI(extern, float, Device::CPU)
DLAF_TILE_ETI(extern, double, Device::CPU)
DLAF_TILE_ETI(extern, std::complex<float>, Device::CPU)
DLAF_TILE_ETI(extern, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_GPU)
DLAF_TILE_ETI(extern, float, Device::GPU)
DLAF_TILE_ETI(extern, double, Device::GPU)
DLAF_TILE_ETI(extern, std::complex<float>, Device::GPU)
DLAF_TILE_ETI(extern, std::complex<double>, Device::GPU)
#endif
}
