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

#include <pika/future.hpp>

#include "dlaf/matrix/tile.h"

namespace dlaf::matrix::internal {

// Attach the promise to the tile included in old_future with a continuation and returns a new future to it.
template <class ReturnTileType>
pika::future<ReturnTileType> setPromiseTileFuture(
    pika::future<typename ReturnTileType::TileDataType> old_future,
    pika::lcos::local::promise<typename ReturnTileType::TileDataType> p) noexcept {
  namespace ex = pika::execution::experimental;

  using TileDataType = typename ReturnTileType::TileDataType;
  using NonConstTileType = typename ReturnTileType::TileType;

  DLAF_ASSERT_HEAVY(old_future.valid(), "");

  // This uses keep_future because we want to handle exceptions in a special
  // way. This is special case where we want to use keep_future on a regular
  // future. This case is ok because we:
  // 1. are not dealing with Tiles but TileData
  // 2. are using plain then and don't have special lifetime requirements like
  //    in transform
  auto set_promise = [p = std::move(p)](pika::future<TileDataType>&& tile) mutable {
    std::exception_ptr current_exception_ptr;

    try {
      return ReturnTileType(std::move(NonConstTileType(tile.get()).setPromise(std::move(p))));
    }
    catch (...) {
      current_exception_ptr = std::current_exception();
    }

    // The exception is set outside the catch block since set_exception may
    // yield. Ending the catch block on a different worker thread than where it
    // was started may lead to segfaults.
    p.set_exception(current_exception_ptr);
    std::rethrow_exception(current_exception_ptr);
  };
  return ex::keep_future(std::move(old_future)) | ex::then(std::move(set_promise)) | ex::make_future();
}

// Returns a future<ReturnTileType> setting a new promise p to the tile contained in tile_future.
// tile_future is then updated with the new internal state future (which value is set by p).
template <class ReturnTileType>
pika::future<ReturnTileType> getTileFuture(
    pika::future<typename ReturnTileType::TileDataType>& tile_future) noexcept {
  using TileDataType = typename ReturnTileType::TileDataType;

  pika::future<TileDataType> old_future = std::move(tile_future);
  pika::lcos::local::promise<TileDataType> p;
  tile_future = p.get_future();
  return setPromiseTileFuture<ReturnTileType>(std::move(old_future), std::move(p));
}

// TileFutureManager manages the futures and promises for tiles in the Matrix object.
// See misc/synchronization.md for details.
template <class T, Device D>
class TileFutureManager {
public:
  using TileType = Tile<T, D>;
  using ConstTileType = Tile<const T, D>;
  using TileDataType = internal::TileData<T, D>;

  TileFutureManager() {}

  TileFutureManager(TileDataType tile) : tile_future_(pika::make_ready_future(std::move(tile))) {}

  pika::shared_future<ConstTileType> getReadTileSharedFuture() noexcept {
    if (!tile_shared_future_.valid()) {
      tile_shared_future_ = getTileFuture<ConstTileType>(tile_future_);
    }
    return tile_shared_future_;
  }

  pika::future<TileType> getRWTileFuture() noexcept {
    tile_shared_future_ = {};
    return getTileFuture<TileType>(tile_future_);
  }

  // Waits all the work on this tile to be completed
  // and destroys the tile.
  // Note that this operation invalidates the internal state of the object,
  // which shouldn't be used anymore.
  void clearSync() {
    DLAF_ASSERT_HEAVY(tile_future_.valid(), "");
    tile_shared_future_ = {};
    tile_future_.get();
  }

protected:
  // The future of the tile with no promise set.
  pika::future<TileDataType> tile_future_;

  // If valid, a copy of the shared future of the tile,
  // which has the promise set to the promise for tile_future_.
  pika::shared_future<ConstTileType> tile_shared_future_;
};

// TileFutureManager manages the futures and promises for tiles in the Matrix object.
// See misc/synchronization.md for details.
template <class T, Device D>
class SplittedTileFutureManager {
public:
  using TileType = Tile<T, D>;
  using ConstTileType = Tile<const T, D>;
  using TileDataType = internal::TileData<T, D>;
  using DepTrackerType = pika::shared_future<TileType>;
  using DepTrackerSender = pika::execution::experimental::unique_any_sender<DepTrackerType>;

  SplittedTileFutureManager() {}

  SplittedTileFutureManager(pika::future<TileType> tile) {
    namespace ex = pika::execution::experimental;
    auto setup = [](TileType tile) {
      DLAF_ASSERT_MODERATE(std::holds_alternative<pika::shared_future<TileType>>(tile.dep_tracker_), "");

      auto dep = std::get<DepTrackerType>(tile.dep_tracker_);
      tile.dep_tracker_ = std::monostate();

      return std::make_tuple<TileDataType, DepTrackerType>(std::move(tile.data_), std::move(dep));
    };
    auto ret = ex::split_tuple(std::move(tile) | ex::then(std::move(setup)));

    tile_future_ = std::move(std::get<0>(ret)) | ex::make_future();
    dep_tracker_ = std::move(std::get<1>(ret));
  }

  SplittedTileFutureManager(SplittedTileFutureManager&& rhs)
      : tile_future_(std::move(rhs.tile_future_)),
        tile_shared_future_(std::move(rhs.tile_shared_future_)),
        dep_tracker_(std::move(rhs.dep_tracker_)) {
    rhs.dep_tracker_.reset();
  }

  ~SplittedTileFutureManager() {
    clear();
  }

  SplittedTileFutureManager& operator=(SplittedTileFutureManager&& rhs) {
    tile_future_ = std::move(rhs.tile_future_);
    tile_shared_future_ = std::move(rhs.tile_shared_future_);
    dep_tracker_ = std::move(rhs.dep_tracker_);
    rhs.dep_tracker_.reset();
    return *this;
  }

  pika::shared_future<ConstTileType> getReadTileSharedFuture() noexcept {
    DLAF_ASSERT_HEAVY(dep_tracker_.has_value(), "Accessing a cleared Manager");
    if (!tile_shared_future_.valid()) {
      tile_shared_future_ = getTileFuture<ConstTileType>(tile_future_);
    }
    return tile_shared_future_;
  }

  pika::future<TileType> getRWTileFuture() noexcept {
    DLAF_ASSERT_HEAVY(dep_tracker_.has_value(), "Accessing a cleared Manager");
    tile_shared_future_ = {};
    return getTileFuture<TileType>(tile_future_);
  }

  // Destroys the pipeline and when the work is completed triggers the original pipeline.
  // and destroys the tile.
  // Note that this operation invalidates the internal state of the object,
  // which shouldn't be used anymore,
  // however it is safe to call clear() multiple times.
  void clear() {
    namespace ex = pika::execution::experimental;
    if (dep_tracker_) {
      DLAF_ASSERT_HEAVY(tile_future_.valid(), "");

      ex::start_detached(ex::when_all(std::move(tile_future_), std::move(*dep_tracker_)) |
                         ex::then([](TileDataType, DepTrackerType dep) { dep.get(); }));
      dep_tracker_.reset();
    }
  }

protected:
  // The future of the tile with no promise set.
  pika::future<TileDataType> tile_future_;

  // If valid, a copy of the shared future of the tile,
  // which has the promise set to the promise for tile_future_.
  pika::shared_future<ConstTileType> tile_shared_future_;

  // The original dependency tracker to be released when finished working with this pipeline.
  std::optional<DepTrackerSender> dep_tracker_;
};

}
