//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <hpx/hpx.hpp>
#include "dlaf/matrix/tile.h"

namespace dlaf {
namespace matrix {
namespace internal {

// Attach the promise to the tile included in old_future with a continuation and returns a new future to it.
template <class ReturnTileType, class TileType>
hpx::future<ReturnTileType> setPromiseTileFuture(hpx::future<TileType> old_future,
                                                 hpx::lcos::local::promise<TileType> p) noexcept {
  DLAF_ASSERT_HEAVY(old_future.valid(), "");
  return old_future.then(hpx::launch::sync, [p = std::move(p)](hpx::future<TileType>&& fut) mutable {
    std::exception_ptr current_exception_ptr;

    try {
      return ReturnTileType(std::move(fut.get().setPromise(std::move(p))));
    }
    catch (...) {
      current_exception_ptr = std::current_exception();
    }

    // The exception is set outside the catch block since set_exception may
    // yield. Ending the catch block on a different worker thread than where it
    // was started may lead to segfaults.
    p.set_exception(current_exception_ptr);
    std::rethrow_exception(current_exception_ptr);
  });
}

// Returns a future<ReturnTileType> setting a new promise p to the tile contained in tile_future.
// tile_future is then updated with the new internal state future (which value is set by p).
template <class ReturnTileType, class TileType>
hpx::future<ReturnTileType> getTileFuture(hpx::future<TileType>& tile_future) noexcept {
  hpx::future<TileType> old_future = std::move(tile_future);
  hpx::lcos::local::promise<TileType> p;
  tile_future = p.get_future();
  return setPromiseTileFuture<ReturnTileType>(std::move(old_future), std::move(p));
}

// TileFutureManager manages the futures and promises for tiles in the Matrix object.
// See misc/synchronization.md for details.
template <class T, Device device>
class TileFutureManager {
public:
  using TileType = Tile<T, device>;
  using ConstTileType = Tile<const T, device>;

  TileFutureManager() {}

  TileFutureManager(TileType tile) : tile_future_(hpx::make_ready_future(std::move(tile))) {}

  hpx::shared_future<ConstTileType> getReadTileSharedFuture() noexcept {
    if (!tile_shared_future_.valid()) {
      tile_shared_future_ = getTileFuture<ConstTileType>(tile_future_);
    }
    return tile_shared_future_;
  }

  hpx::future<TileType> getRWTileFuture() noexcept {
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
  hpx::future<TileType> tile_future_;

  // If valid, a copy of the shared future of the tile,
  // which has the promise set to the promise for tile_future_.
  hpx::shared_future<ConstTileType> tile_shared_future_;
};

}
}
}
