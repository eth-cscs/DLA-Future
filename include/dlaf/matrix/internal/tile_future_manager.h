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
  TileFutureManager(hpx::future<TileType>&& tile_future,
                    hpx::shared_future<ConstTileType>&& tile_shared_future)
      : tile_future_(std::move(tile_future)), tile_shared_future_(std::move(tile_shared_future)) {}

  // The future of the tile with no promise set.
  hpx::future<TileType> tile_future_;

  // If valid, a copy of the shared future of the tile,
  // which has the promise set to the promise for tile_future_.
  hpx::shared_future<ConstTileType> tile_shared_future_;
};

enum class TileStatus { None = 0, Read = 1, RW = 2 };

template <class T, Device device>
class ViewTileFutureManager : private TileFutureManager<T, device> {
  using TileManagerType = TileFutureManager<T, device>;

public:
  using typename TileManagerType::TileType;
  using typename TileManagerType::ConstTileType;

  ViewTileFutureManager() : tile_status_(TileStatus::None) {}

  ViewTileFutureManager(TileManagerType& tile_manager, bool /* force_RW */) {
    setUpRW(tile_manager);
  }

  ViewTileFutureManager(ViewTileFutureManager& tile_manager, bool force_RW) {
    // RW tiles are fine in any case.
    if (tile_manager.status() == TileStatus::RW) {
      setUpRW(tile_manager);
      return;
    }

    // Non-RW tiles are fine only if force_RW is false.
    assert(!force_RW);

    if (tile_manager.status() == TileStatus::Read)
      tile_shared_future_ = tile_manager.getReadTileSharedFuture();

    tile_status_ = tile_manager.status();
  }

  ViewTileFutureManager(ViewTileFutureManager&& tile_manager) noexcept
      : TileManagerType(std::move(tile_manager)), tile_status_(tile_manager.tile_status_),
        tile_promise_(std::move(tile_manager.tile_promise_)),
        tile_shared_promise_(std::move(tile_manager.tile_shared_promise_)) {
    tile_manager.tile_status_ = TileStatus::None;
  }

  ~ViewTileFutureManager() {
    release();
  }

  hpx::shared_future<ConstTileType> getReadTileSharedFuture() noexcept {
    assert(tile_status_ != TileStatus::None);
    return TileManagerType::getReadTileSharedFuture();
  }

  hpx::future<TileType> getRWTileFuture() noexcept {
    assert(tile_status_ == TileStatus::RW);
    return TileManagerType::getRWTileFuture();
  }

  void makeRead() {
    if (tile_status_ == TileStatus::RW) {
      tile_status_ = TileStatus::Read;

      auto sf = getReadTileSharedFuture();

      hpx::lcos::local::promise<TileType> p;
      auto future = p.get_future();

      // Set the original matrix shared future to a duplicate of the view shared_future.
      sf.then(  //
          hpx::launch::sync, [p = std::move(p), sp = std::move(tile_shared_promise_)](
                                 const hpx::shared_future<ConstTileType>& sf) mutable {
            try {
              const auto& original_tile = sf.get();
              auto memory_view_copy = original_tile.memory_view_;
              TileType tile(original_tile.size_, std::move(memory_view_copy), original_tile.ld_);
              tile.setPromise(std::move(p));
              sp.set_value(ConstTileType(std::move(tile)));
            }
            catch (...) {
              p.set_exception(std::current_exception());
              sp.set_exception(std::current_exception());
            }
          });

      // Set the original matrix future as ready only when both the view shared_future
      // and the original matrix shared_future have been destroyed.
      hpx::dataflow(
          hpx::launch::sync,
          [p = std::move(tile_promise_)](hpx::future<TileType>&& future1,
                                         hpx::future<TileType>&& future2) mutable {
            try {
              auto tile = future1.get();
              future2.get();
              p.set_value(std::move(tile));
            }
            catch (...) {
              p.set_exception(std::current_exception());
            }
          },
          tile_future_, future);
    }
  }

  TileStatus status() {
    return tile_status_;
  }

  void release() {
    makeRead();

    tile_status_ = TileStatus::None;
    tile_shared_future_ = {};
  }

private:
  void setUpRW(TileManagerType& tile_manager) {
    // the futures of the tile_manager has to be moved in this object and
    // be replaced with the futures of the promises of *this.
    // Therefore we first set up the futures for tile_manager in *this
    // and then swap the base class members (i.e. the futures).
    tile_future_ = tile_promise_.get_future();
    tile_shared_future_ = tile_shared_promise_.get_future();

    TileManagerType& base = *this;
    std::swap(tile_manager, base);

    tile_status_ = TileStatus::RW;
    assert(tile_future_.valid());
  }

  using TileManagerType::tile_future_;
  using TileManagerType::tile_shared_future_;

  TileStatus tile_status_;

  // promise to set tile_future_ in the original matrix.
  hpx::lcos::local::promise<TileType> tile_promise_;

  // promise to set tile_shared_future_ in the original matrix.
  hpx::lcos::local::promise<ConstTileType> tile_shared_promise_;
};
}
}
}
