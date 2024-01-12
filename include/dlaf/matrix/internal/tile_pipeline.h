//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf/common/assert.h>
#include <dlaf/matrix/tile.h>

namespace dlaf::matrix::internal {
template <class T, Device D>
class TilePipeline {
public:
  explicit TilePipeline(Tile<T, D>&& tile) : pipeline(std::move(tile)) {}
  TilePipeline(TilePipeline&&) = default;
  TilePipeline& operator=(TilePipeline&&) = default;
  TilePipeline(const TilePipeline&) = delete;
  TilePipeline& operator=(const TilePipeline&) = delete;

  /// Get a read-only sender to a wrapped tile.
  ///
  /// The returned sender will send its value when the previous access to the
  /// tile has completed. If the previous access is read-only concurrent access
  /// to the tile will be provided.
  ///
  /// @return A sender to a read-only tile wrapper.
  /// @pre valid()
  ReadOnlyTileSender<T, D> read() {
    DLAF_ASSERT(valid(), "");
    return pipeline->read();
  }

  /// Get a read-write sender to a tile.
  ///
  /// The returned sender will send its value when the previous access to the
  /// tile has completed.
  ///
  /// @return A sender to a read-write tile.
  /// @pre valid()
  ReadWriteTileSender<T, D> readwrite() {
    DLAF_ASSERT(valid(), "");
    return pipeline->readwrite() | pika::execution::experimental::then(&createTileAsyncRwMutex<T, D>);
  }

  /// Get a read-write sender to a wrapped tile.
  ///
  /// The returned sender will send its value when the previous access to the
  /// tile has completed.
  ///
  /// @return A sender to a read-write tile wrapper.
  /// @pre valid()
  auto readwrite_with_wrapper() {
    DLAF_ASSERT(valid(), "");
    return pipeline->readwrite();
  }

  /// Check if the pipeline is valid.
  ///
  /// @return true if the pipeline hasn't been reset, otherwise false.
  bool valid() const noexcept {
    return pipeline.has_value();
  }

  /// Reset the pipeline.
  ///
  /// @post !valid()
  void reset() noexcept {
    pipeline.reset();
  }

private:
  std::optional<TileAsyncRwMutex<T, D>> pipeline;
};
}
