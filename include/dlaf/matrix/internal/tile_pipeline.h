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

#include "dlaf/matrix/tile.h"

namespace dlaf::matrix::internal {
template <class T, Device D>
class TilePipeline {
public:
  explicit TilePipeline(Tile<T, D>&& tile) : pipeline(std::move(tile)) {}
  TilePipeline(TilePipeline&&) = default;
  TilePipeline& operator=(TilePipeline&&) = default;
  TilePipeline(const TilePipeline&) = delete;
  TilePipeline& operator=(const TilePipeline&) = delete;

  ReadOnlyTileSender<T, D> read() {
    return pipeline.read();
  }

  ReadWriteTileSender<T, D> readwrite() {
    return pipeline.readwrite() | pika::execution::experimental::then(&createTileAsyncRwMutex<T, D>);
  }

private:
  TileAsyncRwMutex<T, D> pipeline;
};
}
