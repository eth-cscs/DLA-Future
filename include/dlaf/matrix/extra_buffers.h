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

#include <vector>

#include "dlaf/blas/tile_extensions.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf {

template <class T, Device D>
struct ExtraBuffers : protected Matrix<T, D> {
  ExtraBuffers(const TileElementSize bs, const SizeType size)
      : Matrix<T, D>{{bs.rows() * size, bs.cols()}, bs}, nbuffers_(size) {
    namespace ex = pika::execution::experimental;
    for (const auto& i : common::iterate_range2d(Matrix<T, D>::distribution().localNrTiles()))
      ex::start_detached(Matrix<T, D>::readwrite_sender(i) |
                         tile::set0(dlaf::internal::Policy<dlaf::DefaultBackend_v<D>>(
                             pika::execution::thread_priority::high)));
  }

  auto readwrite_sender(SizeType index) {
    index %= nbuffers_;
    return Matrix<T, D>::readwrite_sender(LocalTileIndex{index, 0});
  }

  template <class TileSender>
  [[nodiscard]] auto reduce(TileSender tile) {
    namespace ex = pika::execution::experimental;

    std::vector<pika::future<matrix::Tile<T, D>>> buffers;
    for (const auto& ij : common::iterate_range2d(this->distribution().localNrTiles()))
      buffers.emplace_back(Matrix<T, D>::operator()(ij));
    auto all_buffers = ex::when_all_vector(std::move(buffers));

    return ex::when_all(std::move(tile), std::move(all_buffers)) |
           ex::then([](const matrix::Tile<T, D>& tile, const std::vector<matrix::Tile<T, D>>& buffers) {
             tile::internal::set0(tile);
             for (auto& buffer : buffers)
               dlaf::tile::internal::add(T(1), buffer, tile);
           });
  }

protected:
  SizeType nbuffers_;
};
}
