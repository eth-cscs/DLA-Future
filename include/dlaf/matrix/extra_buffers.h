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

  auto read_sender(SizeType index) {
    return Matrix<T, D>::read_sender(internalIndex(index));
  }

  auto readwrite_sender(SizeType index) {
    return Matrix<T, D>::readwrite_sender(internalIndex(index));
  }

  template <class TileSender>
  [[nodiscard]] auto reduce(TileSender tile) {
    namespace di = dlaf::internal;
    namespace ex = pika::execution::experimental;

    std::vector<ex::any_sender<pika::shared_future<matrix::Tile<const T, D>>>> buffers;
    for (SizeType index = 0; index < nbuffers_; ++index)
      buffers.emplace_back(read_sender(index));

    return ex::when_all(std::move(tile), ex::when_all_vector(std::move(buffers))) |
           di::transform(di::Policy<DefaultBackend_v<D>>(),
                         [](const matrix::Tile<T, D>& tile,
                            const std::vector<pika::shared_future<matrix::Tile<const T, D>>>& buffers,
                            auto&&... ts) {
                           for (const auto& buffer : buffers)
                             dlaf::tile::internal::add(T(1), buffer.get(), tile, ts...);
                         });
  }

protected:
  LocalTileIndex internalIndex(SizeType index) const noexcept {
    return LocalTileIndex{index % nbuffers_, 0};
  }

  SizeType nbuffers_;
};
}
