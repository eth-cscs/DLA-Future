//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <limits>

#include "dlaf/common/vector.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/types.h"
#include "dlaf_test/matrix/util_tile.h"

namespace dlaf::miniapp {

/// Creates a set of tiles that can be used to benchmark kernels.
///
/// It helps removing the cache effect when the same tile is used multiple times.
template <class T, Device device>
class WorkTiles {
public:
  using TileType = matrix::Tile<T, device>;

  WorkTiles(SizeType count, SizeType m, SizeType n, SizeType ld) noexcept {
    DLAF_ASSERT(count > 0, count);
    DLAF_ASSERT(ld >= std::max<SizeType>(1, m), ld, m);

    tiles_.reserve(count);
    const SizeType tiles_in_chunk = ld / m;
    const SizeType chunk_size = ld * n;
    const SizeType chunks = util::ceilDiv(count, tiles_in_chunk);

    memory::MemoryView<T, device> mem(chunks * chunk_size);

    for (SizeType i = 0; i < count; ++i) {
      memory::MemoryView<T, device> sub_mem(mem,
                                            i % tiles_in_chunk * m + i / tiles_in_chunk * chunk_size,
                                            ld * (n - 1) + m);
      tiles_.emplace_back(TileElementSize(m, n), std::move(sub_mem), ld);
    }
  }

  void setElementsFromTile(const matrix::Tile<const T, Device::CPU>& tile0) noexcept {
    for (auto& tile : tiles_)
      matrix::internal::copy_o(tile0, tile);
  }

  template <class F>
  void setElements(F&& el) noexcept {
    const auto size = tiles_[0].size();
    auto tile0 = matrix::test::createTile<const T>(el, size, std::max<SizeType>(1, size.rows()));
    setElementsFromTile(tile0);
  }

  SizeType count() const noexcept {
    return tiles_.size();
  }

  TileType& operator()(SizeType index) noexcept {
    return tiles_[index];
  }

  template <class F>
  BaseType<T> check(F&& f) const noexcept {
    BaseType<T> error = 0;
    for (auto& tile : tiles_) {
      const auto size = tile.size();
      auto tile_cp = matrix::test::createTile<T>(size, std::max<SizeType>(1, size.rows()));
      matrix::internal::copy_o(tile, tile_cp);
      error = std::max(error, checkTile(f, tile_cp));
    }
    return error;
  }

private:
  template <class F>
  static auto checkTile(F&& f, const matrix::Tile<T, Device::CPU>& tile) noexcept {
    auto norm =
        lapack::lange(lapack::Norm::Max, tile.size().rows(), tile.size().cols(), tile.ptr(), tile.ld());
    for (const auto& index : iterate_range2d(tile.size())) {
      tile(index) -= f(index);
    }
    auto norm_delta =
        lapack::lange(lapack::Norm::Max, tile.size().rows(), tile.size().cols(), tile.ptr(), tile.ld());

    return norm_delta / norm / std::numeric_limits<BaseType<T>>::epsilon();
  }

  common::internal::vector<TileType> tiles_;
};

}
