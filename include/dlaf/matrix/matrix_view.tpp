//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

namespace dlaf {
namespace matrix {

template <class T, Device device>
hpx::shared_future<Tile<const T, device>> MatrixView<T, device>::read(
    const LocalTileIndex& index) noexcept {
  return tileManager(index).getReadTileSharedFuture();
}

template <class T, Device device>
hpx::future<Tile<T, device>> MatrixView<T, device>::operator()(const LocalTileIndex& index) noexcept {
  return tileManager(index).getRWTileFuture();
}

template <class T, Device device>
void MatrixView<T, device>::doneWrite(const LocalTileIndex& index) noexcept {
  return tileManager(index).makeRead();
}

template <class T, Device device>
void MatrixView<T, device>::done(const LocalTileIndex& index) noexcept {
  return tileManager(index).release();
}

}
}
