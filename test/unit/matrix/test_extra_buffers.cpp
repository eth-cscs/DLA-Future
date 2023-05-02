//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/extra_buffers.h"

#include <gtest/gtest.h>

#include "dlaf/common/range2d.h"
#include "dlaf/matrix/print_numpy.h"

#include "dlaf_test/matrix/util_tile.h"

using namespace dlaf;

TEST(ExtraBuffersTest, Basic) {
  using T = float;
  constexpr auto D = Device::CPU;

  namespace ex = pika::execution::experimental;
  namespace tt = pika::this_thread::experimental;

  TileElementSize tile_size(2, 2);
  Matrix<T, D> tile({tile_size.rows(), tile_size.cols()}, tile_size);
  constexpr SizeType nbuffers = 10;
  ExtraBuffers<T, D> buffers(tile_size, nbuffers);

  for (SizeType i = 0; i < nbuffers; ++i) {
    tt::sync_wait(ex::when_all(buffers.readwrite_sender(i), ex::just(T(1))) |
                  ex::then([](const auto& tile, const T value) { matrix::test::set(tile, value); }));
  }

  ex::start_detached(buffers.reduce(tile.readwrite_sender(LocalTileIndex{0, 0})));

  print(format::numpy{}, tile.read(LocalTileIndex(0, 0)).get());
}
