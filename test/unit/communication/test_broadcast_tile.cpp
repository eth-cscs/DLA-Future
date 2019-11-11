//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/functions.h"

#include <gtest/gtest.h>

#include "dlaf_test/helper_communicators.h"
#include "dlaf_test/util_types.h"
#include "dlaf_test/util_tile.h"

#include "dlaf/tile.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace dlaf::comm;

using BroadcastTileTest = SplittedCommunicatorsTest;

TEST_F(BroadcastTileTest, SyncTile) {
  using TypeParam = std::complex<float>;

  TypeParam data[4];
  Tile<TypeParam, dlaf::Device::CPU> tile({2, 2}, {data, 4}, 2);

  EXPECT_EQ(2, tile.size().rows());
  EXPECT_EQ(2, tile.size().cols());

  auto message_values = [](const TileElementIndex& index) -> TypeParam {
    return TypeUtilities<TypeParam>::element((index.row() + 1) + (index.col() + 1) * 10,
                                             (index.col() + 1));
  };

  if (splitted_comm.rank() == 0) {
    tile_test::set(tile, message_values);

    broadcast::send(make_message(tile), splitted_comm);
  }
  else {
    CHECK_TILE_NE(message_values, tile);

    broadcast::receive_from(0, make_message(tile), splitted_comm);
  }

  CHECK_TILE_EQ(message_values, tile);
}
