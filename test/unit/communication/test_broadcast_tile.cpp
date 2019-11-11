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

#include "internal/helper_communicators.h"
#include "dlaf_test/util_types.h"

#include "dlaf/tile.h"

using BroadcastTileTest = dlaf_test::SplittedCommunicatorsTest;

TEST_F(BroadcastTileTest, SyncTile) {
  using TypeParam = std::complex<float>;

  TypeParam data[4];
  dlaf::Tile<TypeParam, dlaf::Device::CPU> tile({2, 2}, {data, 4}, 2);

  EXPECT_EQ(2, tile.size().rows());
  EXPECT_EQ(2, tile.size().cols());

  auto message_values = [](dlaf::TileElementIndex index) -> TypeParam {
    return dlaf_test::TypeUtilities<TypeParam>::element((index.row() + 1) + (index.col() + 1) * 10,
                                                        (index.col() + 1));
  };

  if (splitted_comm.rank() == 0) {
    for (int j = 0; j < tile.size().cols(); ++j)
      for (int i = 0; i < tile.size().rows(); ++i)
        *tile.ptr({i, j}) = message_values({i, j});

    dlaf::comm::broadcast::send(dlaf::comm::make_message(tile), splitted_comm);
  }
  else {
    for (int j = 0; j < tile.size().cols(); ++j)
      for (int i = 0; i < tile.size().rows(); ++i)
        EXPECT_NE(message_values({i, j}), *tile.ptr({i, j}));

    dlaf::comm::broadcast::receive_from(0, dlaf::comm::make_message(tile), splitted_comm);
  }

  for (int j = 0; j < tile.size().cols(); ++j)
    for (int i = 0; i < tile.size().rows(); ++i)
      EXPECT_EQ(message_values({i, j}), *tile.ptr({i, j}));
}
