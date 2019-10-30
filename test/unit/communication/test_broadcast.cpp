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

using BroadcastTest = dlaf_test::SplittedCommunicatorsTest;

TEST_F(BroadcastTest, Broadcast_NewAPI) {
  auto broadcaster = 0;
  auto communicator = splitted_comm;

  if (isMasterInSplitted()) {
    const int message = color;
    dlaf::comm::broadcast::send(dlaf::comm::make_message(&message, 1), communicator);
  }
  else {
    int message;
    dlaf::comm::broadcast::receive_from(broadcaster, dlaf::comm::make_message(&message, 1),
                                        communicator);
    EXPECT_EQ(color, message);
  }
}

TEST_F(BroadcastTest, AsyncBroadcast_NewAPI) {
  auto broadcaster = 0;
  auto communicator = splitted_comm;

  bool waited;
  auto what_to_do_before_retesting = [&waited]() { waited = true; };

  if (isMasterInSplitted())
    dlaf::comm::async_broadcast::send(dlaf::comm::make_message(&color, 1), communicator,
                                      what_to_do_before_retesting);
  else {
    int message;
    dlaf::comm::async_broadcast::receive_from(broadcaster, dlaf::comm::make_message(&message, 1),
                                              communicator, what_to_do_before_retesting);
    EXPECT_EQ(color, message);
  }
}
