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

using namespace dlaf_test;
using namespace dlaf::comm;

using BroadcastTest = SplittedCommunicatorsTest;

TEST_F(BroadcastTest, Broadcast_NewAPI) {
  auto broadcaster = 0;
  auto communicator = splitted_comm;

  if (splitted_comm.rank() == 0) {
    const int message = color;
    broadcast::send(make_message(&message, 1), communicator);
  }
  else {
    int message;
    broadcast::receive_from(broadcaster, make_message(&message, 1), communicator);
    EXPECT_EQ(color, message);
  }
}
