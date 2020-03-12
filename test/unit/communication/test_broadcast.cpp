//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/functions_sync.h"

#include <gtest/gtest.h>

#include "dlaf/common/data_descriptor.h"
#include "dlaf_test/helper_communicators.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace dlaf::comm;

using BroadcastTest = SplittedCommunicatorsTest;

TEST_F(BroadcastTest, Broadcast_NewAPI) {
  auto broadcaster = 0;
  auto communicator = splitted_comm;

  if (splitted_comm.rank() == 0) {
    const int message = color;
    sync::broadcast::send(communicator, common::make_data(&message, 1));
  }
  else {
    int message;
    sync::broadcast::receive_from(broadcaster, communicator, common::make_data(&message, 1));
    EXPECT_EQ(color, message);
  }
}
