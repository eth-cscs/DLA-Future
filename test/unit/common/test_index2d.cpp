//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/index2d.h"

#include <gtest/gtest.h>

TEST(Index2D, basic) {
  dlaf::common::Index2D index(5, 3);

  EXPECT_EQ(5, index.row());
  EXPECT_EQ(3, index.col());
}
