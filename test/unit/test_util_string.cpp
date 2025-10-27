//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <string>
#include <vector>

#include <dlaf/util_string.h>

#include <gtest/gtest.h>

using namespace dlaf;
using namespace testing;

std::vector<std::string> strings{"asdf, (+6]g", "ASDF, (+6]G", "AsDf, (+6]G", "aSdF, (+6]g",
                                 "asDF, (+6]G"};

TEST(StringUtilTest, ToLower) {
  std::string expected("asdf, (+6]g");

  for (const auto& s : strings) {
    EXPECT_EQ(expected, util::copy_to_lower(s));
  }
}
