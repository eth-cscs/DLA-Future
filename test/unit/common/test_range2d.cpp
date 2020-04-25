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
#include "dlaf/common/range2d.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace {
struct CustomTag;
}

TEST(Range2D, ColMajorRange) {
  using namespace dlaf;
  using Index = common::Index2D<SizeType, ::CustomTag>;
  using Size = common::Size2D<SizeType, ::CustomTag>;

  Size sz(7, 2);

  std::vector<Index> exp_indices = {
      Index(0, 0), Index(1, 0), Index(2, 0), Index(3, 0), Index(4, 0), Index(5, 0), Index(6, 0),
      Index(0, 1), Index(1, 1), Index(2, 1), Index(3, 1), Index(4, 1), Index(5, 1), Index(6, 1),
  };

  std::vector<Index> act_indices;
  act_indices.reserve(exp_indices.size());

  for (Index i : iterateColMajor(sz)) {
    act_indices.push_back(i);
  }

  ASSERT_TRUE(act_indices == exp_indices);
}

TEST(Range2D, RowMajorRange) {
  using namespace dlaf;
  using Index = common::Index2D<SizeType, ::CustomTag>;
  using Size = common::Size2D<SizeType, ::CustomTag>;

  Size sz(6, 3);

  std::vector<Index> exp_indices = {
      Index(0, 0), Index(0, 1), Index(0, 2), Index(1, 0), Index(1, 1), Index(1, 2),
      Index(2, 0), Index(2, 1), Index(2, 2), Index(3, 0), Index(3, 1), Index(3, 2),
      Index(4, 0), Index(4, 1), Index(4, 2), Index(5, 0), Index(5, 1), Index(5, 2),
  };

  std::vector<Index> act_indices;
  act_indices.reserve(exp_indices.size());

  for (Index i : iterateRowMajor(sz)) {
    act_indices.push_back(i);
  }

  ASSERT_TRUE(act_indices == exp_indices);
}
