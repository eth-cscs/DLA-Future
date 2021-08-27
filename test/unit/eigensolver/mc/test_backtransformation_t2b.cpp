//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause

#include "dlaf/eigensolver/backtransformation.h"

#include <gtest/gtest.h>

#include "dlaf/matrix/matrix.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"

using namespace dlaf;
using namespace testing;

template <typename Type>
class BacktransformationT2BTest : public ::testing::Test {};

TYPED_TEST_SUITE(BacktransformationT2BTest, MatrixElementTypes);

TYPED_TEST(BacktransformationT2BTest, CorrectnessLocal) {
  // TODO random V
  // TODO compute taus (compact format)

  // TODO call bt-t2b

  // TODO gather locally V compact
  // TODO apply one reflector per time col-by-col from the bottom

  // TODO gather result
  // TODO compare results
}
