//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/rot.h"

#include <gtest/gtest.h>

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class TridiagEigensolverRotTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(TridiagEigensolverRotTest, MatrixElementTypes);

TYPED_TEST(TridiagEigensolverRotTest, ApplyGivenRotations) {
  using dlaf::eigensolver::internal::applyGivensRotationsToMatrixColumns;
}
