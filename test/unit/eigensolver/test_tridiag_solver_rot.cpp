//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/eigensolver/tridiag_solver/rot.h"

#include <gtest/gtest.h>

#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;

namespace ex = pika::execution::experimental;
namespace di = dlaf::eigensolver::internal;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename T>
class TridiagEigensolverRotTest : public TestWithCommGrids {};

// TODO use MatrixElementTypes
using JustThisType = ::testing::Types<float>;

TYPED_TEST_SUITE(TridiagEigensolverRotTest, JustThisType);

template <class T>
void testApplyGivenRotations(comm::CommunicatorGrid grid, const SizeType m, const SizeType mb,
                             const SizeType idx_begin, const SizeType idx_last,
                             std::vector<di::GivensRotation<T>> rots) {
  using dlaf::eigensolver::internal::applyGivensRotationsToMatrixColumns;

  matrix::Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), {0, 0});

  // TODO initialize matrix randomly
  matrix::Matrix<T, Device::CPU> matrix(dist);

  auto rots_fut = ex::just(rots);

  applyGivensRotationsToMatrixColumns(grid, idx_begin, idx_last, std::move(rots_fut), matrix);

  // TODO collect locally mat
  // TODO apply blas::rot
  // TODO check equality
}

template <class T>
struct config_t {
  SizeType m;
  SizeType mb;
  SizeType i_begin;
  SizeType i_last;
  std::vector<di::GivensRotation<T>> rots;
};

TYPED_TEST(TridiagEigensolverRotTest, ApplyGivenRotations) {
  std::vector<config_t<TypeParam>> configs = {{9,
                                               3,
                                               0,
                                               2,
                                               {di::GivensRotation<TypeParam>{0, 8, 0.5f, 0.5f},
                                                di::GivensRotation<TypeParam>{0, 2, 0.5f, 0.5f}}}};

  for (const auto& grid : this->commGrids()) {
    for (const auto& [m, mb, idx_begin, idx_last, rots] : configs) {
      testApplyGivenRotations<TypeParam>(grid, m, mb, idx_begin, idx_last, rots);
    }
  }
}
