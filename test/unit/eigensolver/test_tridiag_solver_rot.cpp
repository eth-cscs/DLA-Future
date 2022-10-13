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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/matrix_local.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
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
  // TODO check about "independent" rots?

  using dlaf::eigensolver::internal::applyGivensRotationsToMatrixColumns;

  matrix::Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), {0, 0});

  matrix::Matrix<T, Device::CPU> mat(dist);
  matrix::util::set_random(mat);

  matrix::test::MatrixLocal<T> mat_loc = matrix::test::allGather(blas::Uplo::General, mat, grid);

  applyGivensRotationsToMatrixColumns(grid, idx_begin, idx_last, ex::just(rots), mat);

  // Apply Given Rotations
  for (auto rot : rots) {
    const SizeType n = mat_loc.size().rows();
    T* x = mat_loc.ptr({0, rot.i});
    T* y = mat_loc.ptr({0, rot.j});
    blas::rot(n, x, 1, y, 1, rot.c, rot.s);
  }

  auto result = [&dist = mat.distribution(), &mat_local = mat_loc](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, mat, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
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
                                               {// di::GivensRotation<TypeParam>{0, 8, 0.5f, 0.5f},
                                                di::GivensRotation<TypeParam>{0, 2, 0.5f, 0.5f}}}};

  for (const auto& grid : this->commGrids()) {
    for (const auto& [m, mb, idx_begin, idx_last, rots] : configs) {
      testApplyGivenRotations<TypeParam>(grid, m, mb, idx_begin, idx_last, rots);
    }
  }
}
