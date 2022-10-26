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
#include "dlaf/matrix/matrix_mirror.h"

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
struct TridiagEigensolverRotTest : public TestWithCommGrids {
  template <class U>
  struct config_t {
    SizeType m;
    SizeType mb;
    SizeType i_begin;
    SizeType i_last;
    std::vector<di::GivensRotation<U>> rots;
  };

  const std::vector<config_t<T>> configs{
      {9, 3, 0, 2, {di::GivensRotation<T>{0, 8, 0.5f, 0.5f}, di::GivensRotation<T>{0, 2, 0.5f, 0.5f}}}};
};

template <typename T>
using TridiagEigensolverRotMCTest = TridiagEigensolverRotTest<T>;

TYPED_TEST_SUITE(TridiagEigensolverRotMCTest, RealMatrixElementTypes);

template <class T, Device D>
void testApplyGivenRotations(comm::CommunicatorGrid grid, const SizeType m, const SizeType mb,
                             const SizeType idx_begin, const SizeType idx_last,
                             std::vector<di::GivensRotation<T>> rots) {
  // TODO check about "independent" rots?

  using dlaf::eigensolver::internal::applyGivensRotationsToMatrixColumns;

  matrix::Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), {0, 0});

  matrix::Matrix<T, Device::CPU> mat_h(dist);
  matrix::util::set_random(mat_h);

  matrix::test::MatrixLocal<T> mat_loc = matrix::test::allGather(blas::Uplo::General, mat_h, grid);

  {
    matrix::MatrixMirror<T, D, Device::CPU> mat(mat_h);
    applyGivensRotationsToMatrixColumns(grid.rowCommunicator(), idx_begin, idx_last, ex::just(rots),
                                        mat.get());
  }

  // Apply Given Rotations
  for (auto rot : rots) {
    const SizeType n = mat_loc.size().rows();
    T* x = mat_loc.ptr({0, rot.i});
    T* y = mat_loc.ptr({0, rot.j});
    blas::rot(n, x, 1, y, 1, rot.c, rot.s);
  }

  auto result = [&dist = mat_h.distribution(), &mat_local = mat_loc](const GlobalElementIndex& element) {
    const auto tile_index = dist.globalTileIndex(element);
    const auto tile_element = dist.tileElementIndex(element);
    return mat_local.tile_read(tile_index)(tile_element);
  };

  CHECK_MATRIX_NEAR(result, mat_h, m * TypeUtilities<T>::error, m * TypeUtilities<T>::error);
}

TYPED_TEST(TridiagEigensolverRotMCTest, ApplyGivenRotations) {
  for (const auto& grid : this->commGrids()) {
    for (const auto& [m, mb, idx_begin, idx_last, rots] : this->configs) {
      testApplyGivenRotations<TypeParam, Device::CPU>(grid, m, mb, idx_begin, idx_last, rots);
    }
  }
}
