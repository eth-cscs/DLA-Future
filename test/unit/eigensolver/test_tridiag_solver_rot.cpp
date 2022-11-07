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

#include "dlaf/communication/communicator.h"
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
  using GRot = di::GivensRotation<T>;

  struct config_t {
    SizeType m;
    SizeType mb;
    SizeType i_begin;
    SizeType i_last;
    std::vector<GRot> rots;
  };

  const std::vector<config_t> configs{
      // range with one-sided margin
      {9, 3, 1, 2, {GRot{3, 8, cos(2.6f), sin(2.6f)}}},
      {9, 3, 0, 1, {GRot{2, 4, cos(2.6f), sin(2.6f)}}},
      // range fully in-bound
      {12, 3, 1, 2, {GRot{3, 8, cos(2.6f), sin(2.6f)}}},
      // incomplete tile
      {8, 3, 1, 2, {GRot{3, 7, cos(2.6f), sin(2.6f)}}},
      // full-range, multiple rotations
      {9, 3, 0, 2, {GRot{0, 8, cos(2.6f), sin(2.6f)}, GRot{0, 2, cos(2.6f), sin(2.6f)}}},
      // range fully in-bound, independent rotations from same tiles
      {12, 3, 1, 2, {GRot{3, 8, cos(2.6f), sin(2.6f)}, GRot{4, 7, cos(2.6f), sin(2.6f)}}},
      // range fully in-bound, non-independent rotations, between same pair of tiles
      {12, 3, 1, 2, {GRot{3, 8, cos(2.6f), sin(2.6f)}, GRot{3, 7, cos(2.6f), sin(2.6f)}}},
      {12, 3, 1, 2, {GRot{3, 8, cos(2.6f), sin(2.6f)}, GRot{4, 8, cos(2.6f), sin(2.6f)}}},
  };
};

template <typename T>
using TridiagEigensolverRotMCTest = TridiagEigensolverRotTest<T>;

TYPED_TEST_SUITE(TridiagEigensolverRotMCTest, RealMatrixElementTypes);

template <class T, Device D>
void testApplyGivenRotations(comm::CommunicatorGrid grid, const SizeType m, const SizeType mb,
                             const SizeType idx_begin, const SizeType idx_last,
                             std::vector<di::GivensRotation<T>> rots) {
  using dlaf::eigensolver::internal::applyGivensRotationsToMatrixColumns;

  constexpr comm::IndexT_MPI tag = 0;
  matrix::Distribution dist({m, m}, {mb, mb}, grid.size(), grid.rank(), {0, 0});

  matrix::Matrix<T, Device::CPU> mat_h(dist);
  matrix::util::set_random(mat_h);

  matrix::test::MatrixLocal<T> mat_loc = matrix::test::allGather(blas::Uplo::General, mat_h, grid);

  {
    matrix::MatrixMirror<T, D, Device::CPU> mat(mat_h);
    applyGivensRotationsToMatrixColumns(grid.rowCommunicator(), tag, idx_begin, idx_last, ex::just(rots),
                                        mat.get());
  }

  // Apply Given Rotations
  const SizeType n = std::min((idx_last + 1) * mb, m) - idx_begin * mb;

  for (auto rot : rots) {
    T* x = mat_loc.ptr({idx_begin * mb, rot.i});
    T* y = mat_loc.ptr({idx_begin * mb, rot.j});
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
