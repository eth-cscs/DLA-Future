//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/tridiag_solver.h"

#include "gtest/gtest.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

template <typename Type>
class TridiagEigensolverTest : public ::testing::Test {};

using namespace dlaf::test;

TYPED_TEST_SUITE(TridiagEigensolverTest, MatrixElementTypes);

TYPED_TEST(TridiagEigensolverTest, CorrectnessLocal) {
  using namespace dlaf;
  using RealParam = BaseType<TypeParam>;

  SizeType n = 10;
  SizeType nb = 2;

  matrix::Matrix<RealParam, Device::CPU> mat_a(LocalElementSize(n, 2), TileElementSize(nb, 2));
  matrix::Matrix<TypeParam, Device::CPU> mat_ev(LocalElementSize(n, n), TileElementSize(nb, nb));

  // Tridiagonal matrix : 1D Laplacian
  auto mat_a_fn = [](GlobalElementIndex el) {
    if (el.col() == 0)
      // diagonal
      return RealParam(-1);
    else
      // off-diagoanl
      return RealParam(2);
  };
  matrix::util::set(mat_a, std::move(mat_a_fn));

  eigensolver::tridiagSolver<Backend::MC>(mat_a, mat_ev);

  // TODO: checks
}
