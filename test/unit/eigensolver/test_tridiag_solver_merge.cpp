//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/util_matrix.h"

#include "gtest/gtest.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::eigensolver::internal;

template <typename Type>
class TridiagEigensolverMergeTest : public ::testing::Test {};
TYPED_TEST_SUITE(TridiagEigensolverMergeTest, RealMatrixElementTypes);

TYPED_TEST(TridiagEigensolverMergeTest, ApplyIndex) {
  SizeType n = 10;
  SizeType nb = 3;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  Matrix<SizeType, Device::CPU> index(sz, bk);
  Matrix<TypeParam, Device::CPU> in(sz, bk);
  Matrix<TypeParam, Device::CPU> out(sz, bk);
  // reverse order: n-1, n-2, ... ,0
  dlaf::matrix::util::set(index, [n](GlobalElementIndex i) { return n - i.row() - 1; });
  // n, n+1, n+2, ..., 2*n - 1
  dlaf::matrix::util::set(in, [n](GlobalElementIndex i) { return TypeParam(n + i.row()); });

  TileCollector tc{0, 3};
  pika::dataflow(applyIndex_o, n, tc.readVec(index), tc.readVec(in), tc.readwriteVec(out));

  // 2*n - 1, 2*n - 2, ..., n
  auto expected_out = [n](GlobalElementIndex i) { return TypeParam(2 * n - 1 - i.row()); };
  CHECK_MATRIX_EQ(expected_out, out);
}

TEST(CopyVector, Index) {
  SizeType n = 10;
  SizeType nb = 3;

  LocalElementSize sz(n, 1);
  TileElementSize bk(nb, 1);

  Matrix<SizeType, Device::CPU> in(sz, bk);
  Matrix<SizeType, Device::CPU> out(sz, bk);
  // reverse order: n-1, n-2, ... ,0
  dlaf::matrix::util::set(in, [](GlobalElementIndex i) { return i.row(); });

  copyVector(0, 3, in, out);

  auto expected_out = [](GlobalElementIndex i) { return i.row(); };
  CHECK_MATRIX_EQ(expected_out, out);
}
