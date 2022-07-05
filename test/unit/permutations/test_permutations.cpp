//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "gtest/gtest.h"

#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/permutations/general.h"
#include "dlaf/permutations/general/impl.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;

template <typename Type>
class PermutationsTestCPU : public ::testing::Test {};
TYPED_TEST_SUITE(PermutationsTestCPU, MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <typename Type>
class PermutationsTestGPU : public ::testing::Test {};
TYPED_TEST_SUITE(PermutationsTestGPU, MatrixElementTypes);
#endif

// The portion of the input matrices rows/columns specified by `i_begin` and `i_end` are placed in
// reverse order into the output matrix.
template <Backend B, Device D, class T, Coord C>
void testPermutations(SizeType n, SizeType nb, SizeType i_begin, SizeType i_end) {
  Matrix<SizeType, Device::CPU> perms(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Matrix<T, Device::CPU> mat_in_h(LocalElementSize(n, n), TileElementSize(nb, nb));
  Matrix<T, Device::CPU> mat_out_h(LocalElementSize(n, n), TileElementSize(nb, nb));

  const matrix::Distribution& distr = mat_out_h.distribution();

  SizeType index_start = distr.globalElementFromGlobalTileAndTileElement<C>(i_begin, 0);
  SizeType index_finish = distr.globalElementFromGlobalTileAndTileElement<C>(i_end, 0) +
                          distr.tileSize(GlobalTileIndex(i_end, i_end)).get<C>();
  dlaf::matrix::util::set(perms, [index_start, index_finish](GlobalElementIndex i) {
    if (index_start > i.row() || i.row() >= index_finish)
      return SizeType(0);

    return index_finish - 1 - i.row();
  });
  dlaf::matrix::util::set(mat_in_h, [](GlobalElementIndex i) {
    return T(i.get<C>() - i.get<orthogonal(C)>()) / T(8);
  });
  dlaf::matrix::util::set0<Backend::MC>(pika::threads::thread_priority::normal, mat_out_h);

  {
    matrix::MatrixMirror<T, D, Device::CPU> mat_in(mat_in_h);
    matrix::MatrixMirror<T, D, Device::CPU> mat_out(mat_out_h);

    permutations::permute<B, D, T, C>(i_begin, i_end, perms, mat_in.get(), mat_out.get());
  }

  auto expected_out = [i_begin, i_end, index_start, index_finish, &distr](const GlobalElementIndex i) {
    GlobalTileIndex i_tile = distr.globalTileIndex(i);
    if (i_begin <= i_tile.row() && i_tile.row() <= i_end && i_begin <= i_tile.col() &&
        i_tile.col() <= i_end) {
      GlobalElementIndex i_in(i.get<orthogonal(C)>(), index_finish + index_start - 1 - i.get<C>());
      if constexpr (C == Coord::Row)
        i_in.transpose();
      return T(i_in.get<C>() - i_in.get<orthogonal(C)>()) / T(8);
    }
    return T(0);
  };

  CHECK_MATRIX_EQ(expected_out, mat_out_h);
}

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    // n, nb, i_begin, i_end
    {10, 3, 0, 3},
    {10, 3, 1, 2},
    {10, 3, 1, 1},
    {10, 10, 0, 0},
    {10, 5, 1, 1}};

TYPED_TEST(PermutationsTestCPU, Columns) {
  for (auto [n, nb, i_begin, i_end] : sizes) {
    testPermutations<Backend::MC, Device::CPU, TypeParam, Coord::Col>(n, nb, i_begin, i_end);
  }
}

TYPED_TEST(PermutationsTestCPU, Rows) {
  for (auto [n, nb, i_begin, i_end] : sizes) {
    testPermutations<Backend::MC, Device::CPU, TypeParam, Coord::Row>(n, nb, i_begin, i_end);
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(PermutationsTestGPU, Columns) {
  for (auto [n, nb, i_begin, i_end] : sizes) {
    testPermutations<Backend::GPU, Device::GPU, TypeParam, Coord::Col>(n, nb, i_begin, i_end);
  }
}

TYPED_TEST(PermutationsTestGPU, Rows) {
  for (auto [n, nb, i_begin, i_end] : sizes) {
    testPermutations<Backend::GPU, Device::GPU, TypeParam, Coord::Row>(n, nb, i_begin, i_end);
  }
}
#endif
