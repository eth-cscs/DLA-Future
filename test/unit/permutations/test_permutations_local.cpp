//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <tuple>
#include <vector>

#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/permutations/general.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;

template <typename Type>
class PermutationsTestCPU : public ::testing::Test {};
TYPED_TEST_SUITE(PermutationsTestCPU, RealMatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <typename Type>
class PermutationsTestGPU : public ::testing::Test {};
TYPED_TEST_SUITE(PermutationsTestGPU, RealMatrixElementTypes);
#endif

// The portion of the input matrices rows/columns specified by `i_begin` and `i_end` are placed in
// reverse order into the output matrix.
template <Backend B, Device D, class T, Coord C>
void testPermutations(SizeType n, SizeType nb, SizeType i_begin, SizeType i_end) {
  const matrix::Distribution distr({n, n}, {nb, nb});

  SizeType index_start = i_begin * nb;
  SizeType index_end = std::min(n, i_end * nb);

  Matrix<const SizeType, Device::CPU> perms_h = [n, nb, index_start, index_end]() {
    Matrix<SizeType, Device::CPU> perms_h(LocalElementSize(n, 1), TileElementSize(nb, 1));
    dlaf::matrix::util::set(perms_h, [index_start, index_end](GlobalElementIndex i) {
      if (i.row() < index_start || i.row() >= index_end)
        return SizeType(0);

      return index_end - 1 - i.row();
    });
    return perms_h;
  }();

  Matrix<T, Device::CPU> mat_in_h = [distr]() {
    Matrix<T, Device::CPU> mat_in_h(distr);
    dlaf::matrix::util::set(mat_in_h, [](GlobalElementIndex i) {
      return T(i.get<C>()) - T(i.get<orthogonal(C)>()) / T(8);
    });
    return mat_in_h;
  }();

  Matrix<T, Device::CPU> mat_out_h(distr);
  dlaf::matrix::util::set0<Backend::MC>(pika::execution::thread_priority::normal, mat_out_h);

  {
    matrix::MatrixMirror<const SizeType, D, Device::CPU> perms(perms_h);
    matrix::MatrixMirror<const T, D, Device::CPU> mat_in(mat_in_h);
    matrix::MatrixMirror<T, D, Device::CPU> mat_out(mat_out_h);

    permutations::permute<B, D, T, C>(i_begin, i_end, perms.get(), mat_in.get(), mat_out.get());
  }

  auto expected_out = [i_begin, i_end, index_start, index_end, &distr](const GlobalElementIndex i) {
    GlobalTileIndex i_tile = distr.globalTileIndex(i);
    if (i_begin <= i_tile.row() && i_tile.row() < i_end && i_begin <= i_tile.col() &&
        i_tile.col() < i_end) {
      GlobalElementIndex i_in(i.get<orthogonal(C)>(), index_end + index_start - 1 - i.get<C>());
      if constexpr (C == Coord::Row)
        i_in.transpose();
      return T(i_in.get<C>()) - T(i_in.get<orthogonal(C)>()) / T(8);
    }
    return T(0);
  };

  CHECK_MATRIX_EQ(expected_out, mat_out_h);
}

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    // n, nb, i_begin, i_end
    {10, 3, 0, 4},
    {10, 3, 1, 3},
    {10, 3, 1, 2},
    {10, 10, 0, 1},
    {10, 5, 1, 2},
    // Empty range
    {10, 5, 0, 0},
    {10, 5, 1, 1},
    {10, 5, 2, 2},
};

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

#ifdef DLAF_WITH_GPU
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
