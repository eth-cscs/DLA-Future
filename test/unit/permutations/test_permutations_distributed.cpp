//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <pika/init.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/permutations/general.h>
#include <dlaf/types.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_generic_lapack.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct PermutationsDistTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(PermutationsDistTestMC, RealMatrixElementTypes);

// n, nb, i_begin, i_end, permutation
// permutation has to be defined using element indices wrt to the range described by [i_begin, i_end)
// tiles and not global indices.
// permutation[i]: mat_out[i] = mat_in[perm[i]]
using testcase_t = std::tuple<SizeType, SizeType, SizeType, SizeType, std::vector<SizeType>>;

// Given matrix size and blocksize, this helper converts a range defined with tile indices
// [i_begin_tile, i_end_tile) into a range defined with element indices [i_begin, i_end)
auto tileToElementRange(SizeType m, SizeType mb, SizeType i_begin_tile, SizeType i_end_tile) {
  const SizeType i_begin = std::max<SizeType>(0, std::min<SizeType>(m - 1, i_begin_tile * mb));
  const SizeType i_end = std::max<SizeType>(0, std::min<SizeType>(m, i_end_tile * mb));
  return std::make_tuple(i_begin, i_end);
}

// Helper that given a geometry generates a "mirror" permutation.
// A mirror permutation means that first becomes last, second becomes before-last, ...
testcase_t mirrorPermutation(SizeType m, SizeType mb, SizeType i_begin_tile, SizeType i_end_tile) {
  const auto [i_begin, i_end] = tileToElementRange(m, mb, i_begin_tile, i_end_tile);

  std::vector<SizeType> perms(to_sizet(i_end - i_begin));
  std::generate(perms.rbegin(), perms.rend(), [n = 0]() mutable { return n++; });

  return {m, mb, i_begin_tile, i_end_tile, perms};
}

// Helper that just checks that given geometry and permutations are compatible.
testcase_t customPermutation(SizeType m, SizeType mb, SizeType i_begin_tile, SizeType i_end_tile,
                             std::vector<SizeType> perms) {
  const auto [i_begin, i_end] = tileToElementRange(m, mb, i_begin_tile, i_end_tile);

  const std::size_t nperms = to_sizet(i_end - i_begin);
  DLAF_ASSERT(perms.size() == nperms, perms.size(), nperms);

  return {m, mb, i_begin_tile, i_end_tile, std::move(perms)};
}

const std::vector<testcase_t> params = {
    // simple setup for a (3, 2) process grid,
    mirrorPermutation(6, 2, 0, 3),
    customPermutation(6, 2, 0, 3, {2, 0, 1, 4, 5, 3}),
    // entire range of tiles is inculded
    mirrorPermutation(10, 3, 0, 4),
    customPermutation(10, 3, 0, 4, {0, 2, 3, 4, 6, 8, 1, 5, 7, 9}),
    customPermutation(10, 3, 0, 4, {8, 9, 3, 5, 2, 7, 1, 4, 0, 6}),
    mirrorPermutation(17, 5, 0, 4),
    // only a subset of processes participate
    mirrorPermutation(10, 3, 1, 3),
    // a single tile matrix
    mirrorPermutation(10, 10, 0, 1),
    // each process has multiple tiles
    mirrorPermutation(31, 6, 1, 4),
    mirrorPermutation(50, 4, 1, 9),
    // empty range
    mirrorPermutation(10, 3, 0, 0),
    mirrorPermutation(10, 3, 1, 1),
    mirrorPermutation(10, 3, 3, 3),
};

template <class T, Device D, Coord C>
void testDistPermutations(comm::CommunicatorGrid& grid, SizeType n, SizeType nb, SizeType i_begin,
                          SizeType i_end, std::vector<SizeType> perms) {
  const GlobalElementSize size(n, n);
  const TileElementSize block_size(nb, nb);
  const Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  const Distribution dist(size, block_size, grid.size(), grid.rank(), src_rank_index);

  const auto [index_start, index_end] = tileToElementRange(n, nb, i_begin, i_end);

  Matrix<const SizeType, Device::CPU> perms_h = [=, index_start = index_start, index_end = index_end] {
    Matrix<SizeType, Device::CPU> perms_h(LocalElementSize(n, 1), TileElementSize(nb, 1));
    dlaf::matrix::util::set(perms_h, [=](GlobalElementIndex i) {
      if (index_start > i.row() || i.row() >= index_end)
        return SizeType(0);

      const SizeType i_window = i.row() - index_start;
      return perms[to_sizet(i_window)];
    });
    return perms_h;
  }();

  auto value_in = [](GlobalElementIndex i) { return T(i.get<C>()) - T(i.get<orthogonal(C)>()) / T(8); };
  Matrix<const T, Device::CPU> mat_in_h = [dist, value_in]() {
    Matrix<T, Device::CPU> mat_in_h(dist);
    dlaf::matrix::util::set(mat_in_h, value_in);
    return mat_in_h;
  }();

  auto value_out = [](GlobalElementIndex i) { return T(i.get<orthogonal(C)>()) - T(i.get<C>()) / T(8); };
  Matrix<T, Device::CPU> mat_out_h(dist);
  dlaf::matrix::util::set(mat_out_h, value_out);

  {
    matrix::MatrixMirror<const SizeType, D, Device::CPU> perms(perms_h);
    matrix::MatrixMirror<const T, D, Device::CPU> mat_in(mat_in_h);
    matrix::MatrixMirror<T, D, Device::CPU> mat_out(mat_out_h);

    permutations::permute<DefaultBackend_v<D>, D, T, C>(grid, i_begin, i_end, perms.get(), mat_in.get(),
                                                        mat_out.get());
  }

  auto expected_out = [=, index_start = index_start](const GlobalElementIndex& i) {
    const GlobalTileIndex i_tile = dist.globalTileIndex(i);
    if (i_begin <= i_tile.row() && i_tile.row() < i_end && i_begin <= i_tile.col() &&
        i_tile.col() < i_end) {
      const std::size_t i_window = to_sizet(i.get<C>() - index_start);
      GlobalElementIndex i_in(i.get<orthogonal(C)>(), index_start + perms[i_window]);
      if constexpr (C == Coord::Row)
        i_in.transpose();

      return value_in(i_in);
    }
    return value_out(i);
  };

  CHECK_MATRIX_EQ(expected_out, mat_out_h);
}

TYPED_TEST(PermutationsDistTestMC, Columns) {
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& [n, nb, i_begin, i_end, perms] : params) {
      testDistPermutations<TypeParam, Device::CPU, Coord::Col>(comm_grid, n, nb, i_begin, i_end, perms);
      pika::wait();
    }
  }
}

TYPED_TEST(PermutationsDistTestMC, Rows) {
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& [n, nb, i_begin, i_end, perms] : params) {
      testDistPermutations<TypeParam, Device::CPU, Coord::Row>(comm_grid, n, nb, i_begin, i_end, perms);
      pika::wait();
    }
  }
}
