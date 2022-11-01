//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include <gtest/gtest.h>
#include <pika/runtime.hpp>

#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/permutations/general.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

#include "dlaf/matrix/print_csv.h"

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

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> params = {
    // n, nb, i_begin, i_end
    {6, 2, 0, 2},
    {10, 3, 0, 3},
    {17, 5, 0, 3}};

template <class T, Device D, Coord C>
void testDistPermutaitons(comm::CommunicatorGrid grid, SizeType n, SizeType nb, SizeType i_begin,
                          SizeType i_end) {
  const GlobalElementSize size(n, n);
  const TileElementSize block_size(nb, nb);
  // Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));
  Index2D src_rank_index(0, 0);

  Distribution dist(size, block_size, grid.size(), grid.rank(), src_rank_index);

  Matrix<SizeType, Device::CPU> perms_h(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Matrix<T, Device::CPU> mat_in_h(dist);
  Matrix<T, Device::CPU> mat_out_h(dist);

  SizeType index_start = dist.globalElementFromGlobalTileAndTileElement<C>(i_begin, 0);
  SizeType index_finish = dist.globalElementFromGlobalTileAndTileElement<C>(i_end, 0) +
                          dist.tileSize(GlobalTileIndex(i_end, i_end)).get<C>();
  dlaf::matrix::util::set(perms_h, [index_start, index_finish](GlobalElementIndex i) {
    if (index_start > i.row() || i.row() >= index_finish)
      return SizeType(0);

    return index_finish - 1 - i.row();
  });
  dlaf::matrix::util::set(mat_in_h, [](GlobalElementIndex i) {
    return T(i.get<C>()) - T(i.get<orthogonal(C)>()) / T(8);
  });
  dlaf::matrix::util::set0<Backend::MC>(pika::execution::thread_priority::normal, mat_out_h);

  Matrix<T, Device::CPU> mat_print_in(LocalElementSize(n, n), TileElementSize(nb, nb));
  dlaf::matrix::util::set(mat_print_in, [](GlobalElementIndex i) {
    return T(i.get<C>()) - T(i.get<orthogonal(C)>()) / T(8);
  });
  matrix::print(format::csv{}, "INPUT", mat_print_in);
  matrix::print(format::csv{}, "INDEX", perms_h);

  {
    matrix::MatrixMirror<const SizeType, D, Device::CPU> perms(perms_h);
    matrix::MatrixMirror<T, D, Device::CPU> mat_in(mat_in_h);
    matrix::MatrixMirror<T, D, Device::CPU> mat_out(mat_out_h);

    permutations::permute<DefaultBackend_v<D>, D, T, C>(grid, i_begin, i_end, perms.get(), mat_in.get(),
                                                        mat_out.get());
  }

  auto expected_out = [i_begin, i_end, index_start, index_finish, &dist](const GlobalElementIndex i) {
    GlobalTileIndex i_tile = dist.globalTileIndex(i);
    if (i_begin <= i_tile.row() && i_tile.row() <= i_end && i_begin <= i_tile.col() &&
        i_tile.col() <= i_end) {
      GlobalElementIndex i_in(i.get<orthogonal(C)>(), index_finish + index_start - 1 - i.get<C>());
      if constexpr (C == Coord::Row)
        i_in.transpose();
      return T(i_in.get<C>()) - T(i_in.get<orthogonal(C)>()) / T(8);
    }
    return T(0);
  };

  CHECK_MATRIX_EQ(expected_out, mat_out_h);
}

TEST(PermutationsDistTestMC, Columns) {
  using TypeParam = float;
  CommunicatorGrid comm_grid(MPI_COMM_WORLD, 3, 2, common::Ordering::RowMajor);
  // for (const auto& comm_grid : this->commGrids()) {
  for (const auto& [n, nb, i_begin, i_end] : params) {
    testDistPermutaitons<TypeParam, Device::CPU, Coord::Col>(comm_grid, n, nb, i_begin, i_end);
    pika::threads::get_thread_manager().wait();
  }
  //}
}
