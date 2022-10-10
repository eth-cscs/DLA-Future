//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/permutations/general.h"

#include <gtest/gtest.h>
#include <pika/runtime.hpp>

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct PermutationsDistrTestMC : public TestWithCommGrids {};

TYPED_TEST_SUITE(PermutationsDistrTestMC, RealMatrixElementTypes);

const std::vector<std::tuple<SizeType, SizeType>> params = {
    // n, nb, k
    {0, 2},                              // n = 0
    {5, 8}, {34, 34},                    // n <= nb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // n > nb
};

template <class T>
void testMerge(comm::CommunicatorGrid grid, SizeType n, SizeType nb) {
  const GlobalElementSize size(n, n);
  const TileElementSize block_size(nb, nb);
  Index2D src_rank_index(std::max(0, grid.size().rows() - 1), std::min(1, grid.size().cols() - 1));

  Distribution distribution(size, block_size, grid.size(), grid.rank(), src_rank_index);

  Matrix<T, Device::CPU> perms(LocalElementSize(n, 1), TileElementSize(nb, 1));
  Matrix<T, Device::CPU> mat_in(distribution);
  Matrix<T, Device::CPU> mat_out(distribution);

  // each column is filled with an index of it's value
  auto in_el_f = [](const GlobalElementIndex& idx) {
    // TODO
    (void) idx;
    return T(0);
  };

  set(mat_in, in_el_f);
}

TYPED_TEST(PermutationsDistrTestMC, Correctness) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [n, nb] : params) {
      testMerge<TypeParam>(comm_grid, n, nb);
      pika::threads::get_thread_manager().wait();
    }
  }
}
