//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/utility/mc.h"

#include <gtest/gtest.h>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class NormMaxDistributedTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

using COMPLEX_DOES_NOT_WORK =
    ::testing::Types<float, double>;  // TODO fix for complex MatrixElementTypes
TYPED_TEST_SUITE(NormMaxDistributedTest, COMPLEX_DOES_NOT_WORK);

const std::vector<LocalElementSize> square_sizes({{10, 10}, {25, 25}, {12, 12}, {0, 0}});
const std::vector<TileElementSize> square_block_sizes({{3, 3}, {5, 5}});

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

TYPED_TEST(NormMaxDistributedTest, Correctness) {
  auto el = [](const GlobalElementIndex&) { return 0; };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : square_sizes) {
      for (const auto& block_size : square_block_sizes) {
        Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        GlobalElementSize sz = globalTestSize(size);
        Distribution distribution(sz, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> mat(std::move(distribution));

        set(mat, el);

        const TypeParam result = Utility<Backend::MC>::norm_max(comm_grid, blas::Uplo::Lower, mat);

        EXPECT_NEAR(0, result, TypeUtilities<TypeParam>::error);
      }
    }
  }
}
