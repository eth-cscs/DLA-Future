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

TYPED_TEST_SUITE(NormMaxDistributedTest, MatrixElementTypes);

const std::vector<LocalElementSize> sizes({{10, 10}, {25, 25}, {12, 12}, {0, 0}});
const std::vector<TileElementSize> block_sizes({{3, 3}, {5, 5}});

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

TYPED_TEST(NormMaxDistributedTest, Correctness) {
  using NormT = dlaf::BaseType<TypeParam>;
  const TypeParam max_value = dlaf_test::TypeUtilities<TypeParam>::element(13, -13);

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : sizes) {
      for (const auto& block_size : block_sizes) {
        Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        GlobalElementSize sz = globalTestSize(size);
        Distribution distribution(sz, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> mat(std::move(distribution));

        const blas::Uplo uplo = blas::Uplo::Lower;

        const NormT norm_expected = sz.isEmpty() ? 0 : std::abs(max_value);

        {
          auto el_L = [size = sz, max_value](const GlobalElementIndex& index) {
            if (GlobalElementIndex{size.rows() - 1, 0} == index)
              return max_value;
            return TypeParam(0);
          };

          set(mat, el_L);

          const dlaf::BaseType<TypeParam> result =
              Utility<Backend::MC>::norm(comm_grid, lapack::Norm::Max, uplo, mat);

          SCOPED_TRACE(::testing::Message() << "Max in Lower Triangular " << sz);

          if (Index2D{0, 0} == comm_grid.rank())
            EXPECT_NEAR(norm_expected, result, TypeUtilities<NormT>::error);
        }

        // TODO max in upper does not make sense if checking in lower
        //{
        //  if (Index2D{0, 0} == comm_grid.rank())
        //    SCOPED_TRACE("Max in Upper Triangular");

        //  auto el_U = [size=sz, max_value](const GlobalElementIndex& index) {
        //    if (GlobalElementIndex{0, size.cols() - 1} == index)
        //      return max_value;
        //    return TypeParam(0);
        //  };

        //  set(mat, el_U);

        //  const dlaf::BaseType<TypeParam> result =
        //    Utility<Backend::MC>::norm(comm_grid, lapack::Norm::Max, blas::Uplo::Lower, mat);

        //  if (Index2D{0, 0} == comm_grid.rank())
        //    EXPECT_NEAR(norm_expected, result, TypeUtilities<NormT>::error);
        //}

        {
          auto el_D = [size = sz, max_value](const GlobalElementIndex& index) {
            if (GlobalElementIndex{size.rows() - 1, size.cols() - 1} == index)
              return max_value;
            return TypeParam(0);
          };

          set(mat, el_D);

          const dlaf::BaseType<TypeParam> result =
              Utility<Backend::MC>::norm(comm_grid, lapack::Norm::Max, uplo, mat);

          SCOPED_TRACE(::testing::Message() << "Max on Diagonal " << sz);

          if (Index2D{0, 0} == comm_grid.rank())
            EXPECT_NEAR(norm_expected, result, TypeUtilities<NormT>::error);
        }
      }
    }
  }
}
