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

template <class T>
using NormT = dlaf::BaseType<T>;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class NormDistributedTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(NormDistributedTest, MatrixElementTypes);

const std::vector<GlobalElementSize> sizes({{10, 10}, {0, 0}});
const std::vector<TileElementSize> block_sizes({{3, 3}, {5, 5}});

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper, blas::Uplo::General});

TYPED_TEST(NormDistributedTest, NormMax) {
  const lapack::Norm norm = lapack::Norm::Max;

  const TypeParam value = dlaf_test::TypeUtilities<TypeParam>::element(13, -13);

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : sizes) {
      for (const auto& block_size : block_sizes) {
        Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        Distribution distribution(size, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> mat(std::move(distribution));

        for (const auto& uplo : blas_uplos) {
          if (blas::Uplo::Lower != uplo)
            continue;

          const NormT<TypeParam> norm_expected = size.isEmpty() ? 0 : std::abs(value);

          {
            auto el_L = [size, value](const GlobalElementIndex& index) {
              if (GlobalElementIndex{size.rows() - 1, 0} == index)
                return value;
              return TypeParam(0);
            };

            set(mat, el_L);

            const NormT<TypeParam> result = Utility<Backend::MC>::norm(comm_grid, norm, uplo, mat);

            SCOPED_TRACE(::testing::Message() << lapack::norm2str(norm) << " Lower Triangular " << size);

            if (Index2D{0, 0} == comm_grid.rank())
              EXPECT_NEAR(norm_expected, result, TypeUtilities<NormT<TypeParam>>::error);
          }

          {
            auto el_D = [size, value](const GlobalElementIndex& index) {
              if (GlobalElementIndex{size.rows() - 1, size.cols() - 1} == index)
                return value;
              return TypeParam(0);
            };

            set(mat, el_D);

            const NormT<TypeParam> result = Utility<Backend::MC>::norm(comm_grid, norm, uplo, mat);

            SCOPED_TRACE(::testing::Message() << lapack::norm2str(norm) << " Diagonal " << size);

            if (Index2D{0, 0} == comm_grid.rank())
              EXPECT_NEAR(norm_expected, result, TypeUtilities<NormT<TypeParam>>::error);
          }
        }
      }
    }
  }
}
