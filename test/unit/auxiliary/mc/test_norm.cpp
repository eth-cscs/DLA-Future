//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/auxiliary/mc.h"

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

const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper, blas::Uplo::General});

// Given a global index of an element, change its value using setter
// A reference to the elemenet is given as parameter to the setter function [void (*setter)(T& element)]
template <class T, class ElementSetter>
void modify_element(Matrix<T, Device::CPU>& matrix, GlobalElementIndex index, ElementSetter set) {
  const auto& distribution = matrix.distribution();

  const GlobalTileIndex tile_index = distribution.globalTileIndex(index);
  if (distribution.rankIndex() != distribution.rankGlobalTile(tile_index))
    return;

  const TileElementIndex index_wrt_local = distribution.tileElementIndex(index);
  matrix(tile_index).then(hpx::util::unwrapping([set, index_wrt_local](auto&& tile) {
    set(tile(index_wrt_local));
  }));
}

// Change the specified value of the matrix, re-compute the norm with given parameters and check if the
// result is the expected one
template <class T>
void set_and_test(CommunicatorGrid comm_grid, Matrix<T, Device::CPU>& matrix, GlobalElementIndex index,
                  T new_value, NormT<T> norm_expected, lapack::Norm norm_type, blas::Uplo uplo) {
  if (index.isIn(matrix.size()))
    modify_element(matrix, index, [new_value](T& element) { element = new_value; });

  const NormT<T> norm = Auxiliary<Backend::MC>::norm(comm_grid, norm_type, uplo, matrix);

  SCOPED_TRACE(::testing::Message() << lapack::norm2str(norm_type) << " " << blas::uplo2str(uplo)
                                    << " changed element=" << index << " in matrix size="
                                    << matrix.size() << " grid_size=" << comm_grid.size());

  if (Index2D{0, 0} == comm_grid.rank())
    EXPECT_NEAR(norm_expected, norm, TypeUtilities<NormT<T>>::error);
}

TYPED_TEST(NormDistributedTest, NormMax) {
  const lapack::Norm norm_type = lapack::Norm::Max;

  const std::vector<GlobalElementSize> sizes({{10, 10}, {0, 0}});
  const std::vector<TileElementSize> block_sizes({{3, 3}, {5, 5}});

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : sizes) {
      for (const auto& block_size : block_sizes) {
        Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        Distribution distribution(size, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> matrix(std::move(distribution));

        for (const auto& uplo : blas_uplos) {
          if (blas::Uplo::Lower != uplo)
            continue;

          dlaf::matrix::util::set_random(matrix);

          const NormT<TypeParam> norm = Auxiliary<Backend::MC>::norm(comm_grid, norm_type, uplo, matrix);

          if (Index2D{0, 0} == comm_grid.rank()) {
            EXPECT_GE(norm, -1);
            EXPECT_LE(norm, +1);
          }

          NormT<TypeParam> norm_current = norm;
          TypeParam new_value;

          // TOP LEFT
          {
            new_value = 100;
            const GlobalElementIndex index{0, 0};
            const NormT<TypeParam> norm_expected = matrix.size().isEmpty() ? 0 : std::abs(new_value);
            norm_current = norm_expected;

            set_and_test(comm_grid, matrix, index, new_value, norm_expected, norm_type, uplo);
          }

          // BOTTOM LEFT
          {
            new_value += 100;
            const GlobalElementIndex index{matrix.size().rows() - 1, 0};
            const NormT<TypeParam> norm_expected = matrix.size().isEmpty() ? 0 : std::abs(new_value);
            norm_current = norm_expected;

            set_and_test(comm_grid, matrix, index, new_value, norm_expected, norm_type, uplo);
          }

          // TOP RIGHT
          {
            new_value += 100;
            const GlobalElementIndex index{0, matrix.size().cols() - 1};
            const NormT<TypeParam> norm_expected = norm_current;

            set_and_test(comm_grid, matrix, index, new_value, norm_expected, norm_type, uplo);
          }

          // BOTTOM RIGHT
          {
            new_value += 100;
            const GlobalElementIndex index{matrix.size().rows() - 1, matrix.size().cols() - 1};
            const NormT<TypeParam> norm_expected = matrix.size().isEmpty() ? 0 : std::abs(new_value);
            norm_current = norm_expected;

            set_and_test(comm_grid, matrix, index, new_value, norm_expected, norm_type, uplo);
          }
        }
      }
    }
  }
}
