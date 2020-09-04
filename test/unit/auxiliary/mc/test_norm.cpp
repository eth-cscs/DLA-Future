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

#include <limits>

#include <gtest/gtest.h>
#include <hpx/include/util.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"
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

const std::vector<lapack::Norm> lapack_norms({lapack::Norm::Fro, lapack::Norm::Inf, lapack::Norm::Max,
                                              lapack::Norm::One, lapack::Norm::Two});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper, blas::Uplo::General});

TYPED_TEST(NormDistributedTest, EmptyMatrices) {
  const std::vector<GlobalElementSize> sizes({{13, 0}, {0, 13}, {0, 0}});
  const std::vector<TileElementSize> block_sizes({{3, 3}, {5, 5}});

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : sizes) {
      for (const auto& block_size : block_sizes) {
        Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        Distribution distribution(size, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> matrix(std::move(distribution));

        for (const auto& norm_type : lapack_norms) {
          for (const auto& uplo : blas_uplos) {
            const NormT<TypeParam> norm =
                Auxiliary<Backend::MC>::norm(comm_grid, {0, 0}, norm_type, uplo, matrix);

            if (Index2D{0, 0} == comm_grid.rank()) {
              EXPECT_NEAR(0, norm, std::numeric_limits<NormT<TypeParam>>::epsilon());
            }
          }
        }
      }
    }
  }
}

// Given a global index of an element, set it with given value
template <class T>
void modify_element(Matrix<T, Device::CPU>& matrix, GlobalElementIndex index, const T value) {
  const auto& distribution = matrix.distribution();

  const GlobalTileIndex tile_index = distribution.globalTileIndex(index);
  if (distribution.rankIndex() != distribution.rankGlobalTile(tile_index))
    return;

  const TileElementIndex index_wrt_local = distribution.tileElementIndex(index);
  matrix(tile_index).then(hpx::util::unwrapping([value, index_wrt_local](auto&& tile) {
    tile(index_wrt_local) = value;
  }));
}

// Change the specified value of the matrix, re-compute the norm with given parameters and check if the
// result is the expected one
template <class T>
void set_and_test(CommunicatorGrid comm_grid, comm::Index2D rank, Matrix<T, Device::CPU>& matrix,
                  GlobalElementIndex index, const T new_value, const NormT<T> norm_expected,
                  lapack::Norm norm_type, blas::Uplo uplo) {
  if (index.isIn(matrix.size()))
    modify_element(matrix, index, new_value);

  const NormT<T> norm = Auxiliary<Backend::MC>::norm(comm_grid, rank, norm_type, uplo, matrix);

  SCOPED_TRACE(::testing::Message() << "norm=" << lapack::norm2str(norm_type)
                                    << " uplo=" << blas::uplo2str(uplo) << " changed element=" << index
                                    << " in matrix size=" << matrix.size()
                                    << " grid_size=" << comm_grid.size() << " rank=" << rank);

  if (rank == comm_grid.rank()) {
    EXPECT_NEAR(norm_expected, norm, norm * std::numeric_limits<NormT<T>>::epsilon());
  }
}

TYPED_TEST(NormDistributedTest, NormMax) {
  const lapack::Norm norm_type = lapack::Norm::Max;

  const std::vector<GlobalElementSize> sizes({{10, 10}});
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

          dlaf::matrix::util::set_random_hermitian(matrix);

          const Index2D rank_result{comm_grid.size().rows() - 1, comm_grid.size().cols() - 1};
          const NormT<TypeParam> norm =
              Auxiliary<Backend::MC>::norm(comm_grid, rank_result, norm_type, uplo, matrix);

          if (rank_result == comm_grid.rank()) {
            EXPECT_GE(norm, -1);
            EXPECT_LE(norm, +1);
          }

          SizeType nrows = matrix.size().rows();
          SizeType ncols = matrix.size().cols();

          std::vector<GlobalElementIndex> test_indices{{0, 0}, {nrows - 1, ncols - 1}};

          if (blas::Uplo::Lower == uplo || blas::Uplo::General == uplo) {
            test_indices.emplace_back(nrows - 1, 0);  // bottom left
          }
          else if (blas::Uplo::Upper == uplo || blas::Uplo::General == uplo) {
            test_indices.emplace_back(0, ncols - 1);  // top right
          }
          else {
            FAIL() << "this should not be reached";
          }

          TypeParam new_value = TypeUtilities<TypeParam>::element(13.13, 26.26);

          for (const auto& test_index : test_indices) {
            new_value = new_value + TypeUtilities<TypeParam>::element(26.05, 20.10);

            const NormT<TypeParam> norm_expected = std::abs(new_value);

            set_and_test(comm_grid, rank_result, matrix, test_index, new_value, norm_expected, norm_type,
                         uplo);
          }
        }
      }
    }
  }
}
