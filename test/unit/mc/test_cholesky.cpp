//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/mc/cholesky.h"

#include "gtest/gtest.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::comm;
using namespace dlaf_test;
using namespace dlaf_test::matrix_test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class CholeskyLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(CholeskyLocalTest, MatrixElementTypes);

template <typename Type>
class CholeskyDistributedTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(CholeskyDistributedTest, MatrixElementTypes);

std::vector<LocalElementSize> square_sizes({{10, 10}, {25, 25}, {12, 12}, {0, 0}});
std::vector<LocalElementSize> rectangular_sizes({{10, 20}, {50, 20}, {0, 10}, {20, 0}});
std::vector<TileElementSize> square_block_sizes({{5, 5}, {20, 20}});
std::vector<TileElementSize> rectangular_block_sizes({{10, 30}, {20, 10}});

GlobalElementSize globalTestSize(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

TYPED_TEST(CholeskyLocalTest, Correctness) {
  // Note: The tile elements are chosen such that:
  // - res_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // a_ij = Sum_k(res_ik * ConjTrans(res)_kj) =
  //      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
  // where k = 0 .. min(i,j)
  // Therefore,
  // a_ij = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(-i+j))
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    if (i < j)
      return TypeUtilities<TypeParam>::element(-9.9, 0.0);

    return TypeUtilities<TypeParam>::polar(std::exp2(-(i + j)) / 3 *
                                               (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                           -i + j);
  };

  // Analytical results
  auto res = [](const TileElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    if (i < j)
      return TypeUtilities<TypeParam>::element(-9.9, 0.0);

    return TypeUtilities<TypeParam>::polar(std::exp2(-std::abs(i - j)), -i + j);
  };

  for (const auto& size : square_sizes) {
    for (const auto& block_size : square_block_sizes) {
      // Matrix to undergo Cholesky decomposition
      Matrix<TypeParam, Device::CPU> mat(size, block_size);
      set(mat, el);

      EXPECT_NO_THROW(cholesky(blas::Uplo::Lower, mat));

      CHECK_MATRIX_NEAR(res, mat, 4 * (mat.size().rows() + 1) * TypeUtilities<TypeParam>::error,
                        4 * (mat.size().rows() + 1) * TypeUtilities<TypeParam>::error);
    }
  }
}

TYPED_TEST(CholeskyLocalTest, NoSquareMatrixException) {
  for (const auto& size : rectangular_sizes) {
    for (const auto& block_size : square_block_sizes) {
      Matrix<TypeParam, Device::CPU> mat(size, block_size);

      EXPECT_THROW(cholesky(blas::Uplo::Lower, mat), std::invalid_argument);
    }
  }
}

TYPED_TEST(CholeskyLocalTest, NoSquareBlockException) {
  for (const auto& size : square_sizes) {
    for (const auto& block_size : rectangular_block_sizes) {
      Matrix<TypeParam, Device::CPU> mat(size, block_size);

      EXPECT_THROW(cholesky(blas::Uplo::Lower, mat), std::invalid_argument);
    }
  }
}

TYPED_TEST(CholeskyDistributedTest, Correctness) {
  // Note: The tile elements are chosen such that:
  // - res_ij = 1 / 2^(|i-j|) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // a_ij = Sum_k(res_ik * ConjTrans(res)_kj) =
  //      = Sum_k(1 / 2^(|i-k| + |j-k|) * exp(I*(-i+j))),
  // where k = 0 .. min(i,j)
  // Therefore,
  // a_ij = (4^(min(i,j)+1) - 1) / (3 * 2^(i+j)) * exp(I*(-i+j))
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    if (i < j)
      return TypeUtilities<TypeParam>::element(-9.9, 0.0);

    return TypeUtilities<TypeParam>::polar(std::exp2(-(i + j)) / 3 *
                                               (std::exp2(2 * (std::min(i, j) + 1)) - 1),
                                           -i + j);
  };

  // Analytical results
  auto res = [](const TileElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    if (i < j)
      return TypeUtilities<TypeParam>::element(-9.9, 0.0);

    return TypeUtilities<TypeParam>::polar(std::exp2(-std::abs(i - j)), -i + j);
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : square_sizes) {
      for (const auto& block_size : square_block_sizes) {
        // Matrix to undergo Cholesky decomposition
        comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                     std::min(1, comm_grid.size().cols() - 1));
        GlobalElementSize sz = globalTestSize(size, comm_grid.size());
        Distribution distribution(sz, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> mat(std::move(distribution));
        set(mat, el);

        EXPECT_NO_THROW(cholesky(comm_grid, blas::Uplo::Lower, mat));

        CHECK_MATRIX_NEAR(res, mat, 4 * (mat.size().rows() + 1) * TypeUtilities<TypeParam>::error,
                          4 * (mat.size().rows() + 1) * TypeUtilities<TypeParam>::error);
      }
    }
  }
}

TYPED_TEST(CholeskyDistributedTest, NoSquareMatrixException) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : rectangular_sizes) {
      for (const auto& block_size : square_block_sizes) {
        comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                     std::min(1, comm_grid.size().cols() - 1));
        GlobalElementSize sz = globalTestSize(size, comm_grid.size());
        Distribution distribution(sz, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> mat(std::move(distribution));

        EXPECT_THROW(cholesky(comm_grid, blas::Uplo::Lower, mat), std::invalid_argument);
      }
    }
  }
}

TYPED_TEST(CholeskyDistributedTest, NoSquareBlockException) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& size : square_sizes) {
      for (const auto& block_size : rectangular_block_sizes) {
        comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                     std::min(1, comm_grid.size().cols() - 1));
        GlobalElementSize sz = globalTestSize(size, comm_grid.size());
        Distribution distribution(sz, block_size, comm_grid.size(), comm_grid.rank(), src_rank_index);
        Matrix<TypeParam, Device::CPU> mat(std::move(distribution));

        EXPECT_THROW(cholesky(comm_grid, blas::Uplo::Lower, mat), std::invalid_argument);
      }
    }
  }
}
