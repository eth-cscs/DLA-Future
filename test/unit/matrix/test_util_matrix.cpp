//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/util_matrix.h"

#include <gtest/gtest.h>
#include <vector>

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

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new dlaf_test::CommunicatorGrid6RanksEnvironment);

template <class T>
class MatrixUtilsTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(MatrixUtilsTest, MatrixElementTypes);

struct TestSizes {
  LocalElementSize size;
  TileElementSize block_size;
};

std::vector<TestSizes> sizes_tests({
    {{0, 0}, {11, 13}},
    {{3, 0}, {1, 2}},
    {{0, 1}, {7, 32}},
    {{15, 18}, {5, 9}},
    {{6, 6}, {2, 2}},
    {{3, 4}, {24, 15}},
    {{16, 24}, {3, 5}},
});

GlobalElementSize globalTestSize(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

TYPED_TEST(MatrixUtilsTest, Set) {
  auto identity = [](const GlobalElementIndex& index) {
    if (index.row() == index.col())
      return 1;
    return 0;
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      dlaf::matrix::util::set(matrix, identity);

      CHECK_MATRIX_EQ(identity, matrix);
    }
  }
}

TYPED_TEST(MatrixUtilsTest, SetRandom) {
  auto zero = [](const GlobalElementIndex& index) {
    return dlaf_test::TypeUtilities<TypeParam>::element(0, 0);
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      dlaf::matrix::util::set_random(matrix);

      CHECK_MATRIX_NEAR(zero, matrix, 0, std::abs(dlaf_test::TypeUtilities<TypeParam>::element(1, 1)));
    }
  }
}

TYPED_TEST(MatrixUtilsTest, SetRandomHermitianPositiveDefinite) {
  std::vector<TestSizes> square_blocks_configs({
      {{0, 0}, {13, 13}},  // square null matrix
      {{26, 26}, {2, 2}},  // square matrix multi block
      {{2, 2}, {6, 6}},    // square matrix single block
  });

  auto globalSquareTestSize = [](const LocalElementSize& size, const Size2D& grid_size) {
    auto k = std::max(grid_size.rows(), grid_size.cols());
    return GlobalElementSize{size.rows() * k, size.cols() * k};
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : square_blocks_configs) {
      GlobalElementSize size = globalSquareTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      auto N = std::max(matrix.size().cols(), matrix.size().rows());
      auto identity_2N = [N](const GlobalElementIndex& index) {
        if (index.row() == index.col())
          return dlaf_test::TypeUtilities<TypeParam>::element(2 * N, 0);
        return dlaf_test::TypeUtilities<TypeParam>::element(0, 0);
      };

      dlaf::matrix::util::set_random_hermitian_positive_definite(matrix);

      CHECK_MATRIX_NEAR(identity_2N, matrix, 0,
                        std::abs(dlaf_test::TypeUtilities<TypeParam>::element(1, 1)));
    }
  }
}
