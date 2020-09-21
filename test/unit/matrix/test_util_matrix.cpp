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

#include <vector>

#include <gtest/gtest.h>
#include <hpx/local/future.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/basic.h"
#include "dlaf/matrix.h"

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;

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

const std::vector<TestSizes> sizes_tests({
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
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      auto linear_matrix = [size = matrix.size()](const GlobalElementIndex& index) {
        auto linear_index = common::computeLinearIndex<int>(common::Ordering::RowMajor, index, size);
        return TypeUtilities<TypeParam>::element(linear_index, linear_index);
      };

      dlaf::matrix::util::set(matrix, linear_matrix);

      CHECK_MATRIX_EQ(linear_matrix, matrix);
    }
  }
}

TYPED_TEST(MatrixUtilsTest, SetRandom) {
  auto zero = [](const GlobalElementIndex&) {
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

      CHECK_MATRIX_NEAR(zero, matrix, 0, 1);
    }
  }
}

template <class T>
void check_is_hermitian(dlaf::Matrix<const T, Device::CPU>& matrix,
                        dlaf::comm::CommunicatorGrid comm_grid) {
  const auto& distribution = matrix.distribution();
  const auto current_rank = distribution.rankIndex();

  for (auto j = 0; j < matrix.nrTiles().cols(); ++j) {
    for (auto i = 0; i <= j; ++i) {
      const GlobalTileIndex index_tile_original{i, j};
      const auto owner_original = distribution.rankGlobalTile(index_tile_original);

      const GlobalTileIndex index_tile_transposed{j, i};
      const auto owner_transposed = distribution.rankGlobalTile(index_tile_transposed);

      if (current_rank != owner_original && current_rank != owner_transposed)
        continue;

      if (current_rank == owner_original) {
        const auto& tile_original = matrix.read(index_tile_original).get();
        hpx::shared_future<dlaf::Tile<const T, Device::CPU>> tile_transposed;
        auto size_tile_transposed = transposed(tile_original.size());

        if (current_rank == owner_transposed) {
          tile_transposed = matrix.read(index_tile_transposed);
        }
        else {
          dlaf::Tile<T, Device::CPU> workspace(size_tile_transposed,
                                               dlaf::memory::MemoryView<T, Device::CPU>(
                                                   dlaf::util::size_t::mul(size_tile_transposed.rows(),
                                                                           size_tile_transposed.cols())),
                                               size_tile_transposed.rows());

          // recv from owner_transposed
          const auto sender_rank = comm_grid.rankFullCommunicator(owner_transposed);
          dlaf::comm::sync::receive_from(sender_rank, comm_grid.fullCommunicator(), workspace);

          tile_transposed =
              hpx::make_ready_future<dlaf::Tile<const T, Device::CPU>>(std::move(workspace));
        }

        auto transposed_conj_tile = [&tile_original](const TileElementIndex& index) {
          return dlaf::conj(tile_original({index.col(), index.row()}));
        };

        CHECK_TILE_NEAR(transposed_conj_tile, tile_transposed.get(), dlaf_test::TypeUtilities<T>::error,
                        dlaf_test::TypeUtilities<T>::error);
      }
      else if (current_rank == owner_transposed) {
        // send to owner_original
        auto receiver_rank = comm_grid.rankFullCommunicator(owner_original);
        dlaf::comm::sync::send_to(receiver_rank, comm_grid.fullCommunicator(),
                                  matrix.read(index_tile_transposed).get());
      }
    }
  }
}

TYPED_TEST(MatrixUtilsTest, SetRandomHermitianPositiveDefinite) {
  std::vector<TestSizes> square_blocks_configs({
      {{0, 0}, {13, 13}},  // square null matrix
      {{5, 5}, {26, 26}},  // square matrix single block
      {{9, 9}, {3, 3}},    // square matrix multi block "full-tile"
      {{13, 13}, {3, 3}},  // square matrix multi block
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

      auto N = matrix.size().rows();
      auto identity_2N = [N](const GlobalElementIndex& index) {
        if (index.row() == index.col())
          return dlaf_test::TypeUtilities<TypeParam>::element(2 * N, 0);
        return dlaf_test::TypeUtilities<TypeParam>::element(0, 0);
      };

      dlaf::matrix::util::set_random_hermitian_positive_definite(matrix);

      CHECK_MATRIX_NEAR(identity_2N, matrix, 0, 1);

      check_is_hermitian(matrix, comm_grid);
    }
  }
}
