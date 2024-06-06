//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <utility>
#include <vector>

#include <pika/execution.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/sync/basic.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;

using pika::execution::thread_priority;
using pika::execution::experimental::start_detached;
using pika::this_thread::experimental::sync_wait;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <class T>
struct MatrixUtilsTest : public TestWithCommGrids {};

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

GlobalElementSize globalTestSize(const LocalElementSize& size, const comm::Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

GlobalElementSize globalSquareTestSize(const LocalElementSize& size, const comm::Size2D& grid_size) {
  auto k = std::max(grid_size.rows(), grid_size.cols());
  return GlobalElementSize{size.rows() * k, size.cols() * k};
}

TYPED_TEST(MatrixUtilsTest, Set0) {
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      auto null_matrix = [](const GlobalElementIndex&) { return TypeParam(0); };

      matrix::util::set0<Backend::MC>(thread_priority::normal, matrix);

      CHECK_MATRIX_EQ(null_matrix, matrix);
    }
  }
}

TYPED_TEST(MatrixUtilsTest, Set) {
  for (auto& comm_grid : this->commGrids()) {
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

      matrix::util::set(matrix, linear_matrix);

      CHECK_MATRIX_EQ(linear_matrix, matrix);
    }
  }
}

TYPED_TEST(MatrixUtilsTest, SetRandom) {
  auto zero = [](const GlobalElementIndex&) { return TypeUtilities<TypeParam>::element(0, 0); };

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      matrix::util::set_random(matrix);

      CHECK_MATRIX_NEAR(zero, matrix, 0, 1);
    }
  }
}

template <class T>
void check_is_hermitian(Matrix<const T, Device::CPU>& matrix, comm::CommunicatorGrid& comm_grid) {
  using dlaf::util::size_t::mul;
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
        const auto tile_original = sync_wait(matrix.read(index_tile_original));
        const auto size_tile_transposed = transposed(tile_original.get().size());

        auto transposed_conj_tile = [&tile_original =
                                         tile_original.get()](const TileElementIndex& index) {
          return dlaf::conj(tile_original({index.col(), index.row()}));
        };

        if (current_rank == owner_transposed) {
          auto tile_transposed = sync_wait(matrix.read(index_tile_transposed));
          CHECK_TILE_NEAR(transposed_conj_tile, tile_transposed.get(), TypeUtilities<T>::error,
                          TypeUtilities<T>::error);
        }
        else {
          Tile<T, Device::CPU> tile_transposed(
              size_tile_transposed,
              memory::MemoryView<T, Device::CPU>(size_tile_transposed.linear_size()),
              size_tile_transposed.rows());

          // recv from owner_transposed
          const auto sender_rank = comm_grid.rankFullCommunicator(owner_transposed);
          comm::sync::receive_from(sender_rank, comm_grid.fullCommunicator(), tile_transposed);

          CHECK_TILE_NEAR(transposed_conj_tile, tile_transposed, TypeUtilities<T>::error,
                          TypeUtilities<T>::error);
        }
      }
      else if (current_rank == owner_transposed) {
        // send to owner_original
        auto tile_transposed = sync_wait(matrix.read(index_tile_transposed));
        auto receiver_rank = comm_grid.rankFullCommunicator(owner_original);
        comm::sync::send_to(receiver_rank, comm_grid.fullCommunicator(), tile_transposed.get());
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

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : square_blocks_configs) {
      GlobalElementSize size = globalSquareTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      auto N = matrix.size().rows();
      auto identity_2N = [N](const GlobalElementIndex& index) {
        if (index.row() == index.col())
          return TypeUtilities<TypeParam>::element(2 * N, 0);
        return TypeUtilities<TypeParam>::element(0, 0);
      };

      matrix::util::set_random_hermitian_positive_definite(matrix);

      CHECK_MATRIX_NEAR(identity_2N, matrix, 0, 1);

      check_is_hermitian(matrix, comm_grid);
    }
  }
}

TYPED_TEST(MatrixUtilsTest, SetRandomNonZeroDiagonal) {
  std::vector<TestSizes> square_blocks_configs({
      {{0, 0}, {13, 13}},  // square null matrix
      {{5, 5}, {26, 26}},  // square matrix single block
      {{9, 9}, {3, 3}},    // square matrix multi block "full-tile"
      {{13, 13}, {3, 3}},  // square matrix multi block
  });

  auto zero = [](const GlobalElementIndex&) { return TypeUtilities<TypeParam>::element(0, 0); };

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : square_blocks_configs) {
      GlobalElementSize size = globalSquareTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<TypeParam, Device::CPU> mem(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> matrix(std::move(distribution), layout, mem());

      matrix::util::set_random_non_zero_diagonal(matrix);

      CHECK_MATRIX_NEAR(zero, matrix, 0, 1);

      for (auto j = 0; j < matrix.nrTiles().cols(); ++j) {
        GlobalTileIndex jj{j, j};
        if (matrix.distribution().rankGlobalTile(jj) == comm_grid.rank()) {
          auto tile = pika::this_thread::experimental::sync_wait(matrix.readwrite(jj));
          for (auto j_el = 0; j_el < tile.size().rows(); ++j_el) {
            // 0.099 instead of 0.1 to account for rounding.
            EXPECT_LE(BaseType<TypeParam>{0.099f}, std::abs(tile({j_el, j_el})));
          }
        }
      }
    }
  }
}

template <typename Type>
struct PanelUtilsTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(PanelUtilsTest, MatrixElementTypes);

struct config_t {
  const GlobalElementSize sz;
  const TileElementSize blocksz;
  const GlobalTileIndex offset;
};

std::vector<config_t> test_params{
    {{0, 0}, {3, 3}, {0, 0}},  // empty matrix
    {{26, 13}, {3, 3}, {1, 2}},
};

template <class TypeParam, Coord panel_axis>
void testSet0(const config_t& cfg, const comm::CommunicatorGrid& comm_grid) {
  constexpr Coord coord1D = orthogonal(panel_axis);

  Distribution dist(cfg.sz, cfg.blocksz, comm_grid.size(), comm_grid.rank(), {0, 0});
  Panel<panel_axis, TypeParam, dlaf::Device::CPU> panel(dist, cfg.offset);

  auto null_tile = [](const TileElementIndex&) { return TypeParam(0); };

  for (SizeType head = cfg.offset.get<coord1D>(), tail = dist.nrTiles().get(coord1D); head <= tail;
       ++head, --tail) {
    panel.setRange(GlobalTileIndex(coord1D, head), GlobalTileIndex(coord1D, tail));

    for (const auto& idx : panel.iteratorLocal())
      start_detached(dlaf::internal::whenAllLift(blas::Uplo::General, TypeParam(1), TypeParam(1),
                                                 panel.readwrite(idx)) |
                     tile::laset(dlaf::internal::Policy<dlaf::Backend::MC>()));

    matrix::util::set0<Backend::MC>(thread_priority::normal, panel);

    for (const auto& idx : panel.iteratorLocal())
      CHECK_TILE_EQ(null_tile, sync_wait(panel.read(idx)).get());

    panel.reset();
  }
}

TYPED_TEST(PanelUtilsTest, Set0) {
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& cfg : test_params) {
      testSet0<TypeParam, Coord::Col>(cfg, comm_grid);
      testSet0<TypeParam, Coord::Row>(cfg, comm_grid);
    }
  }
}
