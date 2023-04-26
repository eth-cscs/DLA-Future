//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/retiled_matrix.h"

#include <vector>

#include <gtest/gtest.h>
#include <pika/future.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace std::chrono_literals;

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;
using pika::unwrapping;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class RetiledMatrixLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(RetiledMatrixLocalTest, MatrixElementTypes);

template <typename Type>
struct RetiledMatrixTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(RetiledMatrixTest, MatrixElementTypes);

const std::vector<std::tuple<LocalElementSize, TileElementSize, LocalTileSize>> local_sizes_tests({
    // size, tile_size, tiles_per_block
    {{0, 0}, {2, 3}, {2, 2}},
    {{3, 0}, {5, 2}, {1, 3}},
    {{0, 1}, {4, 6}, {1, 1}},
    {{15, 18}, {2, 3}, {2, 2}},
    {{6, 6}, {2, 1}, {1, 2}},
    {{3, 4}, {2, 3}, {2, 2}},
    {{16, 24}, {2, 3}, {3, 2}},
});

const std::vector<std::tuple<GlobalElementSize, TileElementSize, LocalTileSize>> global_sizes_tests({
    // size, tile_size, tiles_per_block
    {{0, 0}, {2, 3}, {2, 2}},
    {{3, 0}, {5, 2}, {1, 3}},
    {{0, 1}, {4, 6}, {1, 1}},
    {{45, 32}, {2, 3}, {2, 2}},
    {{6, 15}, {2, 1}, {1, 2}},
    {{3, 14}, {2, 3}, {2, 2}},
    {{36, 14}, {2, 3}, {3, 2}},
});

const std::vector<std::tuple<LocalElementSize, TileElementSize, LocalTileSize>> deps_sizes_tests({
    // size, tile_size, tiles_per_block
    {{10, 10}, {4, 4}, {1, 1}},
    {{12, 12}, {4, 4}, {1, 3}},
    {{10, 10}, {4, 4}, {2, 1}},
    {{12, 12}, {4, 4}, {3, 2}},
    {{4, 4}, {4, 4}, {1, 1}},
});

template <class T, Device D>
void testStaticAPI() {
  using matrix_t = RetiledMatrix<T, D>;

  // RetiledMatrixLike Traits
  using ncT = std::remove_const_t<T>;
  static_assert(std::is_same_v<ncT, typename matrix_t::ElementType>, "wrong ElementType");
  static_assert(std::is_same_v<Tile<ncT, D>, typename matrix_t::TileType>, "wrong TileType");
  static_assert(std::is_same_v<Tile<const T, D>, typename matrix_t::ConstTileType>,
                "wrong ConstTileType");
}

TYPED_TEST(RetiledMatrixLocalTest, StaticAPI) {
  testStaticAPI<TypeParam, Device::CPU>();
  testStaticAPI<TypeParam, Device::GPU>();
}

TYPED_TEST(RetiledMatrixTest, LocalConstructor) {
  using Type = TypeParam;

  auto el1 = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024., j - i / 128.);
  };
  auto el2 = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(2. * i + j / 1024., j / 3. - i / 18.);
  };

  for (const auto& [size, tile_size, tiles_per_block] : local_sizes_tests) {
    const TileElementSize block_size(tile_size.rows() * tiles_per_block.rows(),
                                     tile_size.cols() * tiles_per_block.cols());
    Distribution expected_distribution({size.rows(), size.cols()}, block_size, tile_size, {1, 1}, {0, 0},
                                       {0, 0});

    Matrix<Type, Device::CPU> mat(size, block_size);

    set(mat, el1);
    {
      RetiledMatrix<Type, Device::CPU> rt_mat(mat, tiles_per_block);
      EXPECT_EQ(expected_distribution, rt_mat.distribution());
      CHECK_MATRIX_EQ(el1, rt_mat);

      set(rt_mat, el2);
    }
    CHECK_MATRIX_EQ(el2, mat);
  }
}

TYPED_TEST(RetiledMatrixTest, GlobalConstructor) {
  using Type = TypeParam;

  auto el1 = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024., j - i / 128.);
  };
  auto el2 = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(2. * i + j / 1024., j / 3. - i / 18.);
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& [size, tile_size, tiles_per_block] : global_sizes_tests) {
      const TileElementSize block_size(tile_size.rows() * tiles_per_block.rows(),
                                       tile_size.cols() * tiles_per_block.cols());
      Distribution expected_distribution(size, block_size, tile_size, comm_grid.size(), comm_grid.rank(),
                                         {0, 0});

      Matrix<Type, Device::CPU> mat(size, block_size, comm_grid);

      set(mat, el1);
      {
        RetiledMatrix<Type, Device::CPU> rt_mat(mat, tiles_per_block);
        EXPECT_EQ(expected_distribution, rt_mat.distribution());
        CHECK_MATRIX_EQ(el1, rt_mat);

        set(rt_mat, el2);
      }
      CHECK_MATRIX_EQ(el2, mat);
    }
  }
}

TYPED_TEST(RetiledMatrixTest, Dependencies) {
  using Type = TypeParam;

  for (const auto& [size, tile_size, tiles_per_block] : deps_sizes_tests) {
    const TileElementSize block_size(tile_size.rows() * tiles_per_block.rows(),
                                     tile_size.cols() * tiles_per_block.cols());

    // Note the test assumes that at least one full block is available.
    ASSERT_GE(size.rows(), block_size.rows());
    ASSERT_GE(size.cols(), block_size.cols());

    // Dependencies graph:
    // fut0 - fut1 - shfut2a - fut3 - fut4
    //             \ shfut2b /

    Matrix<Type, Device::CPU> mat(size, block_size);
    auto fut0 = mat(LocalTileIndex{0, 0});
    EXPECT_TRUE(fut0.is_ready());

    decltype(fut0) fut4;
    {
      RetiledMatrix<Type, Device::CPU> rt_mat(mat, tiles_per_block);

      fut4 = mat(LocalTileIndex{0, 0});
      EXPECT_FALSE(fut4.is_ready());

      auto fut1 = rt_mat(LocalTileIndex{0, 0});
      EXPECT_FALSE(fut1.is_ready());

      auto shfut2a = rt_mat.read(LocalTileIndex{0, 0});
      EXPECT_FALSE(shfut2a.is_ready());
      auto shfut2b = rt_mat.read(LocalTileIndex{0, 0});
      EXPECT_FALSE(shfut2b.is_ready());

      auto fut3 = rt_mat(LocalTileIndex{0, 0});
      EXPECT_FALSE(fut3.is_ready());

      rt_mat.done(LocalTileIndex{0, 0});
      EXPECT_FALSE(fut4.is_ready());

      fut0.get();
      EXPECT_TRUE(fut1.is_ready());
      EXPECT_FALSE(shfut2a.is_ready());
      EXPECT_FALSE(shfut2b.is_ready());
      fut1.get();
      EXPECT_TRUE(shfut2a.is_ready());
      EXPECT_TRUE(shfut2b.is_ready());
      shfut2b.get();
      shfut2b = {};
      EXPECT_FALSE(fut3.is_ready());
      shfut2a.get();
      shfut2a = {};
      EXPECT_TRUE(fut3.is_ready());
      fut3.get();

      if (tiles_per_block == LocalTileSize(1, 1))
        EXPECT_TRUE(fut4.is_ready());
      else
        EXPECT_FALSE(fut4.is_ready());
    }

    EXPECT_TRUE(fut4.is_ready());
  }
}
