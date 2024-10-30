//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_senders.h>
#include <dlaf_test/util_types.h>

using namespace std::chrono_literals;

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class RetiledMatrixRefLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(RetiledMatrixRefLocalTest, MatrixElementTypes);

template <typename Type>
struct RetiledMatrixRefTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(RetiledMatrixRefTest, MatrixElementTypes);

const std::vector<
    std::tuple<LocalElementSize, TileElementSize, LocalTileSize, GlobalElementIndex, GlobalElementSize>>
    local_sizes_tests({
        // size, tile_size, tiles_per_block, distribution origin, distribution size
        {{8, 8}, {2, 2}, {2, 2}, {0, 0}, {4, 4}},
        {{8, 8}, {2, 2}, {2, 2}, {0, 0}, {8, 4}},
        {{8, 8}, {2, 2}, {2, 2}, {0, 0}, {4, 8}},
//        {{8, 8}, {2, 2}, {2, 2}, {2, 2}, {4, 4}},
        // {{0, 0}, {2, 3}, {2, 2}},
        // {{0, 0}, {2, 3}, {2, 2}},
        // {{3, 0}, {5, 2}, {1, 3}},
        // {{0, 1}, {4, 6}, {1, 1}},
        // {{15, 18}, {2, 3}, {2, 2}},
        // {{6, 6}, {2, 1}, {1, 2}},
        // {{3, 4}, {2, 3}, {2, 2}},
        // {{16, 24}, {2, 3}, {3, 2}},
    });

const std::vector<
    std::tuple<GlobalElementSize, TileElementSize, LocalTileSize, GlobalElementIndex, GlobalElementSize>>
    global_sizes_tests({
        {{8, 8}, {2, 2}, {2, 2}, {0, 0}, {4, 4}},
        {{8, 8}, {2, 2}, {2, 2}, {0, 0}, {8, 4}},
        {{8, 8}, {2, 2}, {2, 2}, {0, 0}, {4, 8}},
        //     {{0, 0}, {2, 3}, {2, 2}, {0, 0}, {0, 0}},
        //     {{3, 0}, {5, 2}, {1, 3}},
        //     {{0, 1}, {4, 6}, {1, 1}},
        //     {{45, 32}, {2, 3}, {2, 2}},
        //     {{6, 15}, {2, 1}, {1, 2}},
        //     {{3, 14}, {2, 3}, {2, 2}},
        //     {{36, 14}, {2, 3}, {3, 2}},
    });

template <typename T, Device D, typename F1, typename F2>
void check(F1 el1, F2 el2, dlaf::matrix::Matrix<T, D>& mat, SubMatrixSpec& spec) {
  auto dist = mat.distribution();

  auto el1_shifted_rows = [&spec, &el1](const GlobalElementIndex& index) {
    auto shifted_idx = GlobalElementIndex{index.row() + spec.size.rows(), index.col()};
    return el1(shifted_idx);
  };

  auto el1_shifted_cols = [&spec, &el1](const GlobalElementIndex& index) {
    auto shifted_idx = GlobalElementIndex{index.row(), index.col() + spec.size.cols()};
    return el1(shifted_idx);
  };

  // TODO: Remove
  DLAF_ASSERT(spec.origin.row() == 0, spec.origin.row());
  DLAF_ASSERT(spec.origin.col() == 0, spec.origin.col());

  // Check modified part (el2)
  MatrixRef<T, D> mat_ref(mat, spec);
  CHECK_MATRIX_EQ(el2, mat_ref);

  // Check unmodified part (el1)
  // Double pass on parts of the matrix is possible, but harmless

  if (dist.size().rows() - spec.size.rows() > 0) {
    SubMatrixSpec spec2{{spec.size.rows(), 0},
                        {dist.size().rows() - spec.size.rows(), dist.size().cols()}};
    MatrixRef<T, D> mat_ref2(mat, spec2);
    CHECK_MATRIX_EQ(el1_shifted_rows, mat_ref2);
  }

  if (dist.size().cols() - spec.size.cols() > 0) {
    SubMatrixSpec spec2{{0, spec.size.cols()},
                        {dist.size().rows(), dist.size().cols() - spec.size.cols()}};
    MatrixRef<T, D> mat_ref2(mat, spec2);
    CHECK_MATRIX_EQ(el1_shifted_cols, mat_ref2);
  }
}

TYPED_TEST(RetiledMatrixRefLocalTest, LocalConstructor) {
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

  for (const auto& [size, tile_size, tiles_per_block, dist_origin, dist_size] : local_sizes_tests) {
    const TileElementSize block_size(tile_size.rows() * tiles_per_block.rows(),
                                     tile_size.cols() * tiles_per_block.cols());

    // Expected distribution of the retiled matrix should match the distribution of the matrix reference
    Distribution expected_distribution({dist_size.rows(), dist_size.cols()}, block_size, tile_size,
                                       {1, 1}, {0, 0}, {0, 0});

    Matrix<Type, Device::CPU> mat(size, block_size);

    // Matrix ref
    SubDistributionSpec spec{dist_origin, dist_size};
    MatrixRef<Type, Device::CPU> mat_ref(mat, spec);

    // Non-const retiled matrix reference
    {
      set(mat, el1);
      CHECK_MATRIX_EQ(el1, mat_ref);
      {
        Matrix<Type, Device::CPU> rt_mat = mat_ref.retiledSubPipeline(tiles_per_block);
        EXPECT_EQ(expected_distribution, rt_mat.distribution());
        CHECK_MATRIX_EQ(el1, rt_mat);

        set(rt_mat, el2);
        CHECK_MATRIX_EQ(el2, rt_mat);
      }

      check(el1, el2, mat, spec);
    }

    // Non-const retiled matrix reference
    {
      set(mat, el1);
      CHECK_MATRIX_EQ(el1, mat_ref);
      {
        Matrix<const Type, Device::CPU> rt_mat = mat_ref.retiledSubPipelineConst(tiles_per_block);
        EXPECT_EQ(expected_distribution, rt_mat.distribution());
        CHECK_MATRIX_EQ(el1, rt_mat);
      }

      check(el1, el1, mat, spec);
    }

    // Const retiled matrix from const matrix
    {
      set(mat, el1);
      Matrix<const Type, Device::CPU>& mat_const = mat;
      MatrixRef<const Type, Device::CPU> mat_ref_const(mat_const, spec);

      CHECK_MATRIX_EQ(el1, mat_ref);
      {
        Matrix<const Type, Device::CPU> rt_mat = mat_ref_const.retiledSubPipelineConst(tiles_per_block);
        EXPECT_EQ(expected_distribution, rt_mat.distribution());
        CHECK_MATRIX_EQ(el1, rt_mat);
      }
      check(el1, el1, mat, spec);
    }
  }
}

TYPED_TEST(RetiledMatrixRefTest, GlobalConstructor) {
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

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& [size, tile_size, tiles_per_block, dist_origin, dist_size] : global_sizes_tests) {
      const TileElementSize block_size(tile_size.rows() * tiles_per_block.rows(),
                                       tile_size.cols() * tiles_per_block.cols());

      // Expected distribution of the retiled matrix should match the distribution of the matrix reference
      Distribution expected_distribution({dist_size.rows(), dist_size.cols()}, block_size, tile_size,
                                         comm_grid.size(), comm_grid.rank(), {0, 0});

      Matrix<Type, Device::CPU> mat(size, block_size, comm_grid);

      // Matrix ref
      SubDistributionSpec spec{dist_origin, dist_size};
      MatrixRef<Type, Device::CPU> mat_ref(mat, spec);

      // Non-const retiled matrix reference
      {
        set(mat, el1);
        CHECK_MATRIX_EQ(el1, mat_ref);
        {
          Matrix<Type, Device::CPU> rt_mat = mat_ref.retiledSubPipeline(tiles_per_block);
          EXPECT_EQ(expected_distribution, rt_mat.distribution());
          CHECK_MATRIX_EQ(el1, rt_mat);

          set(rt_mat, el2);
          CHECK_MATRIX_EQ(el2, rt_mat);
        }

        check(el1, el2, mat, spec);
      }

      // Non-const retiled matrix reference
      {
        set(mat, el1);
        CHECK_MATRIX_EQ(el1, mat_ref);
        {
          Matrix<const Type, Device::CPU> rt_mat = mat_ref.retiledSubPipelineConst(tiles_per_block);
          EXPECT_EQ(expected_distribution, rt_mat.distribution());
          CHECK_MATRIX_EQ(el1, rt_mat);
        }

        check(el1, el1, mat, spec);
      }

      // Const retiled matrix from const matrix
      {
        set(mat, el1);
        Matrix<const Type, Device::CPU>& mat_const = mat;
        MatrixRef<const Type, Device::CPU> mat_ref_const(mat_const, spec);

        CHECK_MATRIX_EQ(el1, mat_ref);
        {
          Matrix<const Type, Device::CPU> rt_mat =
              mat_ref_const.retiledSubPipelineConst(tiles_per_block);
          EXPECT_EQ(expected_distribution, rt_mat.distribution());
          CHECK_MATRIX_EQ(el1, rt_mat);
        }
        check(el1, el1, mat, spec);
      }
    }
  }
}

// TYPED_TEST(RetiledMatrixTest, GlobalConstructor) {
//   using Type = TypeParam;
//
//   auto el1 = [](const GlobalElementIndex& index) {
//     SizeType i = index.row();
//     SizeType j = index.col();
//     return TypeUtilities<Type>::element(i + j / 1024., j - i / 128.);
//   };
//   auto el2 = [](const GlobalElementIndex& index) {
//     SizeType i = index.row();
//     SizeType j = index.col();
//     return TypeUtilities<Type>::element(2. * i + j / 1024., j / 3. - i / 18.);
//   };
//
//   for (auto& comm_grid : this->commGrids()) {
//     for (const auto& [size, tile_size, tiles_per_block] : global_sizes_tests) {
//       const TileElementSize block_size(tile_size.rows() * tiles_per_block.rows(),
//                                        tile_size.cols() * tiles_per_block.cols());
//       Distribution expected_distribution(size, block_size, tile_size, comm_grid.size(), comm_grid.rank(),
//                                          {0, 0});
//
//       Matrix<Type, Device::CPU> mat(size, block_size, comm_grid);
//
//       // Non-const retiled matrix
//       {
//         set(mat, el1);
//
//         {
//           Matrix<Type, Device::CPU> rt_mat = mat.retiledSubPipeline(tiles_per_block);
//           EXPECT_EQ(expected_distribution, rt_mat.distribution());
//           CHECK_MATRIX_EQ(el1, rt_mat);
//
//           set(rt_mat, el2);
//           CHECK_MATRIX_EQ(el2, rt_mat);
//         }
//         CHECK_MATRIX_EQ(el2, mat);
//       }
//
//       // Const retiled matrix from non-const matrix
//       {
//         set(mat, el1);
//
//         {
//           Matrix<const Type, Device::CPU> rt_mat = mat.retiledSubPipelineConst(tiles_per_block);
//           EXPECT_EQ(expected_distribution, rt_mat.distribution());
//           CHECK_MATRIX_EQ(el1, rt_mat);
//         }
//         CHECK_MATRIX_EQ(el1, mat);
//       }
//
//       // Const retiled matrix from const matrix
//       {
//         set(mat, el1);
//         Matrix<const Type, Device::CPU>& mat_const = mat;
//
//         {
//           Matrix<const Type, Device::CPU> rt_mat = mat_const.retiledSubPipelineConst(tiles_per_block);
//           EXPECT_EQ(expected_distribution, rt_mat.distribution());
//           CHECK_MATRIX_EQ(el1, rt_mat);
//         }
//         CHECK_MATRIX_EQ(el1, mat);
//       }
//     }
//   }
// }
