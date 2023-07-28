//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>

#include <pika/execution.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/sender/transform.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct MatrixRefTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(MatrixRefTest, MatrixElementTypes);

struct TestSubMatrix {
  GlobalElementSize size;
  TileElementSize block_size;
  GlobalElementIndex sub_origin;
  GlobalElementSize sub_size;
};

const std::vector<TestSubMatrix> tests_sub_matrix({
    // Empty matrix
    {{0, 0}, {1, 1}, {0, 0}, {0, 0}},
    // Empty sub-matrices
    {{3, 4}, {3, 4}, {0, 0}, {0, 0}},
    {{3, 4}, {3, 4}, {2, 3}, {0, 0}},
    // Single-block matrix
    {{3, 4}, {3, 4}, {0, 0}, {3, 4}},
    {{3, 4}, {3, 4}, {0, 0}, {2, 1}},
    {{3, 4}, {3, 4}, {1, 2}, {2, 1}},
    {{3, 4}, {8, 6}, {0, 0}, {3, 4}},
    {{3, 4}, {8, 6}, {0, 0}, {2, 1}},
    {{3, 4}, {8, 6}, {1, 2}, {2, 1}},
    // Larger matrices
    {{10, 15}, {5, 5}, {6, 7}, {0, 0}},
    {{10, 15}, {5, 5}, {6, 7}, {0, 0}},
    {{10, 15}, {5, 5}, {1, 2}, {0, 0}},
    {{10, 15}, {5, 5}, {0, 0}, {10, 15}},
    {{10, 15}, {5, 5}, {0, 0}, {10, 15}},
    {{10, 15}, {5, 5}, {0, 0}, {10, 15}},
    {{10, 15}, {5, 5}, {6, 7}, {2, 2}},
    {{10, 15}, {5, 5}, {6, 7}, {4, 7}},
    {{10, 15}, {5, 5}, {1, 2}, {8, 7}},
    {{256, 512}, {32, 16}, {45, 71}, {87, 55}},
});

inline bool indexInSubMatrix(const GlobalElementIndex& index, const SubMatrixSpec& spec) {
  bool r = spec.origin.row() <= index.row() && index.row() < spec.origin.row() + spec.size.rows() &&
           spec.origin.col() <= index.col() && index.col() < spec.origin.col() + spec.size.cols();
  return r;
}

TYPED_TEST(MatrixRefTest, Basic) {
  using Type = TypeParam;
  constexpr Device device = Device::CPU;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : tests_sub_matrix) {
      Matrix<Type, device> mat(test.size, test.block_size, comm_grid);
      Matrix<const Type, device>& mat_const = mat;

      const SubMatrixSpec spec{test.sub_origin, test.sub_size};
      MatrixRef<Type, device> mat_ref(mat, spec);
      MatrixRef<Type, device> mat_const_ref1(mat, spec);
      MatrixRef<const Type, device> mat_const_ref2(mat_const, spec);

      EXPECT_EQ(mat_ref.distribution(), mat_const_ref1.distribution());
      EXPECT_EQ(mat_ref.distribution(), mat_const_ref2.distribution());
      EXPECT_EQ(mat_ref.size(), test.sub_size);
      EXPECT_EQ(mat_ref.blockSize(), mat.blockSize());
      EXPECT_EQ(mat_ref.baseTileSize(), mat.baseTileSize());
      EXPECT_EQ(mat_ref.rankIndex(), mat.rankIndex());
      EXPECT_EQ(mat_ref.commGridSize(), mat.commGridSize());
      if (test.sub_origin.isIn(GlobalElementSize(test.block_size.rows(), test.block_size.cols()))) {
        EXPECT_EQ(mat_ref.sourceRankIndex(), mat.sourceRankIndex());
      }
    }
  }
}

TYPED_TEST(MatrixRefTest, NonConstRefFromNonConstMatrix) {
  using Type = TypeParam;
  constexpr Device device = Device::CPU;
  constexpr Type el_submatrix(1);
  constexpr Type el_border(-1);

  const auto f_el_submatrix = [=](const GlobalElementIndex&) { return el_submatrix; };
  const auto f_el_border = [=](const GlobalElementIndex&) { return el_border; };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : tests_sub_matrix) {
      const SubMatrixSpec spec{test.sub_origin, test.sub_size};
      const auto f_el_full = [=](const GlobalElementIndex& index) {
        return indexInSubMatrix(index, spec) ? el_submatrix : el_border;
      };

      Matrix<Type, device> mat_expected(test.size, test.block_size, comm_grid);
      Matrix<Type, device> mat(test.size, test.block_size, comm_grid);
      MatrixRef<Type, device> mat_ref(mat, spec);

      set(mat_expected, f_el_full);
      set(mat, f_el_border);
      for (const auto& ij_local : iterate_range2d(mat_ref.distribution().localNrTiles())) {
        ex::start_detached(mat_ref.readwrite(ij_local) |
                           dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(),
                                                     [=](const auto& tile) {
                                                       set(tile, el_submatrix);
                                                     }));
      }

      CHECK_MATRIX_EQ(f_el_full, mat);
      CHECK_MATRIX_EQ(f_el_submatrix, mat_ref);
    }
  }
}

TYPED_TEST(MatrixRefTest, ConstRefFromNonConstMatrix) {
  using Type = TypeParam;
  constexpr Device device = Device::CPU;
  constexpr Type el_submatrix(1);
  constexpr Type el_border(-1);

  const auto f_el_submatrix = [=](const GlobalElementIndex&) { return el_submatrix; };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : tests_sub_matrix) {
      const SubMatrixSpec spec{test.sub_origin, test.sub_size};
      const auto f_el_full = [=](const GlobalElementIndex& index) {
        return indexInSubMatrix(index, spec) ? el_submatrix : el_border;
      };

      Matrix<Type, device> mat_expected(test.size, test.block_size, comm_grid);
      Matrix<Type, device> mat(test.size, test.block_size, comm_grid);
      MatrixRef<const Type, device> mat_const_ref(mat, spec);

      set(mat_expected, f_el_full);
      set(mat, f_el_full);

      CHECK_MATRIX_EQ(f_el_full, mat);
      CHECK_MATRIX_EQ(f_el_submatrix, mat_const_ref);
    }
  }
}

TYPED_TEST(MatrixRefTest, ConstRefFromConstMatrix) {
  using Type = TypeParam;
  constexpr Device device = Device::CPU;
  constexpr Type el_submatrix(1);
  constexpr Type el_border(-1);

  const auto f_el_submatrix = [=](const GlobalElementIndex&) { return el_submatrix; };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : tests_sub_matrix) {
      const SubMatrixSpec spec{test.sub_origin, test.sub_size};
      const auto f_el_full = [=](const GlobalElementIndex& index) {
        return indexInSubMatrix(index, spec) ? el_submatrix : el_border;
      };

      Matrix<Type, device> mat_expected(test.size, test.block_size, comm_grid);
      Matrix<Type, device> mat(test.size, test.block_size, comm_grid);
      Matrix<const Type, device>& mat_const = mat;
      MatrixRef<const Type, device> mat_const_ref(mat_const, spec);

      set(mat_expected, f_el_full);
      set(mat, f_el_full);

      CHECK_MATRIX_EQ(f_el_full, mat);
      CHECK_MATRIX_EQ(f_el_submatrix, mat_const_ref);
    }
  }
}
