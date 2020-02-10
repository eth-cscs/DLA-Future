//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/mc/triangular_solve_local.h"

#include <exception>
#include <functional>
#include <sstream>
#include <tuple>
#include "../test_blas_tile/test_trsm.h"
#include "gtest/gtest.h"
#include "dlaf/matrix.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});
std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

template <typename Type>
class TriangularSolveLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(TriangularSolveLocalTest, MatrixElementTypes);

std::vector<LocalElementSize> square_sizes(
    {{2, 2}, {3, 3}, {4, 4}, {6, 6}, {10, 10}, {25, 25}, {15, 15}, {0, 0}});
std::vector<LocalElementSize> rectangular_sizes({{12, 20}, {50, 20}, {0, 12}, {20, 0}});

std::vector<unsigned int> col_b({{1}, {3}, {10}, {20}});

std::vector<TileElementSize> square_block_sizes({{2, 2}, {3, 3}, {5, 5}});
std::vector<TileElementSize> rectangular_block_sizes({{12, 30}, {20, 12}});

std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {0, 0, 1, 1},                                                // m, n = 0
    {0, 2, 1, 2}, {7, 0, 2, 1},                                  // m = 0 or n = 0
    {2, 2, 5, 5}, {10, 10, 2, 3}, {7, 7, 3, 2},                  // m = n
    {3, 2, 7, 7}, {12, 3, 5, 5},  {7, 6, 3, 2}, {15, 7, 3, 5},   // m > n
    {2, 3, 7, 7}, {4, 13, 5, 5},  {7, 8, 2, 9}, {19, 25, 6, 5},  // m < n
};

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T>
void testTriangularSolve(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                         SizeType m, SizeType n, SizeType mb, SizeType nb) {
  std::function<T(const GlobalElementIndex&)> el_op_a, el_b, res_b;

  LocalElementSize size_a(m, m);
  TileElementSize block_size_a(mb, mb);

  if (side == blas::Side::Right) {
    size_a = {n, n};
    block_size_a = {nb, nb};
  }

  Matrix<T, Device::CPU> mat_a(size_a, block_size_a);

  LocalElementSize size_b(m, n);
  TileElementSize block_size_b(mb, nb);
  Matrix<T, Device::CPU> mat_b(size_b, block_size_b);

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) =
        testTrsmElementFunctionsLeft<GlobalElementIndex, T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) =
        testTrsmElementFunctionsRight<GlobalElementIndex, T>(uplo, op, diag, alpha, n);

  set(mat_a, el_op_a, op);
  set(mat_b, el_b);

  triangular_solve(side, uplo, op, diag, alpha, mat_a, mat_b);

  CHECK_MATRIX_NEAR(res_b, mat_b, 40 * (mat_b.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_b.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(TriangularSolveLocalTest, Correctness) {
  SizeType m, n, mb, nb;

  for (auto diag : blas_diags) {
    for (auto op : blas_ops) {
      for (auto side : blas_sides) {
        for (auto uplo : blas_uplos) {
          for (auto sz : sizes) {
            std::tie(m, n, mb, nb) = sz;
            TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

            testTriangularSolve(side, uplo, op, diag, alpha, m, n, mb, nb);
          }
        }
      }
    }
  }
}

TYPED_TEST(TriangularSolveLocalTest, MatrixNotSquareException) {
  for (auto diag : blas_diags) {
    for (auto op : blas_ops) {
      for (auto side : blas_sides) {
        for (auto uplo : blas_uplos) {
          for (const auto& size : rectangular_sizes) {
            for (const auto& block_size : square_block_sizes) {
              for (const auto& col : col_b) {
                Matrix<TypeParam, Device::CPU> mat_a(size, block_size);
                LocalElementSize b_size(size.cols(), col);
                Matrix<TypeParam, Device::CPU> mat_b(b_size, block_size);
                TypeParam alpha = 1.0;
                EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, mat_a, mat_b),
                             std::invalid_argument);
              }
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(TriangularSolveLocalTest, BlockNotSquareException) {
  for (auto diag : blas_diags) {
    for (auto op : blas_ops) {
      for (auto side : blas_sides) {
        for (auto uplo : blas_uplos) {
          for (const auto& size : square_sizes) {
            for (const auto& block_size : rectangular_block_sizes) {
              for (const auto& col : col_b) {
                Matrix<TypeParam, Device::CPU> mat_a(size, block_size);
                LocalElementSize b_size(size.cols(), col);
                Matrix<TypeParam, Device::CPU> mat_b(b_size, block_size);
                TypeParam alpha = 1.0;
                EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, mat_a, mat_b),
                             std::invalid_argument);
              }
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(TriangularSolveLocalTest, MultipliableMatricesException) {
  for (auto diag : blas_diags) {
    for (auto op : blas_ops) {
      for (auto side : blas_sides) {
        for (auto uplo : blas_uplos) {
          for (const auto& size : square_sizes) {
            for (const auto& block_size : square_block_sizes) {
              for (const auto& col : col_b) {
                Matrix<TypeParam, Device::CPU> mat_a(size, block_size);

                LocalElementSize b_size(size.cols() * 2 + 3, col);
                if (side == blas::Side::Right)
                  b_size.transpose();

                Matrix<TypeParam, Device::CPU> mat_b(b_size, block_size);
                TypeParam alpha = 1.0;
                EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, mat_a, mat_b),
                             std::invalid_argument);
              }
            }
          }
        }
      }
    }
  }
}

TYPED_TEST(TriangularSolveLocalTest, MatrixNotLocalException) {
  for (auto diag : blas_diags) {
    for (auto op : blas_ops) {
      for (auto side : blas_sides) {
        for (auto uplo : blas_uplos) {
          for (const auto& size : square_sizes) {
            for (const auto& block_size : rectangular_block_sizes) {
              for (const auto& col : col_b) {
                {
                  GlobalElementSize sz = globalTestSize(size);
                  Distribution distribution(sz, block_size, {2, 1}, {0, 0}, {0, 0});
                  Matrix<TypeParam, Device::CPU> mat_a(std::move(distribution));
                  LocalElementSize b_size(size.cols(), col);
                  Matrix<TypeParam, Device::CPU> mat_b(b_size, block_size);
                  TypeParam alpha = 1.0;

                  EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, mat_a, mat_b),
                               std::invalid_argument);
                }

                {
                  Matrix<TypeParam, Device::CPU> mat_a(size, block_size);
                  LocalElementSize b_size(size.cols(), col);
                  GlobalElementSize sz = globalTestSize(b_size);
                  Distribution distribution(sz, block_size, {2, 1}, {0, 0}, {0, 0});
                  Matrix<TypeParam, Device::CPU> mat_b(std::move(distribution));
                  TypeParam alpha = 1.0;

                  EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, mat_a, mat_b),
                               std::invalid_argument);
                }
              }
            }
          }
        }
      }
    }
  }
}
