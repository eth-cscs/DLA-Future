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
#include "dlaf_test/util_matrix.h"
#include "dlaf_test/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::comm;
using namespace dlaf_test;
using namespace dlaf_test::matrix_test;
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

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T>
void testTriangularSolve(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                         LocalElementSize size, unsigned int colB, TileElementSize block_size) {
  std::function<T(const GlobalElementIndex&)> el_op_a, el_b, res_b;

  auto m = size.rows();
  auto n = size.cols();

  Matrix<T, Device::CPU> matA(size, block_size);
  LocalElementSize B_size(m, colB);
  if (side == blas::Side::Right)
    B_size.transpose();

  Matrix<T, Device::CPU> matB(B_size, block_size);

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) =
        testTrsmElementFunctionsLeft<T, GlobalElementIndex>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) =
        testTrsmElementFunctionsRight<T, GlobalElementIndex>(uplo, op, diag, alpha, n);

  set(matA, el_op_a, op);
  set(matB, el_b);

  triangular_solve(side, uplo, op, diag, alpha, matA, matB);

  CHECK_MATRIX_NEAR(res_b, matB, 20 * (matB.size().rows() + 1) * TypeUtilities<T>::error,
                    20 * (matB.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(TriangularSolveLocalTest, Correctness) {
  LocalElementSize MatSize(0, 0);
  unsigned int colB;
  TileElementSize BlockSize(0, 0);

  std::vector<std::tuple<LocalElementSize, unsigned int, TileElementSize>> sizes;
  for (const auto& size : square_sizes) {
    for (const auto& col : col_b) {
      for (const auto& block_size : square_block_sizes) {
        sizes.push_back(std::make_tuple(size, col, block_size));
      }
    }
  }

  for (auto diag : blas_diags) {
    for (auto op : blas_ops) {
      for (auto side : blas_sides) {
        for (auto uplo : blas_uplos) {
          for (auto sz : sizes) {
            std::tie(MatSize, colB, BlockSize) = sz;
            TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

            testTriangularSolve(side, uplo, op, diag, alpha, MatSize, colB, BlockSize);
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
                Matrix<TypeParam, Device::CPU> matA(size, block_size);
                LocalElementSize B_size(size.cols(), col);
                Matrix<TypeParam, Device::CPU> matB(B_size, block_size);
                TypeParam alpha = 1.0;
                EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, matA, matB),
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
                Matrix<TypeParam, Device::CPU> matA(size, block_size);
                LocalElementSize B_size(size.cols(), col);
                Matrix<TypeParam, Device::CPU> matB(B_size, block_size);
                TypeParam alpha = 1.0;
                EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, matA, matB),
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
                Matrix<TypeParam, Device::CPU> matA(size, block_size);

                LocalElementSize B_size(size.cols() * 2 + 3, col);
                if (side == blas::Side::Right)
                  B_size.transpose();

                Matrix<TypeParam, Device::CPU> matB(B_size, block_size);
                TypeParam alpha = 1.0;
                EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, matA, matB),
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
                  Matrix<TypeParam, Device::CPU> matA(std::move(distribution));
                  LocalElementSize B_size(size.cols(), col);
                  Matrix<TypeParam, Device::CPU> matB(B_size, block_size);
                  TypeParam alpha = 1.0;

                  EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, matA, matB),
                               std::invalid_argument);
                }

                {
                  Matrix<TypeParam, Device::CPU> matA(size, block_size);
                  LocalElementSize B_size(size.cols(), col);
                  GlobalElementSize sz = globalTestSize(B_size);
                  Distribution distribution(sz, block_size, {2, 1}, {0, 0}, {0, 0});
                  Matrix<TypeParam, Device::CPU> matB(std::move(distribution));
                  TypeParam alpha = 1.0;

                  EXPECT_THROW(triangular_solve(side, uplo, op, diag, alpha, matA, matB),
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
