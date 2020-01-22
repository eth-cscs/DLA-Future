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

std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::NonUnit});
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

/// @brief Returns el_op_a, el_b, res_b for side = Left. Same implementation as in test_trsm.h (there
/// referred to tiles).
template <class T>
auto testTriangularSolveElementFunctionsLeft(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                                             SizeType m) {
  // Note: The tile elements are chosen such that:
  // - op(a)_ik = (i+1) / (k+.5) * exp(I*(2*i-k)) for the referenced elements
  //   op(a)_ik = -9.9 otherwise,
  // - res_kj = (k+.5) / (j+2) * exp(I*(k+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the elements of b should be:
  // b_ij = (Sum_k op(a)_ik * res_kj) / alpha
  //      = (op(a)_ii * res_ij + (kk-1) * gamma) / alpha,
  // where gamma = (i+1) / (j+2) * exp(I*(2*i+j)),
  //       kk = i+1 if op(a) is an lower triangular matrix, or
  //       kk = m-i if op(a) is an upper triangular matrix.
  // Therefore
  // b_ij = (res_ij + (kk-1) * gamma) / alpha, if diag == Unit
  // b_ij = kk * gamma / alpha, otherwise.
  bool op_a_lower = false;
  if ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
      (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans))
    op_a_lower = true;

  std::function<T(const GlobalElementIndex&)> el_op_a = [op_a_lower,
                                                         diag](const GlobalElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double i = index.row();
    double k = index.col();

    return TypeUtilities<T>::polar((i + 1) / (k + .5), 2 * i - k);
  };

  std::function<T(const GlobalElementIndex&)> res_b = [](const GlobalElementIndex& index) {
    double k = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar((k + .5) / (j + 2), k + j);
  };

  std::function<T(const GlobalElementIndex&)> el_b = [m, alpha, diag, op_a_lower, uplo,
                                                      res_b](const GlobalElementIndex& index) {
    BaseType<T> kk = op_a_lower ? index.row() + 1 : m - index.row();

    double i = index.row();
    double j = index.col();

    T gamma = TypeUtilities<T>::polar((i + 1) / (j + 2), 2 * i + j);

    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + res_b(index)) / alpha;
    else
      return kk * gamma / alpha;
  };

  return std::make_tuple(el_op_a, el_b, res_b);
}

/// @brief Returns el_op_a, el_b, res_b for side = Right. Same implementation as in test_trsm.h (there
/// referred to tiles).
template <class T>
auto testTriangularSolveElementFunctionsRight(blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                                              SizeType n) {
  // Note: The tile elements are chosen such that:
  // - res_ik = (k+.5) / (i+2) * exp(I*(i+k)),
  // - op(a)_kj = (j+1) / (k+.5) * exp(I*(2*j-k)) for the referenced elements
  //   op(a)_kj = -9.9 otherwise,
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the elements of b should be:
  // b_ij = (Sum_k res_ik * op(a)_kj) / alpha
  //      = (res_ij * op(a)_jj + (kk-1) * gamma) / alpha,
  // where gamma = (j+1) / (i+2) * exp(I*(i+2*j)),
  //       kk = j+1 if op(a) is an upper triangular matrix, or
  //       kk = n-j if op(a) is an lower triangular matrix.
  // Therefore
  // b_ij = (res_ij + (kk-1) * gamma) / alpha, if diag == Unit
  // b_ij = kk * gamma / alpha, otherwise.

  bool op_a_lower = false;
  if ((uplo == blas::Uplo::Lower && op == blas::Op::NoTrans) ||
      (uplo == blas::Uplo::Upper && op != blas::Op::NoTrans))
    op_a_lower = true;

  auto res_b = [](const GlobalElementIndex& index) {
    double i = index.row();
    double k = index.col();

    return TypeUtilities<T>::polar((k + .5) / (i + 2), i + k);
  };

  auto el_op_a = [op_a_lower, diag](const GlobalElementIndex& index) {
    if ((op_a_lower && index.row() < index.col()) || (!op_a_lower && index.row() > index.col()) ||
        (diag == blas::Diag::Unit && index.row() == index.col()))
      return TypeUtilities<T>::element(-9.9, 0);

    double k = index.row();
    double j = index.col();

    return TypeUtilities<T>::polar((j + 1) / (k + .5), 2 * j - k);
  };

  auto el_b = [n, alpha, diag, op_a_lower, res_b](const GlobalElementIndex& index) {
    BaseType<T> kk = op_a_lower ? n - index.col() : index.col() + 1;

    double i = index.row();
    double j = index.col();
    T gamma = TypeUtilities<T>::polar((j + 1) / (i + 2), i + 2 * j);

    if (diag == blas::Diag::Unit)
      return ((kk - 1) * gamma + res_b(index)) / alpha;
    else
      return kk * gamma / alpha;
  };

  return std::make_tuple(el_op_a, el_b, res_b);
}

template <class T>
void testTriangularSolve(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                         Matrix<T, Device::CPU>& matA, Matrix<T, Device::CPU>& matB,
                         Matrix<T, Device::CPU>& matX) {
  std::function<T(const GlobalElementIndex&)> el_op_a, el_b, res_b;

  auto m = matB.size().rows();
  auto n = matB.size().cols();

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) =
        testTriangularSolveElementFunctionsLeft<T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) =
        testTriangularSolveElementFunctionsRight<T>(uplo, op, diag, alpha, n);

  set(matA, el_op_a, op);
  set(matB, el_b);
  set(matX, res_b);

  triangular_solve(side, uplo, op, diag, alpha, matA, matB);

  CHECK_MATRIX_NEAR(res_b, matB, 20 * (matB.size().rows() + 1) * TypeUtilities<T>::error,
                    20 * (matB.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(TriangularSolveLocalTest, Correctness) {
  for (auto diag : blas_diags) {
    for (auto op : blas_ops) {
      for (auto side : blas_sides) {
        for (auto uplo : blas_uplos) {
          for (const auto& size : square_sizes) {
            for (const auto& block_size : square_block_sizes) {
              for (const auto& col : col_b) {
                Matrix<TypeParam, Device::CPU> matA(size, block_size);
                LocalElementSize B_size(size.rows(), col);
                if (side == blas::Side::Right)
                  B_size.transpose();

                Matrix<TypeParam, Device::CPU> matB(B_size, block_size);
                Matrix<TypeParam, Device::CPU> matX(B_size, block_size);

                TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

                testTriangularSolve(side, uplo, op, diag, alpha, matA, matB, matX);
              }
            }
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

// NoLocalMatrixException test to be added
