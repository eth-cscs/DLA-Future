//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/solver/triangular.h"

#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

template <typename Type>
class TriangularSolverLocalTest : public ::testing::Test {};
TYPED_TEST_SUITE(TriangularSolverLocalTest, MatrixElementTypes);

const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
    {0, 0, 1, 1},                                                // m, n = 0
    {0, 2, 1, 2}, {7, 0, 2, 1},                                  // m = 0 or n = 0
    {2, 2, 5, 5}, {10, 10, 2, 3}, {7, 7, 3, 2},                  // m = n
    {3, 2, 7, 7}, {12, 3, 5, 5},  {7, 6, 3, 2}, {15, 7, 3, 5},   // m > n
    {2, 3, 7, 7}, {4, 13, 5, 5},  {7, 8, 2, 9}, {19, 25, 6, 5},  // m < n
};

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

// TODO: Only for debugging. Useful elsewhere?
template <typename T>
void print_matrix(Matrix<T, Device::CPU>& m) {
  auto mrows = m.nrTiles().rows();
  auto mcols = m.nrTiles().cols();
  std::cout << "matrix rows = " << m.size().rows() << ", matrix columns = " << m.size().rows()
            << std::endl;
  std::cout << "matrix tile rows = " << mrows << ", matrix tile columns = " << mcols << std::endl;
  for (SizeType r = 0; r < mrows; ++r) {
    for (SizeType c = 0; c < mcols; ++c) {
      auto t = m(LocalTileIndex{r, c}).get();
      auto trows = t.size().rows();
      auto tcols = t.size().cols();
      std::cout << "tile row = " << r << ", tile column = " << c << std::endl;
      std::cout << "tile rows = " << trows << ", tile columns = " << tcols << std::endl;
      for (SizeType rr = 0; rr < trows; ++rr) {
        for (SizeType cc = 0; cc < tcols; ++cc) {
          std::cout << t(TileElementIndex{rr, cc}) << " ";
        }
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
  }
}

// TODO: Only for debugging. Useful elsewhere?
template <typename T, Device b>
void fence_matrix(Matrix<T, b>& m) {
  auto mrows = m.nrTiles().rows();
  auto mcols = m.nrTiles().cols();
  for (SizeType r = 0; r < mrows; ++r) {
    for (SizeType c = 0; c < mcols; ++c) {
      m.read(LocalTileIndex{r, c}).get();
    }
  }
}

template <class T>
void testTriangularSolver(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, T alpha,
                          SizeType m, SizeType n, SizeType mb, SizeType nb) {
  hpx::cuda::experimental::enable_user_polling p;

  std::function<T(const GlobalElementIndex&)> el_op_a, el_b, res_b;

  LocalElementSize size_a(m, m);
  TileElementSize block_size_a(mb, mb);

  if (side == blas::Side::Right) {
    size_a = {n, n};
    block_size_a = {nb, nb};
  }

  Matrix<T, Device::CPU> mat_a(size_a, block_size_a);
  Matrix<T, Device::GPU> mat_ad(size_a, block_size_a);

  LocalElementSize size_b(m, n);
  TileElementSize block_size_b(mb, nb);
  Matrix<T, Device::CPU> mat_b(size_b, block_size_b);
  Matrix<T, Device::GPU> mat_bd(size_b, block_size_b);

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) =
        getLeftTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) =
        getRightTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, n);

  set(mat_a, el_op_a, op);
  set(mat_b, el_b);

  copy(mat_a, mat_ad);
  copy(mat_b, mat_bd);

  solver::triangular<Backend::GPU>(side, uplo, op, diag, alpha, mat_ad, mat_bd);

  copy(mat_bd, mat_b);

  CHECK_MATRIX_NEAR(res_b, mat_b, 40 * (mat_b.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_b.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(TriangularSolverLocalTest, Correctness) {
  SizeType m, n, mb, nb;

  for (auto side : blas_sides) {
    for (auto uplo : blas_uplos) {
      for (auto op : blas_ops) {
        for (auto diag : blas_diags) {
          for (auto sz : sizes) {
            std::tie(m, n, mb, nb) = sz;
            TypeParam alpha = TypeUtilities<TypeParam>::element(-1.2, .7);

            testTriangularSolver(side, uplo, op, diag, alpha, m, n, mb, nb);
          }
        }
      }
    }
  }
}
