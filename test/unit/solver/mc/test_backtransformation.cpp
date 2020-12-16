//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/solver/backtransformation.h"

#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix_output.h"
//#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;
using namespace testing;

template <typename Type>
class BackTransformationSolverLocalTest : public ::testing::Test {};
TYPED_TEST_SUITE(BackTransformationSolverLocalTest, MatrixElementTypes);

const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Diag> blas_diags({blas::Diag::NonUnit, blas::Diag::Unit});

//const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> sizes = {
//    {0, 0, 1, 1},                                                // m, n = 0
//    {0, 2, 1, 2}, {7, 0, 2, 1},                                  // m = 0 or n = 0
//    {2, 2, 5, 5}, {10, 10, 2, 3}, {7, 7, 3, 2},                  // m = n
//    {3, 2, 7, 7}, {12, 3, 5, 5},  {7, 6, 3, 2}, {15, 7, 3, 5},   // m > n
//    {2, 3, 7, 7}, {4, 13, 5, 5},  {7, 8, 2, 9}, {19, 25, 6, 5},  // m < n
//};
//
//GlobalElementSize globalTestSize(const LocalElementSize& size) {
//  return {size.rows(), size.cols()};
//}

 TYPED_TEST(BackTransformationSolverLocalTest, Correctness3x3) {
  const SizeType n = 3;
  const SizeType nb = 1;
  
  // DATA
  auto el_C = [](const GlobalElementIndex& index) {
    // ColMajor
    static const double values[] = {12, 6, -4, -51, 167, 24, 4, -68, -41};
    return values[index.row() + 3 * index.col()];
  };

  auto el_V = [](const GlobalElementIndex& index) {
    // ColMajor
    static const double values[] = {1, 0.23077, -0.15385, 0, 1, 0.055556, 0, 0, 0};
    return values[index.row() + 3 * index.col()];
  };

  auto el_T = [](const GlobalElementIndex& index) {
    // ColMajor
    static const double values[] = {1.8571, 0.0, 0.0, -0.82, 1.9938, 0.0, 0., 0., 0.};
    //static const double values[] = {1.8571, 1.8571, 1.8571, 0.0, 1.9938, 1.9938, 0, 0, 0};
    return values[index.row() + 3 * index.col()];
  };

  // RESULT
  auto res = [](const GlobalElementIndex& index) {
    // ColMajor
    static const double values[] = {-14., 0., 0., -21., -175., 0., 14., 70., -35.};
    return values[index.row() + 3 * index.col()];
  };

  LocalElementSize sizeC(n, n);
  TileElementSize blockSizeC(nb, nb);
  Matrix<double, Device::CPU> mat_c(sizeC, blockSizeC);
  set(mat_c, el_C);

  LocalElementSize sizeV(n, n);
  TileElementSize blockSizeV(nb, nb);
  Matrix<double, Device::CPU> mat_v(sizeV, blockSizeV);
  set(mat_v, el_V);

  LocalElementSize sizeT(n, n);
  TileElementSize blockSizeT(nb, nb);
  Matrix<double, Device::CPU> mat_t(sizeT, blockSizeT);
  set(mat_t, el_T);

  std::cout << "Matrix C" << std::endl;
  printElements(mat_c);
  std::cout << "" << std::endl;
  std::cout << "Matrix V" << std::endl;
  printElements(mat_v);
  std::cout << "" << std::endl;
  std::cout << "Matrix T" << std::endl;
  printElements(mat_t);
  std::cout << "" << std::endl;

  solver::backTransformation<Backend::MC>(mat_c, mat_v, mat_t);

  std::cout << "Result: " << std::endl;
  printElements(mat_c);

  double error = 0.1;
  CHECK_MATRIX_NEAR(res, mat_c, error, error);
}

