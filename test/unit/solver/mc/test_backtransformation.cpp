//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
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
//#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_output.h"
#include "dlaf/util_matrix.h"
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

//::testing::Environment* const comm_grids_env =
//    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class BackTransformationSolverLocalTest : public ::testing::Test {};
//TYPED_TEST_SUITE(BackTransformationSolverLocalTest, MatrixElementTypes);
TYPED_TEST_SUITE(BackTransformationSolverLocalTest, double);

//template <typename Type>
//class BackTransformationSolverDistributedTest : public ::testing::Test {
//public:
//  const std::vector<CommunicatorGrid>& commGrids() {
//    return comm_grids;
//}
//};
//TYPED_TEST_SUITE(BackTransformationSolverDistributedTest, double);

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

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}


//TYPED_TEST(BackTransformationSolverLocalTest, Correctness_n3_nb1) {
//  const SizeType m = 3;
//  const SizeType n = 3;
//  const SizeType mb = 1;
//  const SizeType nb = 1;
//  
//  // DATA
//  auto el_C = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {12, 6, -4, -51, 167, 24, 4, -68, -41};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  auto el_V = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {1, 0.23077, -0.15385, 0, 1, 0.055556, 0, 0, 0};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  auto el_T = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {1.8571, 0.0, 0.0, -0.82, 1.9938, 0.0, 0., 0., 0.};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  // RESULT
//  auto res = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {-14., 0., 0., -21., -175., 0., 14., 70., -35.};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  LocalElementSize sizeC(m, n);
//  TileElementSize blockSizeC(mb, nb);
//  Matrix<double, Device::CPU> mat_c(sizeC, blockSizeC);
//  set(mat_c, el_C);
//
//  LocalElementSize sizeV(m, n);
//  TileElementSize blockSizeV(mb, nb);
//  Matrix<double, Device::CPU> mat_v(sizeV, blockSizeV);
//  set(mat_v, el_V);
//
//  LocalElementSize sizeT(m, n);
//  TileElementSize blockSizeT(mb, nb);
//  Matrix<double, Device::CPU> mat_t(sizeT, blockSizeT);
//  set(mat_t, el_T);
//
//  solver::backTransformation<Backend::MC>(mat_c, mat_v, mat_t);
//
//  double error = 0.1;
//  CHECK_MATRIX_NEAR(res, mat_c, error, error);
//}
//
//

void set_zero(Matrix<double, Device::CPU>& mat) {
  set(mat, [](auto&&){return 0;});
}

template<class T>
MatrixLocal<T> makeLocal(const Matrix<const T, Device::CPU>& matrix) {
  return {matrix.size(), matrix.distribution().blockSize()};
}

template <class TypeParam, Device device>
void larft(const lapack::Direction direction, const lapack::StoreV storeV, const int n, const int k, const Tile<const TypeParam, device>& v, const Tile<const TypeParam, device>& tau, const Tile<TypeParam, device>& t) {    
    lapack::larft(direction, storeV, n, k, v.ptr(), v.ld(), tau.ptr(), t.ptr(), t.ld());	   
}
  
TYPED_TEST(BackTransformationSolverLocalTest, Correctness_random) {
  const SizeType m = 3;
  const SizeType n = 3;
  const SizeType mb = 2;
  const SizeType nb = 2;

  LocalElementSize sizeC(m, n);
  TileElementSize blockSizeC(mb, nb);
  Matrix<double, Device::CPU> mat_c(sizeC, blockSizeC);
  dlaf::matrix::util::set_random_seed(mat_c);
  std::cout << "Random matrix C" << std::endl;
  printElements(mat_c);

  LocalElementSize sizeV(m, n);
  TileElementSize blockSizeV(mb, nb);
  Matrix<double, Device::CPU> mat_v(sizeV, blockSizeV);
  dlaf::matrix::util::set_random_seed_lower(mat_v);
  std::cout << "Random matrix V" << std::endl;
  printElements(mat_v);

  // Impose orthogonality: Q = I - v tau v^H is orthogonal (Q Q^H = I)
  // leads to tau = 2/(vT v)
  LocalElementSize sizeTau(m, 1);
  TileElementSize blockSizeTau(1, 1);
  Matrix<double, Device::CPU> mat_tau(sizeTau, blockSizeTau);
  dlaf::matrix::util::set_random_seed(mat_tau);
  std::cout << "Random matrix Tau" << std::endl;
  printElements(mat_tau);

  for (SizeType j = 0; j < n; ++j) {
    for (SizeType i = 0; i < m; ++i) {
      LocalElementSize szV(m, 1);
      const LocalTileIndex ij{i, j};
      const LocalTileIndex ji{j, i};
      const auto& tileij = mat_v(LocalTileIndex{i,j}).get()({0,0});
      //    const auto& tileji = dlaf::conj(mat_v.read(ji));
//      std::cout << "tileij " << tileij.get().size();
      //      auto tau = blas::dot(1, tileij.get().ptr(), 1, tileji.get().ptr(), 1); 
    }
  }

  
  LocalElementSize sizeT(m, n);
  TileElementSize blockSizeT(mb, nb);
  Matrix<double, Device::CPU> mat_t(sizeT, blockSizeT);
  set_zero(mat_t);
  std::cout << "Zero matrix T" << std::endl;
  printElements(mat_t);

  
  //  hpx::dataflow(hpx::util::unwrapping(larft<TypeParam, Device::CPU>), lapack::Direction::Forward, lapack::StoreV::Columnwise, m-1, m-1, mat_v.read(LocalTileIndex{0,0}).get().ptr(), mat_v.read(LocalTileIndex{0,0}).get().ld(), mat_tau.read(LocalTileIndex{0,0}).get().ptr(), mat_t(LocalTileIndex{0,0}).get().ptr(), mat_t(LocalTileIndex{0,0}).get().ld());

  //for (SizeType i = 0; i < m; ++i) {
  //hpx::dataflow(executor_hp, hpx::util::unwrapping(cpy), mat_v.read(LocalTileIndex(i, k)), mat_w_last(LocalTileIndex(i, 0)));
	   
  lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, m-1, m-1, mat_v.read(LocalTileIndex{0,0}).get().ptr(), mat_v.read(LocalTileIndex{0,0}).get().ld(), mat_tau.read(LocalTileIndex{0,0}).get().ptr(), mat_t(LocalTileIndex{0,0}).get().ptr(), mat_t(LocalTileIndex{0,0}).get().ld());
  
//  solver::backTransformation<Backend::MC>(mat_c, mat_v, mat_t);
//  std::cout << "Output " << std::endl;
//  printElements(mat_t);
  
//  double error = 0.1;
//  CHECK_MATRIX_NEAR(res, mat_c, error, error);
}

