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
#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_base.h"
#include "dlaf/matrix/matrix_output.h"
#include "dlaf/util_matrix.h"
//#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::common;
using namespace dlaf::matrix;
using namespace dlaf::matrix::internal;
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
  const SizeType m = 4;
  const SizeType n = 4;
  const SizeType mb = 2;
  const SizeType nb = 2;

  // TODO: only for DOUBLE so far!!!
  LocalElementSize sizeC(m, n);
  TileElementSize blockSizeC(mb, nb);
  Matrix<double, Device::CPU> mat_c(sizeC, blockSizeC);
  dlaf::matrix::util::set_random(mat_c);
  std::cout << "Random matrix C" << std::endl;
  printElements(mat_c);

  LocalElementSize sizeV(m, n);
  TileElementSize blockSizeV(mb, nb);
  Matrix<double, Device::CPU> mat_v(sizeV, blockSizeV);
  dlaf::matrix::util::set_random(mat_v);
  std::cout << "Random matrix V" << std::endl;
  printElements(mat_v);

  LocalElementSize sizeVv(m, n);
  TileElementSize blockSizeVv(mb, nb);
  Matrix<double, Device::CPU> mat_vv(sizeVv, blockSizeVv);
  auto el_V = [](const GlobalElementIndex& index) {
    // ColMajor
    static const double values[] = {1.0, 0.98429, -0.0317176, 0.835213, 0.0, 1.0, 0.724638, 0.720084, 0.0, 0.0, 1.0, 0.827409, 0.0, 0.0, 0.0, 1.0};
    return values[index.row() + 4 * index.col()];
  };
  set(mat_vv, el_V);
  std::cout << "Matrix Vv" << std::endl;
  printElements(mat_vv);
  
  // Impose orthogonality: Q = I - v tau v^H is orthogonal (Q Q^H = I)
  // leads to tau = 2/(vT v) for real 
  LocalElementSize sizeTau(m, 1);
  TileElementSize blockSizeTau(1, 1);
  Matrix<double, Device::CPU> mat_tau(sizeTau, blockSizeTau);
  
  LocalElementSize sizeT(mb, n);
  TileElementSize blockSizeT(mb, nb);
  Matrix<double, Device::CPU> mat_t(sizeT, blockSizeT);
  set_zero(mat_t);
//  set(mat_t, el_T);
  std::cout << "Zero matrix T" << std::endl;
  printElements(mat_t);


  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);

  // Copy C matrix locally
  auto mat_c_loc = dlaf::matrix::test::all_gather<double>(mat_c, comm_grid);

  // Copy T matrix locally
  auto mat_t_loc = dlaf::matrix::test::all_gather<double>(mat_t, comm_grid);

  // Copy V matrix locally
  auto mat_v_loc = dlaf::matrix::test::all_gather<double>(mat_c, comm_grid);
  //dlaf::matrix::test::print(format::numpy{}, "mat Vloc ", mat_v_loc, std::cout);

  // Reset diagonal and upper values of V
  lapack::laset(lapack::MatrixType::Upper, mat_v_loc.size().rows(), mat_v_loc.size().cols(),
		0, 1,
		mat_v_loc.ptr(), mat_v_loc.ld());
  //  dlaf::matrix::test::print(format::numpy{}, "mat Vloc ", mat_v_loc, std::cout);

  //std::cout << " TAUS " << std::endl;
  //common::internal::vector<hpx::shared_future<TypeParam>> taus;
  //taus.reserve(m);
  //std::vector<double> taus;
  
  MatrixLocal<double> taus({m, 1}, {1, 1});
  MatrixLocal<double> work({m, 1}, {1, 1});
  // Compute taus (as tau = 2 / v^H v)
  for (SizeType i = n-2; i > -1; --i) {
    const GlobalElementIndex v_offset{0, i};
    auto tau = blas::dot(m, mat_v_loc.ptr(v_offset), 1, mat_v_loc.ptr(v_offset), 1);
    std::cout << " tau (" << i << "): " << tau << std::endl;
    //taus.push_back(hpx::make_ready_future<double>(tau));
    //taus.push_back(tau);
    taus({i,0}) = tau;

    lapack::larf(lapack::Side::Left, m, n, mat_v_loc.ptr(v_offset), 1, tau, mat_c_loc.ptr(), mat_c_loc.ld());
    dlaf::matrix::test::print(format::numpy{}, "mat Cloc ", mat_c_loc, std::cout);
  }

  std::cout << " " << std::endl;
  std::cout << " Result simple way " << std::endl;
  dlaf::matrix::test::print(format::numpy{}, "mat Cloc ", mat_c_loc, std::cout);
  std::cout << " " << std::endl;

  std::cout << " " << std::endl;
  std::cout << " Mat Tloc start " << std::endl;
  dlaf::matrix::test::print(format::numpy{}, "mat Tloc ", mat_t_loc, std::cout);
  std::cout << " " << std::endl;
  
  for (SizeType i = 0; i < mat_t.nrTiles().cols(); ++i) {
      std::cout << " i " << i  << std::endl;
      const GlobalElementIndex offset{0, i};
      const GlobalElementIndex tau_offset{i, 0};
      auto tile_t = mat_t(LocalTileIndex{0,i}).get();
      lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, nb, nb, mat_v_loc.ptr(offset), mat_v_loc.ld(), taus.ptr(tau_offset), tile_t.ptr(), tile_t.ld());
  }
  std::cout << "Matrix T" << std::endl;
  printElements(mat_t);
  
//  std::cout << " " << std::endl;
//  solver::backTransformation<Backend::MC>(mat_c, mat_vv, mat_tt);
//  std::cout << "Output " << std::endl;
//  printElements(mat_c);
//  
//  double error = 0.1;
//  CHECK_MATRIX_NEAR(mat_c, mat_c_loc, error, error);
}

