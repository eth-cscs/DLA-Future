//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/mc.h"

#include <exception>
#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix_output.h"
#include "dlaf_test/matrix/util_generic_lapack.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

template <typename Type>
class EigensolverGenToStdLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(EigensolverGenToStdLocalTest, MatrixElementTypes);

const std::vector<std::tuple<SizeType, SizeType>> sizes = {
    {0, 2},                              // m = 0
    {2, 2}, {5, 5}, {13, 13}, {34, 34},  // m = mb
    {4, 2}, {4, 3}, {16, 10}, {34, 13}   // m != mb
};

// TYPED_TEST(EigensolverGenToStdLocalTest, CorrectnessTile1) {
//  LocalElementSize size = {3, 3};
//  TileElementSize block_size = {1, 1};
//
//  // MATRIX A
//  auto el = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0};
//    return values[index.row() + 3 * index.col()];
//  };
//  Matrix<double, Device::CPU> mat_a(size, block_size);
//  set(mat_a, el);
//  std::cout << "Matrix A" << std::endl;
//  std::cout << "Size " << mat_a.size().rows() << " x " << mat_a.size().cols() << "\n";
//  std::cout << "Tile size " << mat_a.blockSize().rows() << " x " << mat_a.blockSize().cols() << "\n";
//  std::cout << "Number of tiles " << mat_a.nrTiles().rows() * mat_a.nrTiles().cols() << "\n";
//  printElements(mat_a);
//
//  // MATRIX L
//  auto el_l = [](const GlobalElementIndex& index) {
//    static const double values[] = {2., 3., 5., 0., 4., 6., 0., 0., 7.};
//    return values[index.row() + 3 * index.col()];
//  };
//  Matrix<double, Device::CPU> mat_l(size, block_size);
//  set(mat_l, el_l);
//  std::cout << "MAT L" << std::endl;
//  std::cout << "Size " << mat_l.size().rows() << " x " << mat_l.size().cols() << "\n";
//  std::cout << "Tile size " << mat_l.blockSize().rows() << " x " << mat_l.blockSize().cols() << "\n";
//  std::cout << "Number of tiles " << mat_l.nrTiles().rows() * mat_l.nrTiles().cols() << "\n";
//  printElements(mat_l);
//
//  // genToStd
//  Eigensolver<Backend::MC>::genToStd(mat_a, mat_l);
//
//  std::cout << "Result" << std::endl;
//  printElements(mat_a);
//
//  auto res = [](const GlobalElementIndex& index) {
//    static const double values[] = {2.5,        0.625, -0.1785714, 20.,      0.15625,
//                                    -0.0446429, 30.,   60.,        0.0127551};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  CHECK_MATRIX_NEAR(res, mat_a, 100000000 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error,
//                    100000000 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error);
//}
//
// TYPED_TEST(EigensolverGenToStdLocalTest, CorrectnessTile2) {
//  LocalElementSize size = {3, 3};
//  TileElementSize block_size = {2, 2};
//
//  // MATRIX A
//  auto el = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0};
//    return values[index.row() + 3 * index.col()];
//  };
//  Matrix<double, Device::CPU> mat_a(size, block_size);
//  set(mat_a, el);
//
//  // MATRIX L
//  auto el_l = [](const GlobalElementIndex& index) {
//    static const double values[] = {2., 3., 5., 0., 4., 6., 0., 0., 7.};
//    return values[index.row() + 3 * index.col()];
//  };
//  Matrix<double, Device::CPU> mat_l(size, block_size);
//  set(mat_l, el_l);
//
//  // genToStd
//  Eigensolver<Backend::MC>::genToStd(mat_a, mat_l);
//
//  std::cout << "Result" << std::endl;
//  printElements(mat_a);
//
//  auto res = [](const GlobalElementIndex& index) {
//    static const double values[] = {2.5,        0.625, -0.1785714, 20.,      0.15625,
//                                    -0.0446429, 30.,   60.,        0.0127551};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  CHECK_MATRIX_NEAR(res, mat_a, 100000000 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error,
//                    100000000 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error);
//}
//
//
// TYPED_TEST(EigensolverGenToStdLocalTest, CorrectnessTile3) {
//  LocalElementSize size = {3, 3};
//  TileElementSize block_size = {3, 3};
//
//  // MATRIX A
//  auto el = [](const GlobalElementIndex& index) {
//    // ColMajor
//    static const double values[] = {10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0};
//    return values[index.row() + 3 * index.col()];
//  };
//  Matrix<double, Device::CPU> mat_a(size, block_size);
//  set(mat_a, el);
//  std::cout << "A" << std::endl;
//  printElements(mat_a);
//
//  // MATRIX L
//  auto el_l = [](const GlobalElementIndex& index) {
//    static const double values[] = {2., 3., 5., 0., 4., 6., 0., 0., 7.};
//    return values[index.row() + 3 * index.col()];
//  };
//  Matrix<double, Device::CPU> mat_l(size, block_size);
//  set(mat_l, el_l);
//  std::cout << "L" << std::endl;
//  printElements(mat_l);
//
//  // genToStd
//  Eigensolver<Backend::MC>::genToStd(mat_a, mat_l);
//
//  std::cout << "Result" << std::endl;
//  printElements(mat_a);
//
//  auto res = [](const GlobalElementIndex& index) {
//    static const double values[] = {2.5,        0.625, -0.1785714, 20.,      0.15625,
//                                    -0.0446429, 30.,   60.,        0.0127551};
//    return values[index.row() + 3 * index.col()];
//  };
//
//  CHECK_MATRIX_NEAR(res, mat_a, 100000000 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error,
//                    100000000 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error);
//
//}

template <class T>
void testGenToStdEigensolver(T alpha, T beta, T gamma, SizeType m, SizeType mb) {
  std::function<T(const GlobalElementIndex&)> el_l, el_a, res_b;

  LocalElementSize size(m, m);
  TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_a(size, block_size);
  Matrix<T, Device::CPU> mat_l(size, block_size);

  std::tie(el_l, el_a, res_b) = test::getLowerHermitianSystem<GlobalElementIndex, T>(alpha, beta, gamma);

  set(mat_a, el_a);
  set(mat_l, el_l);

  Eigensolver<Backend::MC>::genToStd(mat_a, mat_l);

  CHECK_MATRIX_NEAR(res_b, mat_a, 10 * (mat_a.size().rows() + 1) * TypeUtilities<T>::error,
                    10 * (mat_a.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(EigensolverGenToStdLocalTest, Correctness) {
  SizeType m, mb;

  for (auto sz : sizes) {
    std::tie(m, mb) = sz;
    TypeParam alpha = TypeUtilities<TypeParam>::element(1., 0.);
    TypeParam beta = TypeUtilities<TypeParam>::element(1., 0.);
    TypeParam gamma = TypeUtilities<TypeParam>::element(1., 0.);

    testGenToStdEigensolver(alpha, beta, gamma, m, mb);
  }
}

// TYPED_TEST(EigensolverGenToStdLocalTest, Random) {
//  //  const std::vector<SizeType> sizes = {3, 4, 5, 10, 25, 30};
//  //  const std::vector<SizeType> blocksizes = {2, 3, 4, 5, 7};
//  const std::vector<SizeType> sizes = {3};
//  const std::vector<SizeType> blocksizes = {3};
//
//  using MatrixType = dlaf::Matrix<TypeParam, Device::CPU>;
//
//  for (auto sz : sizes) {
//    for (auto bs : blocksizes) {
//      LocalElementSize matrix_size(sz, sz);
//      TileElementSize block_size(bs, bs);
//
//      // Allocate memory for the matrix A
//      MatrixType mat_a = [matrix_size, block_size]() {
//	using dlaf::matrix::util::set_random_hermitian_positive_definite;
//
//	MatrixType hermitian_pos_def(matrix_size, block_size);
//	set_random_hermitian_positive_definite(hermitian_pos_def);
//
//	return hermitian_pos_def;
//      }();
//
//      // Allocate memory for the matrix L
//      MatrixType mat_l = [matrix_size, block_size]() {
//	using dlaf::matrix::util::set_random_lowtr;
//
//	MatrixType lower_triangular(matrix_size, block_size);
//	set_random_lowtr(lower_triangular);
//
//	return lower_triangular;
//      }();
//
//      std::cout << "Matrix A" << std::endl;
//      printElements(mat_a);
//      std::cout << " " << std::endl;
//      std::cout << "Matrix L" << std::endl;
//      printElements(mat_l);
//
//      //      lapack::hegst(1, blas::Uplo::Lower, mat_a.size().cols(), mat_a.ptr(), mat_a.ld(),
//      mat_l.ptr(), mat_l.ld());
//
//      // genToStd
//      Eigensolver<Backend::MC>::genToStd(mat_a, mat_l);
//      std::cout << "Result" << std::endl;
//      printElements(mat_a);
//
//      //
//      //  CHECK_MATRIX_NEAR(res, mat_a, 100000000 * (mat_a.size().rows() + 1) *
//      TypeUtilities<TypeParam>::error,
//      //                    100000000 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error);
//    }
//  }
//}
