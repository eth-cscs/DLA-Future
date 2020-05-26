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

TYPED_TEST(EigensolverGenToStdLocalTest, Correctness) {

  LocalElementSize size = {3,3};
  TileElementSize block_size = {3,3};  

  //// MATRIX A
  
  auto el = [](const GlobalElementIndex& index){
    static const double values[] = {10.0, 20.0, 30.0, 20.0, 40.0, 60.0, 30.0, 60.0, 90.0};
    return values[index.row()+3*index.col()];
  };

  Matrix<double, Device::CPU> mat_a(size, block_size);
  set(mat_a, el);

  std::cout << "Mat size " << mat_a.size().rows() << " x " << mat_a.size().cols() << "\n";
  std::cout << "Tile size " << mat_a.blockSize().rows() << " x " << mat_a.blockSize().cols() << "\n";
  std::cout << "Number of tiles " << mat_a.nrTiles().rows()*mat_a.nrTiles().cols()  << "\n";

  SizeType tottilecol = mat_a.nrTiles().cols();
  SizeType tottilerow = mat_a.nrTiles().rows();
  for (SizeType j=0; j<tottilecol; ++j) {
    for (SizeType i=0; i<tottilerow; ++i) {
      std::cout << i << " " << j << "\n";
      auto tile = mat_a(LocalTileIndex(i,j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
	  std::cout << tile({ii, jj}) << " ";
	}
	std::cout << "\n";
      }
      std::cout << "\n";
    }
  }



  //// MATRIX L

    auto el_l = [](const GlobalElementIndex& index){
      static const double values[] = {2., 0., 0., 3., 4., 0., 5., 6., 7.};
    return values[index.row()+3*index.col()];
  };

  Matrix<double, Device::CPU> mat_l(size, block_size);
  set(mat_l, el_l);

  std::cout << "Mat size " << mat_l.size().rows() << " x " << mat_l.size().cols() << "\n";
  std::cout << "Tile size " << mat_l.blockSize().rows() << " x " << mat_l.blockSize().cols() << "\n";
  std::cout << "Number of tiles " << mat_l.nrTiles().rows()*mat_l.nrTiles().cols()  << "\n";

  SizeType tottilecoll = mat_l.nrTiles().cols();
  SizeType tottilerowl = mat_l.nrTiles().rows();
  for (SizeType j=0; j<tottilecoll; ++j) {
    for (SizeType i=0; i<tottilerowl; ++i) {
      std::cout << i << " " << j << "\n";
      auto tile = mat_l(LocalTileIndex(i,j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
	  std::cout << tile({ii, jj}) << " ";
	}
	std::cout << "\n";
      }
      std::cout << "\n";
    }
  }


  // genToStd
  Eigensolver<Backend::MC>::genToStd(mat_a, mat_l);

  std::cout << "Result" << std::endl;
  for (SizeType j=0; j<tottilecol; ++j) {
    for (SizeType i=0; i<tottilerow; ++i) {
      std::cout << i << " " << j << "\n";
      auto tile = mat_a(LocalTileIndex(i,j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
	  std::cout << tile({ii, jj}) << " ";
	}
	std::cout << "\n";
      }
      std::cout << "\n";
    }
  }


  auto res = [](const GlobalElementIndex& index){
    static const double values[] = {2.5, 0.625, -0.179, 0.625, 0.156, -0.045, -0.179, -0.045, 0.013};
    return values[index.row()+3*index.col()];
  };

  CHECK_MATRIX_NEAR(res, mat_a, 40 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error,
                    40 * (mat_a.size().rows() + 1) * TypeUtilities<TypeParam>::error);


}
  
