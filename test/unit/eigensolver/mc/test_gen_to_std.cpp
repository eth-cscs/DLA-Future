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
    {5, 5}, {34, 34},                    // m = mb
    {4, 3}, {16, 10}, {34, 13}, {32, 5}  // m != mb
};

template <class T>
void testGenToStdEigensolver(blas::Uplo uplo, T alpha, T beta, T gamma, SizeType m, SizeType mb) {
  std::function<T(const GlobalElementIndex&)> el_l, el_a, res_b;

  LocalElementSize size(m, m);
  TileElementSize block_size(mb, mb);

  Matrix<T, Device::CPU> mat_a(size, block_size);
  Matrix<T, Device::CPU> mat_l(size, block_size);

  std::tie(el_l, el_a, res_b) =
      test::getHermitianSystem<GlobalElementIndex, T>(uplo, alpha, beta, gamma);

  set(mat_a, el_a);
  set(mat_l, el_l);

  Eigensolver<Backend::MC>::genToStd(uplo, mat_a, mat_l);

  CHECK_MATRIX_NEAR(res_b, mat_a, 10 * (mat_a.size().rows() + 1) * TypeUtilities<T>::error,
                    10 * (mat_a.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(EigensolverGenToStdLocalTest, Correctness) {
  SizeType m, mb;
  blas::Uplo uplo = blas::Uplo::Lower;

  for (auto sz : sizes) {
    std::tie(m, mb) = sz;
    BaseType<TypeParam> alpha = 1.2f;
    BaseType<TypeParam> beta = 1.5f;
    BaseType<TypeParam> gamma = -1.1f;

    testGenToStdEigensolver(uplo, alpha, beta, gamma, m, mb);
  }
}
