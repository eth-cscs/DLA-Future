//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/lapack/laed4.h"

#include "gtest/gtest.h"

#include "dlaf/types.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;

template <class T, Device D>
class TileOperationsTest : public ::testing::Test {};

template <class T>
using RealTileOperationsTestMC = TileOperationsTest<T, Device::CPU>;

TYPED_TEST_SUITE(RealTileOperationsTestMC, RealMatrixElementTypes);

// To reproduce the setup in python:
//
// ```
// import numpy as np
// from scipy.linalg import eigh
// from scipy.linalg import norm
//
// n = 20
// d = np.log(np.arange(2, n + 2))
// z = np.arange(1, n + 1) / n
// z = z / norm(z)
// eigh(np.diag(d) + np.outer(z, np.transpose(z)), eigvals_only=True, subset_by_index=[n / 2, n / 2])
//
// ```
//
TYPED_TEST(RealTileOperationsTestMC, Laed4) {
  int n = 20;
  std::vector<TypeParam> d(to_sizet(n));
  std::vector<TypeParam> z(to_sizet(n));
  TypeParam sumsq = 0;
  for (std::size_t i = 0; i < to_sizet(n); ++i) {
    d[i] = std::log(TypeParam(i + 2));
    z[i] = TypeParam(i + 1) / n;
    sumsq += z[i] * z[i];
  }
  TypeParam rho = 1.0 / sumsq;  // the factor essentially normalizes `z`
  int i = n / 2;                // the index of the middle eigenvalue
  std::vector<TypeParam> delta(to_sizet(n));
  TypeParam lambda;

  dlaf::internal::laed4_wrapper(n, i, d.data(), z.data(), delta.data(), rho, &lambda);

  TypeParam expected_lambda = 2.497336;
  EXPECT_NEAR(lambda, expected_lambda, 1e-7);
}
