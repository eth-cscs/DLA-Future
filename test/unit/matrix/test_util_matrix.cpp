//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/util_matrix.h"

#include <gtest/gtest.h>
#include "dlaf/matrix.h"
#include "dlaf_test/util_matrix.h"

using namespace dlaf;

auto identity = [](const GlobalElementIndex& index) { if (index.row() == index.col()) return 1; return 0; };

template <class T>
struct constant_matrix {
  constant_matrix(const T value) : constant_(value) {}

  T operator()(const GlobalElementIndex& index) {
    return constant_;
  };

  const T constant_;
};

using T = double;

TEST(MatrixUtils, Set) {
  Matrix<T, Device::CPU> matrix({13, 7}, {2, 3});

  dlaf::matrix::util::set(matrix, identity);

  CHECK_MATRIX_EQ(identity, matrix);
}

TEST(MatrixUtils, SetRandom) {
  Matrix<T, Device::CPU> matrix({13, 7}, {2, 3});

  dlaf::matrix::util::set_random(matrix);

  CHECK_MATRIX_NEAR(constant_matrix<T>{0}, matrix, 0, 1);
}

TEST(MatrixUtils, SetRandomPositiveDefinite) {
  Matrix<T, Device::CPU> matrix({13, 7}, {2, 3});

  dlaf::matrix::util::set_random_positive_definite(matrix);

  auto N = std::max(matrix.size().cols(), matrix.size().rows());
  auto identity_2N = [N](const GlobalElementIndex& index) { if (index.row() == index.col()) return 2*N; return 0; };

  CHECK_MATRIX_NEAR(identity_2N, matrix, 0, 1);
}
