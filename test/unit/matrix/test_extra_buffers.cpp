//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/extra_buffers.h"

#include <vector>

#include <gtest/gtest.h>

#include "dlaf/matrix/matrix.h"

#include "dlaf_test/util_types.h"
#include "dlaf_test/matrix/util_matrix.h"

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace testing;

template <typename Type>
class ExtraBuffersTest : public ::testing::Test {};

TYPED_TEST_SUITE(ExtraBuffersTest, MatrixElementTypes);

TYPED_TEST(ExtraBuffersTest, Basic) {
  Matrix<TypeParam, Device::CPU> matrix({1, 1}, {1, 1});
  matrix::test::set(matrix, [](auto&&){ return 1; });

  ExtraBuffers<TypeParam> buffers(matrix, 2);

  matrix::test::set(buffers.get_buffer(1).get(), [](auto&&) { return 3; });
  matrix::test::set(buffers.get_buffer(2).get(), [](auto&&) { return 5; });

  buffers.reduce();

  CHECK_MATRIX_EQ([](const GlobalElementIndex&) { return TypeUtilities<TypeParam>::element(9, 0); }, matrix);
}
