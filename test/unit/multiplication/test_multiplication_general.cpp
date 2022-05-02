//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/multiplication/general.h"

#include <gtest/gtest.h>

#include "dlaf/blas/enum_output.h"
#include "dlaf/common/assert.h"
#include "dlaf/matrix/index.h"
#include "dlaf/util_matrix.h"

#include "dlaf_test/matrix/util_generic_blas.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;

template <class T>
struct GeneralMultiplicationTestMC : public ::testing::Test {};

TYPED_TEST_SUITE(GeneralMultiplicationTestMC, MatrixElementTypes);

const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType, SizeType>> sizes = {
    // m, n, k, mb, a, b
    {3, 3, 3, 1, 0, 2}, {3, 3, 3, 3, 0, 0},    {6, 6, 6, 3, 0, 1},
    {9, 9, 9, 3, 0, 2}, {21, 21, 21, 3, 0, 6},
};

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T, Backend B, Device D>
void testGeneralMultiplication(const SizeType a, const SizeType b, const blas::Op opA,
                               const blas::Op opB, const T alpha, const T beta, const SizeType m,
                               const SizeType n, const SizeType k, const SizeType mb) {
  const SizeType a_el = a * mb;
  const SizeType b_el = std::min((b + 1) * mb - 1, m - 1);

  auto [refA, refB, refC, refResult] =
      matrix::test::getSubMatrixMatrixMultiplication(a_el, b_el, m, n, k, alpha, beta, opA, opB);

  auto setMatrix = [&](auto elSetter, const LocalElementSize size, const TileElementSize block_size) {
    Matrix<T, Device::CPU> matrix(size, block_size);
    dlaf::matrix::util::set(matrix, elSetter);
    return matrix;
  };

  Matrix<const T, Device::CPU> mat_a = setMatrix(refA, {m, k}, {mb, mb});
  Matrix<const T, Device::CPU> mat_b = setMatrix(refB, {k, n}, {mb, mb});
  Matrix<T, Device::CPU> mat_c = setMatrix(refC, {m, n}, {mb, mb});

  dlaf::multiplication::generalSubMatrix(a, b, opA, opB, alpha, mat_a, mat_b, beta, mat_c);

  CHECK_MATRIX_NEAR(refResult, mat_c, 40 * (mat_c.size().rows() + 1) * TypeUtilities<T>::error,
                    40 * (mat_c.size().rows() + 1) * TypeUtilities<T>::error);
}

TYPED_TEST(GeneralMultiplicationTestMC, CorrectnessLocal) {
  for (const auto opA : blas_ops) {
    for (const auto opB : blas_ops) {
      // Note: not yet implemented
      if (opA != blas::Op::NoTrans || opB != blas::Op::NoTrans)
        continue;

      for (const auto& [m, n, k, mb, a, b] : sizes) {
        const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .5);
        const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
        testGeneralMultiplication<TypeParam, Backend::MC, Device::CPU>(a, b, opA, opB, alpha, beta, m, n,
                                                                       k, mb);
      }
    }
  }
}
