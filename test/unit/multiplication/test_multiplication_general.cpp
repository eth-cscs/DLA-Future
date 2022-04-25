//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/common/assert.h"
#include "dlaf/multiplication/general.h"

#include <gtest/gtest.h>
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;

template <class T>
struct GeneralMultiplicationTestMC : public ::testing::Test {};

TYPED_TEST_SUITE(GeneralMultiplicationTestMC, MatrixElementTypes);

const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});

const std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType>> sizes;

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

template <class T, Backend B, Device D>
void testGeneralMultiplication(const blas::Op opA, const blas::Op opB, const T alpha, const T beta,
                               SizeType m, SizeType n, SizeType k, SizeType mb, SizeType nb) {
  dlaf::internal::silenceUnusedWarningFor(opA, opB, alpha, beta, m, n, k, mb, nb);
}

TYPED_TEST(GeneralMultiplicationTestMC, CorrectnessLocal) {
  for (const auto opA : blas_ops) {
    for (const auto opB : blas_ops) {
      for (const auto& [m, n, k, mb, nb] : sizes) {
        const TypeParam alpha = TypeUtilities<TypeParam>::element(-1.3, .7);
        const TypeParam beta = TypeUtilities<TypeParam>::element(-2.6, .7);
        testGeneralMultiplication<TypeParam, Backend::MC, Device::CPU>(opA, opB, alpha, beta, m, n, k,
                                                                       mb, nb);
      }
    }
  }
}
