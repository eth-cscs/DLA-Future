//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/blas/tile.h>

#include "test_scal/test_scal.h"
#include "test_add/test_scal.h"

#include <gtest/gtest.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;

const std::vector<blas::Diag> blas_diags({blas::Diag::Unit, blas::Diag::NonUnit});
const std::vector<blas::Op> blas_ops({blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans});
const std::vector<blas::Side> blas_sides({blas::Side::Left, blas::Side::Right});
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::Lower, blas::Uplo::Upper});

template <class T, Device D>
class TileOperationsTest : public ::testing::Test {};

template <class T>
using TileOperationsTestMC = TileOperationsTest<T, Device::CPU>;

TYPED_TEST_SUITE(TileOperationsTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using TileOperationsTestGPU = TileOperationsTest<T, Device::GPU>;

TYPED_TEST_SUITE(TileOperationsTestGPU, MatrixElementTypes);
#endif

// Tuple elements:  m, n, k, extra_lda, extra_ldb, extra_ldc
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType, SizeType>> gemm_sizes = {
    {0, 0, 0, 0, 0, 0},                                               // all 0 sizes
    {7, 0, 0, 3, 1, 0},  {0, 5, 0, 0, 0, 1},    {0, 0, 11, 1, 1, 2},  // two 0 sizes
    {0, 5, 13, 1, 0, 1}, {7, 0, 4, 1, 2, 0},    {3, 11, 0, 0, 1, 0},  // one 0 size
    {1, 1, 1, 0, 3, 0},  {1, 12, 1, 1, 0, 7},   {17, 12, 16, 1, 3, 0}, {11, 23, 8, 0, 3, 4},
    {6, 9, 12, 1, 1, 1}, {32, 32, 32, 0, 0, 0}, {32, 32, 32, 4, 5, 7}, {128, 128, 128, 0, 0, 0},
};
//Cosa metto al posto di Gemm riga 51
TYPED_TEST(TileOperationsTestMC, Gemm) {
  using Type = TypeParam;
//i cicli for non penso siano corretti, es. riga 56
  for (const auto op_a : blas_ops) {
    for (const auto op_b : blas_ops) {
      for (const auto& [m, n, k, extra_lda, extra_ldb, extra_ldc] : gemm_sizes) {
        // Test a and b const Tiles.
        dlaf::test::testScal<Device::CPU, Type>(op_a, m, n, extra_lda);

        // Test a and b non const Tiles.
        dlaf::test::testScal<Device::CPU, Type>(op_a, m, n, extra_lda);
      }
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsTestGPU, Gemm) {
  using Type = TypeParam;

  for (const auto op_a : blas_ops) {
    for (const auto op_b : blas_ops) {
      for (const auto& [m, n, k, extra_lda, extra_ldb, extra_ldc] : gemm_sizes) {
        // Test a and b const Tiles.
        dlaf::test::testScal<Device::GPU, Type>(op_a, m, n, extra_lda);

        // Test a and b non const Tiles.
       dlaf::test::testScal<Device::CPU, Type, Type>(op_a, m, n, extra_lda);
      }
    }
  }
}
#endif
