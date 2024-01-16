//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/blas/tile.h>

#include "test_blas_tile/test_add.h"
#include "test_blas_tile/test_scal.h"

#include <gtest/gtest.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace testing;
template <class T, Device D>
class TileOperationsTest : public ::testing::Test {};

template <class T>
using TileOperationsExtensionsTestMC = TileOperationsTest<T, Device::CPU>;

TYPED_TEST_SUITE(TileOperationsExtensionsTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using TileOperationsExtensionsTestGPU = TileOperationsTest<T, Device::GPU>;

TYPED_TEST_SUITE(TileOperationsExtensionsTestGPU, MatrixElementTypes);
#endif

// Tuple elements:  m, n, k, extra_lda, extra_ldb, extra_ldc
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType, SizeType, SizeType>> gemm_sizes = {
    {0, 0, 0, 0, 0, 0},                                               // all 0 sizes
    {7, 0, 0, 3, 1, 0},  {0, 5, 0, 0, 0, 1},    {0, 0, 11, 1, 1, 2},  // two 0 sizes
    {0, 5, 13, 1, 0, 1}, {7, 0, 4, 1, 2, 0},    {3, 11, 0, 0, 1, 0},  // one 0 size
    {1, 1, 1, 0, 3, 0},  {1, 12, 1, 1, 0, 7},   {17, 12, 16, 1, 3, 0}, {11, 23, 8, 0, 3, 4},
    {6, 9, 12, 1, 1, 1}, {32, 32, 32, 0, 0, 0}, {32, 32, 32, 4, 5, 7}, {128, 128, 128, 0, 0, 0},
};

// Tuple elements:  m, n, extra_lda
std::vector<std::tuple<SizeType, SizeType, SizeType>> scal_sizes = {
    {0, 0, 0, 0, 0, 0},                                               // all 0 sizes
    {7, 0, 0, 3, 1, 0},  {0, 5, 0, 0, 0, 1},    {0, 0, 11, 1, 1, 2},  // two 0 sizes
    {0, 5, 13, 1, 0, 1}, {7, 0, 4, 1, 2, 0},    {3, 11, 0, 0, 1, 0},  // one 0 size
    {1, 1, 1, 0, 3, 0},  {1, 12, 1, 1, 0, 7},   {17, 12, 16, 1, 3, 0}, {11, 23, 8, 0, 3, 4},
    {6, 9, 12, 1, 1, 1}, {32, 32, 32, 0, 0, 0}, {32, 32, 32, 4, 5, 7}, {128, 128, 128, 0, 0, 0},
};

// Tuple elements:  m, n, k, extra_lda, extra_ldb
std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> add_sizes = {
    {0, 0, 0, 0, 0, 0},                                               // all 0 sizes
    {7, 0, 0, 3, 1, 0},  {0, 5, 0, 0, 0, 1},    {0, 0, 11, 1, 1, 2},  // two 0 sizes
    {0, 5, 13, 1, 0, 1}, {7, 0, 4, 1, 2, 0},    {3, 11, 0, 0, 1, 0},  // one 0 size
    {1, 1, 1, 0, 3, 0},  {1, 12, 1, 1, 0, 7},   {17, 12, 16, 1, 3, 0}, {11, 23, 8, 0, 3, 4},
    {6, 9, 12, 1, 1, 1}, {32, 32, 32, 0, 0, 0}, {32, 32, 32, 4, 5, 7}, {128, 128, 128, 0, 0, 0},
};

TYPED_TEST(TileOperationsExtensionsTestMC, Scal) {
  using Type = TypeParam;
  for (const auto& [m, n, k, extra_lda] : scal_sizes) {
    // Test a and b const Tiles.
    dlaf::test::testScal<Device::CPU, Type>(m, n, extra_lda);
    // Test a and b non const Tiles.
    dlaf::test::testScal<Device::CPU, Type>(m, n, extra_lda);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsExtensionsTestGPU, Scal) {
  using Type = TypeParam;

  for (const auto& [m, n, k, extra_lda] : scal_sizes) {
    // Test a and b const Tiles.
    dlaf::test::testScal<Device::GPU, Type>(m, n, extra_lda);

    // Test a and b non const Tiles.
    dlaf::test::testScal<Device::GPU, Type>(m, n, extra_lda);
  }
}
#endif

TYPED_TEST(TileOperationsExtensionsTestMC, Add) {
  using Type = TypeParam;
  for (const auto& [m, n, k, extra_lda, extra_ldb] : add_sizes) {
    // Test a and b const Tiles.
    dlaf::test::testAdd<Device::CPU, Type>(m, n, extra_lda, extra_ldb);
    // Test a and b non const Tiles.
    dlaf::test::testAdd<Device::CPU, Type>(m, n, extra_lda, extra_ldb);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(TileOperationsExtensionsTestGPU, Add) {
  using Type = TypeParam;

  for (const auto& [m, n, k, extra_lda, extra_ldb] : add_sizes) {
    // Test a and b const Tiles.
    dlaf::test::testAdd<Device::GPU, Type>(m, n, extra_lda, extra_ldb);

    // Test a and b non const Tiles.
    dlaf::test::testAdd<Device::GPU, Type>(m, n, extra_lda, extra_ldb);
  }
}
#endif
