//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/common/index2d.h>
#include <dlaf/eigensolver/reduction_to_trid.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/matrix_local.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_local.h>
#include <dlaf_test/matrix/util_tile.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::test;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;

template <class T>
struct ReductionToTridTest : public TestWithCommGrids {};

template <class T>
using ReductionToTridTestMC = ReductionToTridTest<T>;

TYPED_TEST_SUITE(ReductionToTridTestMC, MatrixElementTypes);

#ifdef DLAF_WITH_GPU
template <class T>
using ReductionToTridTestGPU = ReductionToTridTest<T>;

TYPED_TEST_SUITE(ReductionToTridTestGPU, MatrixElementTypes);
#endif

struct config_t {
  LocalElementSize size;
  TileElementSize block_size;
};

std::vector<config_t> configs{
    {{0, 0}, {3, 3}},   {{3, 3}, {3, 3}},  // single tile (nothing to do)
    {{12, 12}, {3, 3}},  // tile always full size (less room for distribution over ranks)
    {{13, 13}, {3, 3}},  // tile incomplete
    {{24, 24}, {3, 3}},  // tile always full size (more room for distribution)
    {{40, 40}, {5, 5}},
};

template <class T, Backend B, Device D>
void testReductionToTridLocal(const LocalElementSize, const TileElementSize) {}

TYPED_TEST(ReductionToTridTestMC, CorrectnessLocal) {
  for (const auto& config : configs) {
    const auto& [size, block_size] = config;
    testReductionToTridLocal<TypeParam, Backend::MC, Device::CPU>(size, block_size);
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(ReductionToTridTestGPU, CorrectnessLocal) {
  for (const auto& config : configs) {
    const auto& [size, block_size] = config;

    testReductionToTridLocal<TypeParam, Backend::GPU, Device::GPU>(size, block_size);
  }
}

#endif
