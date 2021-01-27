//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/matrix_view.h"

#include <vector>
#include "gtest/gtest.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_futures.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class MatrixMirrorTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(MatrixMirrorTest, MatrixElementTypes);

struct TestSizes {
  LocalElementSize size;
  TileElementSize block_size;
};

const std::vector<TestSizes> sizes_tests({
    {{0, 0}, {11, 13}},
    {{3, 0}, {1, 2}},
    {{0, 1}, {7, 32}},
    {{15, 18}, {5, 9}},
    {{6, 6}, {2, 2}},
    {{3, 4}, {24, 15}},
    {{16, 24}, {3, 5}},
});

GlobalElementSize globalTestSize(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

template <typename T, Device Target, Device Source>
void basicsTest(CommunicatorGrid const& comm_grid, TestSizes const& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Matrix<T, Source> mat(size, test.block_size, comm_grid);
  MatrixMirror<T, Target, Source> mat_mirror(mat);
  EXPECT_EQ(mat.distribution(), mat_mirror.get().distribution());
}

TYPED_TEST(MatrixMirrorTest, Basics) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      basicsTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_CUDA
      basicsTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      basicsTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      basicsTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void syncTest(CommunicatorGrid const& comm_grid, TestSizes const& test) {
  BaseType<T> offset = 0.0;
  auto el = [&offset](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024. + offset, j - i / 128. + offset);
  };

  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

  // CPU support matrices for setting values to source and target matrices
  Matrix<T, Device::CPU> mat_source_cpu(size, test.block_size, comm_grid);
  Matrix<T, Device::CPU> mat_target_cpu(size, test.block_size, comm_grid);

  set(mat_source_cpu, el);

  Matrix<T, Source> mat(size, test.block_size, comm_grid);
  copy(mat_source_cpu, mat);

  {
    MatrixMirror<T, Target, Source> mat_mirror(mat);

    copy(mat, mat_source_cpu);
    CHECK_MATRIX_EQ(el, mat_source_cpu);
    copy(mat_mirror.get(), mat_target_cpu);
    CHECK_MATRIX_EQ(el, mat_target_cpu);

    offset = 1.0;
    set(mat_source_cpu, el);
    copy(mat_source_cpu, mat);

    mat_mirror.syncSourceToTarget();

    copy(mat_mirror.get(), mat_target_cpu);
    CHECK_MATRIX_EQ(el, mat_target_cpu);

    offset = 2.0;
    set(mat_target_cpu, el);
    copy(mat_target_cpu, mat_mirror.get());

    mat_mirror.syncTargetToSource();

    copy(mat, mat_source_cpu);
    CHECK_MATRIX_EQ(el, mat_source_cpu);

    offset = 3.0;
    set(mat_target_cpu, el);
    copy(mat_target_cpu, mat_mirror.get());
  }

  copy(mat, mat_source_cpu);
  CHECK_MATRIX_EQ(el, mat_source_cpu);
}

TYPED_TEST(MatrixMirrorTest, Sync) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      syncTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_CUDA
      syncTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      syncTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      syncTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void syncConstTest(CommunicatorGrid const& comm_grid, TestSizes const& test) {
  BaseType<T> offset = 0.0;
  auto el = [&offset](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024. + offset, j - i / 128. + offset);
  };

  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

  // CPU support matrices for setting values to source and target matrices
  Matrix<T, Device::CPU> mat_source_cpu(size, test.block_size, comm_grid);
  Matrix<T, Device::CPU> mat_target_cpu(size, test.block_size, comm_grid);

  set(mat_source_cpu, el);

  Matrix<T, Source> mat_nonconst(size, test.block_size, comm_grid);
  copy(mat_source_cpu, mat_nonconst);
  Matrix<const T, Source> mat(std::move(mat_nonconst));

  MatrixMirror<const T, Target, Source> mat_mirror(mat);

  copy(mat, mat_source_cpu);
  CHECK_MATRIX_EQ(el, mat_source_cpu);
  copy(mat_mirror.get(), mat_target_cpu);
  CHECK_MATRIX_EQ(el, mat_target_cpu);
}

TYPED_TEST(MatrixMirrorTest, SyncConst) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      syncTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_CUDA
      syncTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      syncTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      syncTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void sameDeviceTest(CommunicatorGrid const& comm_grid, TestSizes const& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

  Matrix<T, Source> mat(size, test.block_size, comm_grid);
  MatrixMirror<T, Target, Source> mat_mirror(mat);

  // We assume that the distribution of mat and mat_mirror are identical here.
  // That they actually are equal is tested in a separate test.
  const auto& distribution = mat.distribution();
  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  for (SizeType j = 0; j < local_tile_cols; ++j) {
    for (SizeType i = 0; i < local_tile_rows; ++i) {
      LocalTileIndex idx(i, j);
      EXPECT_EQ(mat.read(idx).get().ptr(), mat_mirror.get().read(idx).get().ptr());
    }
  }
}

TYPED_TEST(MatrixMirrorTest, SameDevicesSameMemory) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      sameDeviceTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_CUDA
      sameDeviceTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

#ifdef DLAF_WITH_CUDA
template <typename T, Device Target, Device Source>
void differentDeviceTest(CommunicatorGrid const& comm_grid, TestSizes const& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

  Matrix<T, Source> mat(size, test.block_size, comm_grid);
  MatrixMirror<T, Target, Source> mat_mirror(mat);

  // We assume that the distribution of mat and mat_mirror are identical here.
  // That they actually are equal is tested in a separate test.
  const auto& distribution = mat.distribution();
  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  for (SizeType j = 0; j < local_tile_cols; ++j) {
    for (SizeType i = 0; i < local_tile_rows; ++i) {
      LocalTileIndex idx(i, j);
      EXPECT_NE(mat.read(idx).get().ptr(), mat_mirror.get().read(idx).get().ptr());
    }
  }
}

TYPED_TEST(MatrixMirrorTest, DifferentDevicesDifferentMemory) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      differentDeviceTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      differentDeviceTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
    }
  }
}
#endif
