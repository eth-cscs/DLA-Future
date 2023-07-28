//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_senders.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

using pika::this_thread::experimental::sync_wait;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct MatrixMirrorTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(MatrixMirrorTest, MatrixElementTypes);

struct TestSizes {
  LocalElementSize size;
  TileElementSize block_size;
  TileElementSize tile_size;
};

const std::vector<TestSizes> sizes_tests({
    {{0, 0}, {11, 13}, {11, 13}},
    {{3, 0}, {1, 2}, {1, 1}},
    {{0, 1}, {7, 32}, {7, 8}},
    {{15, 18}, {5, 9}, {5, 3}},
    {{6, 6}, {2, 2}, {1, 1}},
    {{3, 4}, {24, 15}, {8, 15}},
    {{16, 24}, {3, 5}, {3, 5}},
});

GlobalElementSize globalTestSize(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

template <typename T, Device Target, Device Source>
void basicsTest(const CommunicatorGrid& comm_grid, const TestSizes& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
  Matrix<T, Source> mat(dist);
  MatrixMirror<T, Target, Source> mat_mirror(mat);
  EXPECT_EQ(mat.distribution(), mat_mirror.get().distribution());
}

TYPED_TEST(MatrixMirrorTest, Basics) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      basicsTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_GPU
      basicsTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      basicsTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      basicsTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void getTest(const CommunicatorGrid& comm_grid, const TestSizes& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
  Matrix<T, Source> mat(dist);

  {
    MatrixMirror<T, Target, Source> mat_mirror(mat);

    if constexpr (Target == Source) {
      EXPECT_EQ(&mat, &mat_mirror.get());
    }
    else {
      static_assert(!std::is_same_v<decltype(mat), std::decay_t<decltype(mat_mirror.get())>>);
    }
  }

  {
    Matrix<const T, Source>& mat_const = mat;
    MatrixMirror<const T, Target, Source> mat_mirror(mat_const);

    if constexpr (Target == Source) {
      EXPECT_EQ(&mat, &mat_mirror.get());
    }
    else {
      static_assert(!std::is_same_v<decltype(mat), std::decay_t<decltype(mat_mirror.get())>>);
    }
  }
}

TYPED_TEST(MatrixMirrorTest, Get) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      getTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_GPU
      getTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      getTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      getTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void getSourceTest(const CommunicatorGrid& comm_grid, const TestSizes& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
  Matrix<T, Source> mat(dist);

  {
    MatrixMirror<T, Target, Source> mat_mirror(mat);
    EXPECT_EQ(&mat, &mat_mirror.getSource());
  }

  {
    Matrix<const T, Source>& mat_const = mat;
    MatrixMirror<const T, Target, Source> mat_mirror(mat_const);
    EXPECT_EQ(&mat_const, &mat_mirror.getSource());
  }
}

TYPED_TEST(MatrixMirrorTest, GetSource) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      getSourceTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_GPU
      getSourceTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      getSourceTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      getSourceTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void copyTest(const CommunicatorGrid& comm_grid, const TestSizes& test) {
  BaseType<T> offset = 0.0;
  auto el = [&offset](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024. + offset, j - i / 128. + offset);
  };

  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});

  // CPU support matrices for setting values to source and target matrices
  Matrix<T, Device::CPU> mat_source_cpu(dist);
  Matrix<T, Device::CPU> mat_target_cpu(dist);

  set(mat_source_cpu, el);

  Matrix<T, Source> mat(dist);
  matrix::copy(mat_source_cpu, mat);

  {
    MatrixMirror<T, Target, Source> mat_mirror(mat);

    copy(mat, mat_source_cpu);
    CHECK_MATRIX_EQ(el, mat_source_cpu);
    matrix::copy(mat_mirror.get(), mat_target_cpu);
    CHECK_MATRIX_EQ(el, mat_target_cpu);

    offset = 1.0;
    set(mat_source_cpu, el);
    copy(mat_source_cpu, mat);

    mat_mirror.copySourceToTarget();

    copy(mat_mirror.get(), mat_target_cpu);
    CHECK_MATRIX_EQ(el, mat_target_cpu);

    offset = 2.0;
    set(mat_target_cpu, el);
    copy(mat_target_cpu, mat_mirror.get());

    mat_mirror.copyTargetToSource();

    copy(mat, mat_source_cpu);
    CHECK_MATRIX_EQ(el, mat_source_cpu);

    offset = 3.0;
    set(mat_target_cpu, el);
    copy(mat_target_cpu, mat_mirror.get());
  }

  copy(mat, mat_source_cpu);
  CHECK_MATRIX_EQ(el, mat_source_cpu);
}

TYPED_TEST(MatrixMirrorTest, Copy) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      copyTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_GPU
      copyTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      copyTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      copyTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void copyConstTest(const CommunicatorGrid& comm_grid, const TestSizes& test) {
  BaseType<T> offset = 0.0;
  auto el = [&offset](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024. + offset, j - i / 128. + offset);
  };

  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});

  // CPU support matrices for setting values to source and target matrices
  Matrix<T, Device::CPU> mat_source_cpu(dist);
  Matrix<T, Device::CPU> mat_target_cpu(dist);

  set(mat_source_cpu, el);

  Matrix<T, Source> mat_nonconst(dist);
  copy(mat_source_cpu, mat_nonconst);
  Matrix<const T, Source> mat(std::move(mat_nonconst));

  MatrixMirror<const T, Target, Source> mat_mirror(mat);

  copy(mat, mat_source_cpu);
  CHECK_MATRIX_EQ(el, mat_source_cpu);
  copy(mat_mirror.get(), mat_target_cpu);
  CHECK_MATRIX_EQ(el, mat_target_cpu);
}

TYPED_TEST(MatrixMirrorTest, CopyConst) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      copyConstTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_GPU
      copyConstTest<TypeParam, Device::CPU, Device::GPU>(comm_grid, test);
      copyConstTest<TypeParam, Device::GPU, Device::CPU>(comm_grid, test);
      copyConstTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

template <typename T, Device Target, Device Source>
void sameDeviceTest(const CommunicatorGrid& comm_grid, const TestSizes& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});

  Matrix<T, Source> mat(dist);
  MatrixMirror<T, Target, Source> mat_mirror(mat);

  // We assume that the distribution of mat and mat_mirror are identical here.
  // That they actually are equal is tested in a separate test.
  const auto& distribution = mat.distribution();
  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  for (SizeType j = 0; j < local_tile_cols; ++j) {
    for (SizeType i = 0; i < local_tile_rows; ++i) {
      LocalTileIndex idx(i, j);
      EXPECT_EQ(sync_wait(mat.read(idx)).get().ptr(), sync_wait(mat_mirror.get().read(idx)).get().ptr());
    }
  }
}

TYPED_TEST(MatrixMirrorTest, SameDevicesSameMemory) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      sameDeviceTest<TypeParam, Device::CPU, Device::CPU>(comm_grid, test);
#ifdef DLAF_WITH_GPU
      sameDeviceTest<TypeParam, Device::GPU, Device::GPU>(comm_grid, test);
#endif
    }
  }
}

#ifdef DLAF_WITH_GPU
template <typename T, Device Target, Device Source>
void differentDeviceTest(const CommunicatorGrid& comm_grid, const TestSizes& test) {
  GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
  Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});

  Matrix<T, Source> mat(dist);
  MatrixMirror<T, Target, Source> mat_mirror(mat);

  // We assume that the distribution of mat and mat_mirror are identical here.
  // That they actually are equal is tested in a separate test.
  const auto& distribution = mat.distribution();
  const SizeType local_tile_rows = distribution.localNrTiles().rows();
  const SizeType local_tile_cols = distribution.localNrTiles().cols();

  for (SizeType j = 0; j < local_tile_cols; ++j) {
    for (SizeType i = 0; i < local_tile_rows; ++i) {
      LocalTileIndex idx(i, j);
      EXPECT_NE(sync_wait(mat.read(idx)).get().ptr(), sync_wait(mat_mirror.get().read(idx)).get().ptr());
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
