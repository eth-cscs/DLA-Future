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

#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/util_matrix.h>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/index.h"

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/util_types.h>

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
struct MatrixCopyTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(MatrixCopyTest, MatrixElementTypes);

struct FullMatrixCopyConfig {
  LocalElementSize size;
  TileElementSize block_size;
  TileElementSize tile_size;
};

const std::vector<FullMatrixCopyConfig> sizes_tests({
    {{0, 0}, {11, 13}, {11, 13}},
    {{3, 0}, {1, 2}, {1, 1}},
    {{0, 1}, {7, 32}, {7, 8}},
    {{15, 18}, {5, 9}, {5, 3}},
    {{6, 6}, {2, 2}, {2, 2}},
    {{3, 4}, {24, 15}, {8, 15}},
    {{16, 24}, {3, 5}, {3, 5}},
});

GlobalElementSize globalTestSize(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

template <class T>
T inputValues(const GlobalElementIndex& index) noexcept {
  const SizeType i = index.row();
  const SizeType j = index.col();
  return TypeUtilities<T>::element(i + j / 1024., j - i / 128.);
};

template <class T>
T outputValues(const GlobalElementIndex&) noexcept {
  return TypeUtilities<T>::element(13, 26);
};

TYPED_TEST(MatrixCopyTest, FullMatrixCPU) {
  using dlaf::matrix::util::set;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      const GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      const Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      const LayoutInfo layout = tileLayout(distribution.local_size(), distribution.block_size());

      memory::MemoryView<TypeParam, Device::CPU> mem_src(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> mat_src(distribution, layout, mem_src());
      set(mat_src, inputValues<TypeParam>);
      Matrix<const TypeParam, Device::CPU> mat_src_const = std::move(mat_src);

      memory::MemoryView<TypeParam, Device::CPU> mem_dst(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> mat_dst(distribution, layout, mem_dst());
      set(mat_dst, outputValues<TypeParam>);

      copy(mat_src_const, mat_dst);

      CHECK_MATRIX_NEAR(inputValues<TypeParam>, mat_dst, 0, TypeUtilities<TypeParam>::error);
    }
  }
}

#if DLAF_WITH_GPU
TYPED_TEST(MatrixCopyTest, FullMatrixGPU) {
  using dlaf::matrix::util::set;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      const GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      const Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      const LayoutInfo layout = tileLayout(distribution.local_size(), distribution.block_size());

      memory::MemoryView<TypeParam, Device::CPU> mem_src(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> mat_src(distribution, layout, mem_src());
      set(mat_src, inputValues<TypeParam>);
      Matrix<const TypeParam, Device::CPU> mat_src_const = std::move(mat_src);

      memory::MemoryView<TypeParam, Device::GPU> mem_gpu1(layout.minMemSize());
      Matrix<TypeParam, Device::GPU> mat_gpu1(distribution, layout, mem_gpu1());

      memory::MemoryView<TypeParam, Device::GPU> mem_gpu2(layout.minMemSize());
      Matrix<TypeParam, Device::GPU> mat_gpu2(distribution, layout, mem_gpu2());

      memory::MemoryView<TypeParam, Device::CPU> mem_dst(layout.minMemSize());
      Matrix<TypeParam, Device::CPU> mat_dst(distribution, layout, mem_dst());
      set(mat_dst, outputValues<TypeParam>);

      copy(mat_src_const, mat_gpu1);
      copy(mat_gpu1, mat_gpu2);
      copy(mat_gpu2, mat_dst);

      CHECK_MATRIX_NEAR(inputValues<TypeParam>, mat_dst, 0, TypeUtilities<TypeParam>::error);
    }
  }
}
#endif

struct SubMatrixCopyConfig {
  GlobalElementSize full_in;
  GlobalElementSize full_out;

  TileElementSize tile_size;

  GlobalElementIndex sub_origin_in;
  GlobalElementIndex sub_origin_out;

  GlobalElementSize sub_size;
};

comm::Index2D alignSubRankIndex(const Distribution& dist_in, const GlobalElementIndex& offset_in,
                                const GlobalElementIndex& offset_out) {
  const auto pos_mod = [](const auto& a, const auto& b) {
    const auto mod = a % b;
    return (mod >= 0) ? mod : (mod + b);
  };

  const comm::Size2D grid_size = dist_in.grid_size();
  const comm::Index2D sub_rank(dist_in.rank_global_element<Coord::Row>(offset_in.row()),
                               dist_in.rank_global_element<Coord::Col>(offset_in.col()));
  const GlobalTileIndex offset_rank(offset_out.row() / dist_in.block_size().rows(),
                                    offset_out.col() / dist_in.block_size().cols());

  return {pos_mod(sub_rank.row() - static_cast<comm::IndexT_MPI>(offset_rank.row()), grid_size.rows()),
          pos_mod(sub_rank.col() - static_cast<comm::IndexT_MPI>(offset_rank.col()), grid_size.cols())};
}

bool isFullMatrix(const Distribution& dist_full, const GlobalElementIndex& sub_origin,
                  const GlobalElementSize& sub_size) noexcept {
  return sub_origin == GlobalElementIndex{0, 0} && sub_size == dist_full.size();
}

template <class ElementGetter>
auto subValues(ElementGetter&& fullValues, const GlobalElementIndex& offset) {
  return [fullValues, offset = sizeFromOrigin(offset)](const GlobalElementIndex& ij) {
    return fullValues(ij + offset);
  };
}

template <class OutsideElementGetter, class InsideElementGetter>
auto mixValues(const GlobalElementIndex& offset, const GlobalElementSize& sub_size,
               InsideElementGetter&& insideValues, OutsideElementGetter&& outsideValues) {
  return [outsideValues, insideValues, offset, sub_size](const GlobalElementIndex& ij) {
    if (ij.isInSub(offset, sub_size))
      return insideValues(ij - common::sizeFromOrigin(offset));
    else
      return outsideValues(ij);
  };
}

const std::vector<SubMatrixCopyConfig> sub_configs{
    // full-matrix
    {{10, 10}, {10, 10}, {3, 3}, {0, 0}, {0, 0}, {10, 10}},
    // sub-matrix
    {{10, 10}, {10, 10}, {3, 3}, {3, 3}, {3, 3}, {6, 6}},
    {{10, 10}, {10, 10}, {3, 3}, {3, 3}, {0, 0}, {6, 6}},
    {{13, 26}, {26, 13}, {5, 5}, {7, 3}, {17, 8}, {4, 5}},
};

template <class T>
void testSubMatrix(const SubMatrixCopyConfig& test, const matrix::Distribution& dist_in,
                   const matrix::Distribution& dist_out) {
  const LayoutInfo layout_in = tileLayout(dist_in.local_size(), dist_in.block_size());
  const LayoutInfo layout_out = tileLayout(dist_out.local_size(), dist_out.block_size());

  memory::MemoryView<T, Device::CPU> mem_in(layout_in.minMemSize());
  memory::MemoryView<T, Device::CPU> mem_out(layout_out.minMemSize());

  Matrix<T, Device::CPU> mat_in(dist_in, layout_in, mem_in());
  Matrix<T, Device::CPU> mat_out(dist_out, layout_out, mem_out());

  // Note: currently `subPipeline`-ing does not support sub-matrices
  if (isFullMatrix(dist_in, test.sub_origin_in, test.sub_size)) {
    set(mat_in, inputValues<T>);
    set(mat_out, outputValues<T>);

    {
      Matrix<const T, Device::CPU> mat_sub_src_const = mat_in.subPipelineConst();
      Matrix<T, Device::CPU> mat_sub_dst = mat_out.subPipeline();

      copy(mat_sub_src_const, mat_sub_dst);
    }

    CHECK_MATRIX_NEAR(inputValues<T>, mat_out, 0, TypeUtilities<T>::error);
  }

  // MatrixRef
  set(mat_in, inputValues<T>);
  set(mat_out, outputValues<T>);

  using matrix::internal::MatrixRef;
  MatrixRef<const T, Device::CPU> mat_sub_src(mat_in, {test.sub_origin_in, test.sub_size});
  MatrixRef<T, Device::CPU> mat_sub_dst(mat_out, {test.sub_origin_out, test.sub_size});

  copy(mat_sub_src, mat_sub_dst);

  const auto subMatrixValues = subValues(inputValues<T>, test.sub_origin_in);
  CHECK_MATRIX_NEAR(subMatrixValues, mat_sub_dst, 0, TypeUtilities<T>::error);

  const auto fullMatrixWithSubMatrixValues =
      mixValues(test.sub_origin_out, test.sub_size, subMatrixValues, outputValues<T>);
  CHECK_MATRIX_NEAR(fullMatrixWithSubMatrixValues, mat_out, 0, TypeUtilities<T>::error);
}

TYPED_TEST(MatrixCopyTest, SubMatrixCPULocal) {
  for (const auto& test : sub_configs) {
    const Distribution dist_in({test.full_in.rows(), test.full_in.cols()}, test.tile_size);
    const Distribution dist_out({test.full_out.rows(), test.full_out.cols()}, test.tile_size);

    testSubMatrix<TypeParam>(test, dist_in, dist_out);
  }
}

TYPED_TEST(MatrixCopyTest, SubMatrixCPUDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sub_configs) {
      const comm::Index2D in_src_rank(0, 0);
      const Distribution dist_in(test.full_in, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                 in_src_rank);

      const comm::Index2D out_src_rank =
          alignSubRankIndex(dist_in, test.sub_origin_in, test.sub_origin_out);
      const Distribution dist_out(test.full_out, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                  out_src_rank);

      EXPECT_EQ(dist_in.template rank_global_element<Coord::Row>(test.sub_origin_in.row()),
                dist_out.template rank_global_element<Coord::Row>(test.sub_origin_out.row()));
      EXPECT_EQ(dist_in.template rank_global_element<Coord::Col>(test.sub_origin_in.col()),
                dist_out.template rank_global_element<Coord::Col>(test.sub_origin_out.col()));

      testSubMatrix<TypeParam>(test, dist_in, dist_out);
    }
  }
}

#ifdef DLAF_WITH_GPU
template <class T>
void testSubMatrixOnGPU(const SubMatrixCopyConfig& test, const matrix::Distribution& dist_in,
                        const matrix::Distribution& dist_out) {
  const LayoutInfo layout_in = tileLayout(dist_in.local_size(), dist_in.block_size());
  const LayoutInfo layout_out = tileLayout(dist_out.local_size(), dist_out.block_size());

  // CPU
  memory::MemoryView<T, Device::CPU> mem_in(layout_in.minMemSize());
  memory::MemoryView<T, Device::CPU> mem_out(layout_out.minMemSize());

  Matrix<T, Device::CPU> mat_in(dist_in, layout_in, mem_in());
  Matrix<T, Device::CPU> mat_out(dist_out, layout_out, mem_out());

  // GPU
  memory::MemoryView<T, Device::GPU> mem_in_gpu(layout_in.minMemSize());
  memory::MemoryView<T, Device::GPU> mem_out_gpu(layout_out.minMemSize());

  Matrix<T, Device::GPU> mat_in_gpu(dist_in, layout_in, mem_in_gpu());
  Matrix<T, Device::GPU> mat_out_gpu(dist_out, layout_out, mem_out_gpu());

  // Note: currently `subPipeline`-ing does not support sub-matrices
  if (isFullMatrix(dist_in, test.sub_origin_in, test.sub_size)) {
    set(mat_in, inputValues<T>);
    set(mat_out, outputValues<T>);

    {
      Matrix<const T, Device::CPU> mat_sub_src_const = mat_in.subPipelineConst();
      Matrix<T, Device::GPU> mat_sub_gpu1 = mat_in_gpu.subPipeline();
      Matrix<T, Device::GPU> mat_sub_gpu2 = mat_out_gpu.subPipeline();
      Matrix<T, Device::CPU> mat_sub_dst = mat_out.subPipeline();

      copy(mat_sub_src_const, mat_sub_gpu1);
      copy(mat_sub_gpu1, mat_sub_gpu2);
      copy(mat_sub_gpu2, mat_sub_dst);
    }
    CHECK_MATRIX_NEAR(inputValues<T>, mat_out, 0, TypeUtilities<T>::error);
  }

  // MatrixRef
  set(mat_in, inputValues<T>);
  set(mat_out, outputValues<T>);

  using matrix::internal::MatrixRef;
  MatrixRef<const T, Device::CPU> mat_sub_src(mat_in, {test.sub_origin_in, test.sub_size});
  MatrixRef<T, Device::GPU> mat_sub_gpu1(mat_in_gpu, {test.sub_origin_in, test.sub_size});
  MatrixRef<T, Device::GPU> mat_sub_gpu2(mat_out_gpu, {test.sub_origin_out, test.sub_size});
  MatrixRef<T, Device::CPU> mat_sub_dst(mat_out, {test.sub_origin_out, test.sub_size});

  copy(mat_sub_src, mat_sub_gpu1);
  copy(mat_sub_gpu1, mat_sub_gpu2);
  copy(mat_sub_gpu2, mat_sub_dst);

  const auto subMatrixValues = subValues(inputValues<T>, test.sub_origin_in);
  CHECK_MATRIX_NEAR(subMatrixValues, mat_sub_dst, 0, TypeUtilities<T>::error);

  const auto fullMatrixWithSubMatrixValues =
      mixValues(test.sub_origin_out, test.sub_size, subMatrixValues, outputValues<T>);
  CHECK_MATRIX_NEAR(fullMatrixWithSubMatrixValues, mat_out, 0, TypeUtilities<T>::error);
}

TYPED_TEST(MatrixCopyTest, SubMatrixGPULocal) {
  for (const auto& test : sub_configs) {
    const Distribution dist_in({test.full_in.rows(), test.full_in.cols()}, test.tile_size);
    const Distribution dist_out({test.full_out.rows(), test.full_out.cols()}, test.tile_size);

    testSubMatrixOnGPU<TypeParam>(test, dist_in, dist_out);
  }
}

TYPED_TEST(MatrixCopyTest, SubMatrixGPUDistributed) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sub_configs) {
      const comm::Index2D in_src_rank(0, 0);
      const Distribution dist_in(test.full_in, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                 in_src_rank);

      const comm::Index2D out_src_rank =
          alignSubRankIndex(dist_in, test.sub_origin_in, test.sub_origin_out);
      const Distribution dist_out(test.full_out, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                  out_src_rank);

      EXPECT_EQ(dist_in.template rank_global_element<Coord::Row>(test.sub_origin_in.row()),
                dist_out.template rank_global_element<Coord::Row>(test.sub_origin_out.row()));
      EXPECT_EQ(dist_in.template rank_global_element<Coord::Col>(test.sub_origin_in.col()),
                dist_out.template rank_global_element<Coord::Col>(test.sub_origin_out.col()));

      testSubMatrixOnGPU<TypeParam>(test, dist_in, dist_out);
    }
  }
}
#endif
