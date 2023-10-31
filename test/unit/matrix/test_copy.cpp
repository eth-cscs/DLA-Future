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

// TODO local test

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
  using MemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::CPU>;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;
  using MatrixConstT = dlaf::Matrix<const TypeParam, Device::CPU>;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      using dlaf::matrix::util::set;

      const GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      const Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      const LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      MemoryViewT mem_src(layout.minMemSize());
      MatrixT mat_src = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, mem_src());
      set(mat_src, inputValues<TypeParam>);
      MatrixConstT mat_src_const = std::move(mat_src);

      MemoryViewT mem_dst(layout.minMemSize());
      MatrixT mat_dst = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, mem_dst());
      set(mat_dst, outputValues<TypeParam>);

      copy(mat_src_const, mat_dst);

      CHECK_MATRIX_NEAR(inputValues<TypeParam>, mat_dst, 0, TypeUtilities<TypeParam>::error);
    }
  }
}

#if DLAF_WITH_GPU
TYPED_TEST(MatrixCopyTest, FullMatrixGPU) {
  using MemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::CPU>;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;
  using MatrixConstT = dlaf::Matrix<const TypeParam, Device::CPU>;
  using GPUMemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::GPU>;
  using GPUMatrixT = dlaf::Matrix<TypeParam, Device::GPU>;

  using dlaf::matrix::util::set;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      MemoryViewT mem_src(layout.minMemSize());
      MatrixT mat_src = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, mem_src());
      set(mat_src, inputValues<TypeParam>);
      MatrixConstT mat_src_const = std::move(mat_src);

      GPUMemoryViewT mem_gpu1(layout.minMemSize());
      GPUMatrixT mat_gpu1 =
          createMatrixFromTile<Device::GPU>(size, test.block_size, comm_grid, mem_gpu1());

      GPUMemoryViewT mem_gpu2(layout.minMemSize());
      GPUMatrixT mat_gpu2 =
          createMatrixFromTile<Device::GPU>(size, test.block_size, comm_grid, mem_gpu2());

      MemoryViewT mem_dst(layout.minMemSize());
      MatrixT mat_dst = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, mem_dst());
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

bool isFullMatrix(const Distribution& dist_full, const GlobalElementIndex& sub_origin,
                  const GlobalElementSize& sub_size) noexcept {
  return sub_origin == GlobalElementIndex{0, 0} && sub_size == dist_full.size();
}

bool isInSub(const GlobalElementIndex& ij, const GlobalElementIndex& origin,
             const GlobalElementSize& sub_size) noexcept {
  return ij.row() >= origin.row() && ij.col() >= origin.col() &&
         ij.isIn(sizeFromOrigin(origin + sub_size));
}

template <class ElementGetter>
auto subValues(ElementGetter&& fullValues, const GlobalElementIndex& offset) {
  return [fullValues, offset = sizeFromOrigin(offset)](const GlobalElementIndex& ij) {
    return fullValues(ij + offset);
  };
}

template <class OutsideElementGetter, class InsideElementGetter>
auto mixValues(OutsideElementGetter&& outsideValues, InsideElementGetter&& insideValues,
               const GlobalElementIndex& offset, const GlobalElementSize& sub_size) {
  return [outsideValues, insideValues, offset, sub_size](const GlobalElementIndex& ij) {
    if (isInSub(ij, offset, sub_size))
      return insideValues(ij);
    else
      return outsideValues(ij);
  };
}

const std::vector<SubMatrixCopyConfig> sub_configs{
    // full-matrix
    {{10, 10}, {10, 10}, {3, 3}, {0, 0}, {0, 0}, {10, 10}},
    // sub-matrix
    {{10, 10}, {10, 10}, {3, 3}, {3, 3}, {3, 3}, {6, 6}},
};

TYPED_TEST(MatrixCopyTest, SubMatrixCPU) {
  using T = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sub_configs) {
      const comm::Index2D src_rank_index(0, 0);
      const comm::Index2D dst_rank_index(0, 0);  // TODO ensure that subs are distributed the same way

      const Distribution dist_in(test.full_in, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                 src_rank_index);
      const Distribution dist_out(test.full_out, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                  dst_rank_index);

      const LayoutInfo layout_in = tileLayout(dist_in.localSize(), dist_in.block_size());
      const LayoutInfo layout_out = tileLayout(dist_out.localSize(), dist_out.block_size());

      memory::MemoryView<T, Device::CPU> mem_src(layout_in.minMemSize());
      memory::MemoryView<T, Device::CPU> mem_dst(layout_out.minMemSize());

      Matrix<T, Device::CPU> mat_src =
          createMatrixFromTile<Device::CPU>(dist_in.size(), dist_in.block_size(), comm_grid, mem_src());
      Matrix<T, Device::CPU> mat_dst =
          createMatrixFromTile<Device::CPU>(dist_out.size(), dist_out.block_size(), comm_grid,
                                            mem_dst());

      // Note: currently `subPipeline`-ing does not support sub-matrices
      if (isFullMatrix(dist_in, test.sub_origin_in, test.sub_size)) {
        set(mat_src, inputValues<T>);
        set(mat_dst, outputValues<T>);

        {
          Matrix<const T, Device::CPU> mat_sub_src_const = mat_src.subPipelineConst();
          Matrix<T, Device::CPU> mat_sub_dst = mat_dst.subPipeline();

          copy(mat_sub_src_const, mat_sub_dst);
        }

        CHECK_MATRIX_NEAR(inputValues<T>, mat_dst, 0, TypeUtilities<T>::error);
      }

      // MatrixRef
      set(mat_src, inputValues<T>);
      set(mat_dst, outputValues<T>);

      using matrix::internal::MatrixRef;
      MatrixRef<const T, Device::CPU> mat_sub_src(mat_src, {test.sub_origin_in, test.sub_size});
      MatrixRef<T, Device::CPU> mat_sub_dst(mat_dst, {test.sub_origin_out, test.sub_size});

      copy(mat_sub_src, mat_sub_dst);

      const auto subMatrixValues = subValues(inputValues<T>, test.sub_origin_in);
      CHECK_MATRIX_NEAR(subMatrixValues, mat_sub_dst, 0, TypeUtilities<T>::error);

      // Matrix
      const auto fullMatrixWithSubMatrixValues =
          mixValues(outputValues<T>, inputValues<T>, test.sub_origin_out, test.sub_size);
      CHECK_MATRIX_NEAR(fullMatrixWithSubMatrixValues, mat_dst, 0, TypeUtilities<T>::error);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(MatrixCopyTest, SubMatrixGPU) {
  using MemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::CPU>;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;
  using MatrixConstT = dlaf::Matrix<const TypeParam, Device::CPU>;
  using GPUMemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::GPU>;
  using GPUMatrixT = dlaf::Matrix<TypeParam, Device::GPU>;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      auto input_matrix = [](const GlobalElementIndex& index) {
        SizeType i = index.row();
        SizeType j = index.col();
        return TypeUtilities<TypeParam>::element(i + j / 1024., j - i / 128.);
      };

      MemoryViewT mem_src(layout.minMemSize());
      MatrixT mat_src = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid,
                                                          static_cast<TypeParam*>(mem_src()));
      dlaf::matrix::util::set(mat_src, input_matrix);

      MatrixConstT mat_src_const = std::move(mat_src);

      GPUMemoryViewT mem_gpu1(layout.minMemSize());
      GPUMatrixT mat_gpu1 = createMatrixFromTile<Device::GPU>(size, test.block_size, comm_grid,
                                                              static_cast<TypeParam*>(mem_gpu1()));

      GPUMemoryViewT mem_gpu2(layout.minMemSize());
      GPUMatrixT mat_gpu2 = createMatrixFromTile<Device::GPU>(size, test.block_size, comm_grid,
                                                              static_cast<TypeParam*>(mem_gpu2()));

      MemoryViewT mem_dst(layout.minMemSize());
      MatrixT mat_dst = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid,
                                                          static_cast<TypeParam*>(mem_dst()));
      dlaf::matrix::util::set(mat_dst,
                              [](const auto&) { return TypeUtilities<TypeParam>::element(13, 26); });

      {
        MatrixConstT mat_sub_src_const = mat_src_const.subPipelineConst();
        GPUMatrixT mat_sub_gpu1 = mat_gpu1.subPipeline();
        GPUMatrixT mat_sub_gpu2 = mat_gpu2.subPipeline();
        MatrixT mat_sub_dst = mat_dst.subPipeline();

        copy(mat_sub_src_const, mat_sub_gpu1);
        copy(mat_sub_gpu1, mat_sub_gpu2);
        copy(mat_sub_gpu2, mat_sub_dst);
      }

      CHECK_MATRIX_NEAR(input_matrix, mat_dst, 0, TypeUtilities<TypeParam>::error);
    }
  }
}
#endif
