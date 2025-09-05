//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <atomic>
#include <chrono>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/check_allocation.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/create_matrix.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/util_matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/comm_grids/grids_6_ranks.h>
#include <dlaf_test/matrix/util_matrix.h>
#include <dlaf_test/matrix/util_matrix_senders.h>
#include <dlaf_test/util_types.h>

using namespace std::chrono_literals;

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class MatrixLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixLocalTest, MatrixElementTypes);

template <typename Type>
struct MatrixTest : public TestWithCommGrids {};

TYPED_TEST_SUITE(MatrixTest, MatrixElementTypes);

struct TestSizes {
  SizeType m;
  SizeType n;
  GlobalElementSize block_size;
  TileElementSize tile_size;
};

const std::vector<TestSizes> sizes_tests({
    {0, 0, {11, 13}, {11, 13}},
    {3, 0, {1, 2}, {1, 1}},
    {0, 1, {7, 32}, {7, 8}},
    {15, 18, {5, 9}, {5, 3}},
    {6, 6, {2, 2}, {2, 2}},
    {3, 4, {24, 15}, {8, 15}},
    {16, 24, {3, 5}, {3, 5}},
});

GlobalElementSize global_test_size(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

const matrix::MatrixAllocation tiles_compact(AllocationLayout::Tiles, Ld::Compact);

template <class T, Device D>
void testStaticAPI() {
  using matrix_t = Matrix<T, D>;

  // MatrixLike Traits
  using ncT = std::remove_const_t<T>;
  static_assert(std::is_same_v<ncT, typename matrix_t::ElementType>, "wrong ElementType");
  static_assert(std::is_same_v<Tile<ncT, D>, typename matrix_t::TileType>, "wrong TileType");
  static_assert(std::is_same_v<Tile<const T, D>, typename matrix_t::ConstTileType>,
                "wrong ConstTileType");
}

TYPED_TEST(MatrixLocalTest, StaticAPI) {
  testStaticAPI<TypeParam, Device::CPU>();
  testStaticAPI<TypeParam, Device::GPU>();
}

TYPED_TEST(MatrixLocalTest, StaticAPIConst) {
  testStaticAPI<const TypeParam, Device::CPU>();
  testStaticAPI<const TypeParam, Device::GPU>();
}

TYPED_TEST(MatrixLocalTest, Constructor) {
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (const auto& test : sizes_tests) {
    const Distribution dist = Distribution(LocalElementSize{test.m, test.n}, test.tile_size);
    {
      Matrix<Type, Device::CPU> mat({test.m, test.n}, test.tile_size);
      EXPECT_EQ(dist, mat.distribution());

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
    }
    {
      Matrix<Type, Device::CPU> mat(LocalElementSize{test.m, test.n}, test.tile_size);
      EXPECT_EQ(dist, mat.distribution());

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
    }
    {
      Matrix<Type, Device::CPU> mat(GlobalElementSize{test.m, test.n}, test.tile_size);
      EXPECT_EQ(dist, mat.distribution());

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);

      {
        auto mat_sub = mat.subPipelineConst();
        EXPECT_EQ(mat_sub.distribution(), mat.distribution());

        CHECK_MATRIX_EQ(el, mat_sub);
      }

      c = 1.0;

      {
        auto mat_sub = mat.subPipeline();
        EXPECT_EQ(mat_sub.distribution(), mat.distribution());

        set(mat_sub, el);

        CHECK_MATRIX_EQ(el, mat_sub);
      }

      CHECK_MATRIX_EQ(el, mat);
    }
  }
}

template <class MatrixLike>
SizeType expected_ld_from_00(MatrixLike& mat) {
  if (mat.distribution().local_nr_tiles().isEmpty())
    return 1;
  return tt::sync_wait(mat.read(LocalTileIndex{0, 0})).get().ld();
}

template <class MatrixLike>
void check_const_ld(MatrixLike& mat, SizeType ld) {
  for (auto& ij : iterate_range2d(mat.distribution().local_nr_tiles())) {
    EXPECT_EQ(ld, tt::sync_wait(mat.read(ij)).get().ld());
  }
}

namespace col_major {
template <class MatrixLike>
SizeType expected_ld(MatrixLike& mat, LdSpec ld) {
  const Distribution& dist = mat.distribution();
  const SizeType min_ld =
      dist.local_nr_tiles().isEmpty() ? 1 : std::max<SizeType>(1, dist.local_size().rows());
  SizeType ret = 0;

  if (std::holds_alternative<Ld>(ld)) {
    switch (std::get<Ld>(ld)) {
      case Ld::Compact:
        return min_ld;
      case Ld::Padded: {
        ret = expected_ld_from_00(mat);
        EXPECT_GE(ret, min_ld);
        return ret;
      }
      default:
        DLAF_UNIMPLEMENTED("Invalid Ld");
    }
  }

  ret = std::get<SizeType>(ld);

  EXPECT_GE(ret, min_ld);
  return ret;
}
}

TYPED_TEST(MatrixLocalTest, ConstructorColMajor) {
  using namespace col_major;
  constexpr auto alloc = AllocationLayout::ColMajor;
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (const auto& test : sizes_tests) {
    const Distribution dist = Distribution(LocalElementSize{test.m, test.n}, test.tile_size);
    {
      Matrix<Type, Device::CPU> mat({test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      const SizeType exp_ld = expected_ld_from_00(mat);
      check_const_ld(mat, exp_ld);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_col_major(mat));
    }
    {
      Matrix<Type, Device::CPU> mat(LocalElementSize{test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      const SizeType exp_ld = expected_ld_from_00(mat);
      check_const_ld(mat, exp_ld);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_col_major(mat));
    }
    {
      Matrix<Type, Device::CPU> mat(GlobalElementSize{test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      const SizeType exp_ld = expected_ld_from_00(mat);
      check_const_ld(mat, exp_ld);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_col_major(mat));
    }

    const SizeType min_ld = std::max<SizeType>(1, dist.local_size().rows());
    std::initializer_list<LdSpec> lds{Ld::Compact, Ld::Padded, min_ld, min_ld + 20};
    for (const auto& ld : lds) {
      {
        Matrix<Type, Device::CPU> mat({test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        const SizeType exp_ld = expected_ld(mat, ld);
        check_const_ld(mat, exp_ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_col_major(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(LocalElementSize{test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        const SizeType exp_ld = expected_ld(mat, ld);
        check_const_ld(mat, exp_ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_col_major(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(GlobalElementSize{test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        const SizeType exp_ld = expected_ld(mat, ld);
        check_const_ld(mat, exp_ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_col_major(mat));
      }
    }
  }
}

namespace blocks {

SizeType rows_of_block_containing_tile(const LocalTileIndex& ij, const Distribution& dist) {
  Distribution helper_dist = dlaf::matrix::internal::create_single_tile_per_block_distribution(dist);

  SizeType i_el_local = dist.local_element_from_local_tile_and_tile_element<Coord::Row>(ij.row(), 0);
  SizeType i_bl_local = helper_dist.local_tile_from_local_element<Coord::Row>(i_el_local);

  return helper_dist.local_tile_size_of<Coord::Row>(i_bl_local);
}

template <class MatrixLike>
void check_compact_ld(MatrixLike& mat) {
  for (auto& ij : iterate_range2d(mat.distribution().local_nr_tiles())) {
    auto tile = tt::sync_wait(mat.read(ij));
    SizeType block_rows = rows_of_block_containing_tile(ij, mat.distribution());
    EXPECT_EQ(block_rows, tile.get().ld());
  }
}

template <class MatrixLike>
void check_ld(MatrixLike& mat) {
  for (auto& ij : iterate_range2d(mat.distribution().local_nr_tiles())) {
    auto tile = tt::sync_wait(mat.read(ij));
    SizeType block_rows = rows_of_block_containing_tile(ij, mat.distribution());
    EXPECT_LE(block_rows, tile.get().ld());
  }
}

template <class MatrixLike>
void check_ld(MatrixLike& mat, const LdSpec ld) {
  if (std::holds_alternative<Ld>(ld)) {
    switch (std::get<Ld>(ld)) {
      case Ld::Compact:
        return check_compact_ld(mat);
      case Ld::Padded: {
        return check_ld(mat);
      }
      default:
        DLAF_UNIMPLEMENTED("Invalid Ld");
    }
  }

  check_const_ld(mat, std::get<SizeType>(ld));
}
}

TYPED_TEST(MatrixLocalTest, ConstructorBlocks) {
  using namespace blocks;
  constexpr auto alloc = AllocationLayout::Blocks;
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (const auto& test : sizes_tests) {
    const Distribution dist = Distribution(LocalElementSize{test.m, test.n}, test.tile_size);
    {
      Matrix<Type, Device::CPU> mat({test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      check_ld(mat);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_blocks(mat));
    }
    {
      Matrix<Type, Device::CPU> mat(LocalElementSize{test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      check_ld(mat);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_blocks(mat));
    }
    {
      Matrix<Type, Device::CPU> mat(GlobalElementSize{test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      check_ld(mat);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_blocks(mat));
    }

    const SizeType min_ld = std::max<SizeType>(1, dist.tile_size().rows());
    std::initializer_list<LdSpec> lds{Ld::Compact, Ld::Padded, min_ld, min_ld + 20};
    for (const auto& ld : lds) {
      {
        Matrix<Type, Device::CPU> mat({test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat, ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_blocks(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(LocalElementSize{test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat, ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_blocks(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(GlobalElementSize{test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat, ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_blocks(mat));
      }
    }
  }
}

namespace tiles {
template <class MatrixLike>
void check_compact_ld(MatrixLike& mat) {
  for (auto& ij : iterate_range2d(mat.distribution().local_nr_tiles())) {
    auto tile = tt::sync_wait(mat.read(ij));
    EXPECT_EQ(tile.get().size().rows(), tile.get().ld());
  }
}

template <class MatrixLike>
void check_ld(MatrixLike& mat) {
  for (auto& ij : iterate_range2d(mat.distribution().local_nr_tiles())) {
    auto tile = tt::sync_wait(mat.read(ij));
    EXPECT_LE(tile.get().size().rows(), tile.get().ld());
  }
}

template <class MatrixLike>
void check_ld(MatrixLike& mat, const LdSpec ld) {
  if (std::holds_alternative<Ld>(ld)) {
    switch (std::get<Ld>(ld)) {
      case Ld::Compact:
        return check_compact_ld(mat);
      case Ld::Padded: {
        return check_ld(mat);
      }
      default:
        DLAF_UNIMPLEMENTED("Invalid Ld");
    }
  }

  check_const_ld(mat, std::get<SizeType>(ld));
}
}

TYPED_TEST(MatrixLocalTest, ConstructorTiles) {
  using namespace tiles;
  constexpr auto alloc = AllocationLayout::Tiles;
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (const auto& test : sizes_tests) {
    const Distribution dist = Distribution(LocalElementSize{test.m, test.n}, test.tile_size);
    {
      Matrix<Type, Device::CPU> mat({test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      check_ld(mat);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_tiles(mat));
    }
    {
      Matrix<Type, Device::CPU> mat(LocalElementSize{test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      check_ld(mat);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_tiles(mat));
    }
    {
      Matrix<Type, Device::CPU> mat(GlobalElementSize{test.m, test.n}, test.tile_size, {alloc});
      EXPECT_EQ(dist, mat.distribution());

      check_ld(mat);

      set(mat, el);
      CHECK_MATRIX_EQ(el, mat);
      EXPECT_TRUE(is_allocated_as_tiles(mat));
    }

    const SizeType min_ld = std::max<SizeType>(1, dist.tile_size().rows());
    std::initializer_list<LdSpec> lds{Ld::Compact, Ld::Padded, min_ld, min_ld + 20};
    for (const auto& ld : lds) {
      {
        Matrix<Type, Device::CPU> mat({test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat, ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_tiles(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(LocalElementSize{test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat, ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_tiles(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(GlobalElementSize{test.m, test.n}, test.tile_size, {alloc, ld});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat, ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_tiles(mat));
      }
    }
  }
}

TYPED_TEST(MatrixTest, Constructor) {
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      {
        Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

        EXPECT_EQ(Distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0}),
                  mat.distribution());

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
      }

      {
        const comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                           std::min(1, comm_grid.size().cols() - 1));
        const Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(),
                                comm_grid.rank(), src_rank_index);

        Matrix<Type, Device::CPU> mat(dist);
        EXPECT_EQ(dist, mat.distribution());

        set(mat, el);

        CHECK_MATRIX_EQ(el, mat);
        {
          auto mat_sub = mat.subPipelineConst();
          EXPECT_EQ(mat_sub.distribution(), mat.distribution());

          CHECK_MATRIX_EQ(el, mat_sub);
        }

        c = 1.0;

        {
          auto mat_sub = mat.subPipeline();
          EXPECT_EQ(mat_sub.distribution(), mat.distribution());

          set(mat_sub, el);
          CHECK_MATRIX_EQ(el, mat_sub);
        }

        CHECK_MATRIX_EQ(el, mat);
      }
    }
  }
}

TYPED_TEST(MatrixTest, ConstructorColMajor) {
  using namespace col_major;
  constexpr auto alloc = AllocationLayout::ColMajor;
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      const Distribution dist0(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      const comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                         std::min(1, comm_grid.size().cols() - 1));
      const Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(),
                              src_rank_index);

      {
        Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid, {alloc});

        EXPECT_EQ(dist0, mat.distribution());

        const SizeType exp_ld = expected_ld_from_00(mat);
        check_const_ld(mat, exp_ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_col_major(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(dist, {alloc});
        EXPECT_EQ(dist, mat.distribution());

        const SizeType exp_ld = expected_ld_from_00(mat);
        check_const_ld(mat, exp_ld);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_col_major(mat));
      }

      // Note: min_ld is different as src_rank_index impacts the local matrix sizes.
      {
        const SizeType min_ld = std::max<SizeType>(1, dist0.local_size().rows());
        std::initializer_list<LdSpec> lds{Ld::Compact, Ld::Padded, min_ld, min_ld + 20};
        for (const auto& ld : lds) {
          Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid, {alloc, ld});
          EXPECT_EQ(dist0, mat.distribution());

          const SizeType exp_ld = expected_ld(mat, ld);
          check_const_ld(mat, exp_ld);

          set(mat, el);
          CHECK_MATRIX_EQ(el, mat);
          EXPECT_TRUE(is_allocated_as_col_major(mat));
        }
      }

      {
        const SizeType min_ld = std::max<SizeType>(1, dist.local_size().rows());
        std::initializer_list<LdSpec> lds{Ld::Compact, Ld::Padded, min_ld, min_ld + 20};
        for (const auto& ld : lds) {
          Matrix<Type, Device::CPU> mat(dist, {alloc, ld});
          EXPECT_EQ(dist, mat.distribution());

          const SizeType exp_ld = expected_ld(mat, ld);
          check_const_ld(mat, exp_ld);

          set(mat, el);
          CHECK_MATRIX_EQ(el, mat);
          EXPECT_TRUE(is_allocated_as_col_major(mat));
        }
      }
    }
  }
}

TYPED_TEST(MatrixTest, ConstructorBlocks) {
  using namespace blocks;
  constexpr auto alloc = AllocationLayout::Blocks;
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      const Distribution dist0(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      const comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                         std::min(1, comm_grid.size().cols() - 1));
      const Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(),
                              src_rank_index);
      {
        Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid, {alloc});
        EXPECT_EQ(dist0, mat.distribution());

        check_ld(mat);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_blocks(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(dist, {alloc});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_blocks(mat));
      }

      const SizeType min_ld = test.block_size.rows();
      std::initializer_list<LdSpec> lds{Ld::Compact, Ld::Padded, min_ld, min_ld + 20};
      for (const auto& ld : lds) {
        {
          Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid, {alloc, ld});
          EXPECT_EQ(dist0, mat.distribution());

          check_ld(mat, ld);

          set(mat, el);
          CHECK_MATRIX_EQ(el, mat);
          EXPECT_TRUE(is_allocated_as_blocks(mat));
        }
        {
          Matrix<Type, Device::CPU> mat(dist, {alloc, ld});
          EXPECT_EQ(dist, mat.distribution());

          check_ld(mat, ld);

          set(mat, el);
          CHECK_MATRIX_EQ(el, mat);
          EXPECT_TRUE(is_allocated_as_blocks(mat));
        }
      }
    }
  }
}

TYPED_TEST(MatrixTest, ConstructorTiles) {
  using namespace tiles;
  constexpr auto alloc = AllocationLayout::Tiles;
  using Type = TypeParam;
  BaseType<Type> c = 0.0;
  auto el = [&](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024. + c, j - i / 128.);
  };

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      const Distribution dist0(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      const comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                         std::min(1, comm_grid.size().cols() - 1));
      const Distribution dist(size, test.block_size, test.tile_size, comm_grid.size(), comm_grid.rank(),
                              src_rank_index);
      {
        Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid, {alloc});
        EXPECT_EQ(dist0, mat.distribution());

        check_ld(mat);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_tiles(mat));
      }
      {
        Matrix<Type, Device::CPU> mat(dist, {alloc});
        EXPECT_EQ(dist, mat.distribution());

        check_ld(mat);

        set(mat, el);
        CHECK_MATRIX_EQ(el, mat);
        EXPECT_TRUE(is_allocated_as_tiles(mat));
      }

      const SizeType min_ld = test.tile_size.rows();
      std::initializer_list<LdSpec> lds{Ld::Compact, Ld::Padded, min_ld, min_ld + 20};
      for (const auto& ld : lds) {
        {
          Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid, {alloc, ld});
          EXPECT_EQ(dist0, mat.distribution());

          check_ld(mat, ld);

          set(mat, el);
          CHECK_MATRIX_EQ(el, mat);
          EXPECT_TRUE(is_allocated_as_tiles(mat));
        }
        {
          Matrix<Type, Device::CPU> mat(dist, {alloc, ld});
          EXPECT_EQ(dist, mat.distribution());

          check_ld(mat, ld);

          set(mat, el);
          CHECK_MATRIX_EQ(el, mat);
          EXPECT_TRUE(is_allocated_as_tiles(mat));
        }
      }
    }
  }
}

/// Returns the memory index of the @p index element of the matrix.
///
/// @pre index is contained in @p distribution.size(),
/// @pre index is stored in the current rank.
template <class Layout>
SizeType memoryIndex(const Layout& layout, const GlobalElementIndex& index) {
  const Distribution& distribution = layout.distribution();

  auto global_tile_index = distribution.globalTileIndex(index);
  auto tile_element_index = distribution.tileElementIndex(index);
  auto local_tile_index = distribution.localTileIndex(global_tile_index);
  SizeType tile_offset = layout.tile_offset(local_tile_index);
  SizeType ld = layout.ld_tile(local_tile_index);
  SizeType element_offset = tile_element_index.row() + ld * tile_element_index.col();
  return tile_offset + element_offset;
}

/// Returns true if the memory index is stored in distribution.rankIndex().
bool ownIndex(const Distribution& distribution, const GlobalElementIndex& index) {
  auto global_tile_index = distribution.globalTileIndex(index);
  return distribution.rankIndex() == distribution.rankGlobalTile(global_tile_index);
}

template <class Layout, class T, Device D>
void check_layout(T* p, const Layout& layout, Matrix<T, D>& matrix) {
  const Distribution& distribution = layout.distribution();

  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024., j - i / 128.);
  };
  auto el2 = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(-2 - i + j / 1024., j + i / 64.);
  };

  CHECK_MATRIX_DISTRIBUTION(distribution, matrix);

  auto ptr = [p, layout](const GlobalElementIndex& index) { return p + memoryIndex(layout, index); };
  auto own_element = [distribution](const GlobalElementIndex& index) {
    return ownIndex(distribution, index);
  };
  const auto& size = distribution.size();

  // Set the memory elements.
  // Note: This method is not efficient but for tests is OK.
  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      if (own_element({i, j})) {
        *ptr({i, j}) = el({i, j});
      }
    }
  }

  // Check if the matrix elements correspond to the memory elements.
  CHECK_MATRIX_PTR(ptr, matrix);
  CHECK_MATRIX_EQ(el, matrix);

  // Set the matrix elements.
  set(matrix, el2);

  // Check if the memory elements correspond to the matrix elements.
  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      if (own_element({i, j})) {
        ASSERT_EQ(el2({i, j}), *ptr({i, j})) << "Error at index (" << i << ", " << j << ").";
      }
    }
  }
}

template <class Layout, class T, Device D>
void check_layout(T* p, const Layout& layout, Matrix<const T, D>& matrix) {
  const Distribution& distribution = layout.distribution();

  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024., j - i / 128.);
  };

  CHECK_MATRIX_DISTRIBUTION(distribution, matrix);

  auto ptr = [p, layout](const GlobalElementIndex& index) { return p + memoryIndex(layout, index); };
  auto own_element = [distribution](const GlobalElementIndex& index) {
    return ownIndex(distribution, index);
  };
  const auto& size = distribution.size();

  // Set the memory elements.
  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      if (own_element({i, j}))
        *ptr({i, j}) = el({i, j});
    }
  }

  CHECK_MATRIX_DISTRIBUTION(distribution, matrix);
  // Check if the matrix elements correspond to the memory elements.
  CHECK_MATRIX_PTR(ptr, matrix);
  CHECK_MATRIX_EQ(el, matrix);
}

#define CHECK_LAYOUT(p, layout, mat)                   \
  do {                                                 \
    std::stringstream s;                               \
    s << "Rank " << layout.distribution().rankIndex(); \
    SCOPED_TRACE(s.str());                             \
    check_layout(p, layout, mat);                      \
  } while (0)

#define CHECK_LAYOUT_LOCAL(p, layout, mat)    \
  do {                                        \
    SCOPED_TRACE("Local (i.e. Rank (0, 0))"); \
    check_layout(p, layout, mat);             \
  } while (0)

TYPED_TEST(MatrixTest, LocalGlobalAccessOperatorCall) {
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      comm::Index2D src_rank_index(std::min(1, comm_grid.size().rows() - 1),
                                   std::max(0, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);

      Matrix<TypeParam, Device::CPU> mat(std::move(distribution), tiles_compact);
      const Distribution& dist = mat.distribution();

      for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
        for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
          GlobalTileIndex global_index(i, j);
          comm::Index2D owner = dist.rankGlobalTile(global_index);

          if (dist.rankIndex() == owner) {
            LocalTileIndex local_index = dist.localTileIndex(global_index);

            const TypeParam* ptr_global =
                tt::sync_wait(mat.readwrite(global_index)).ptr(TileElementIndex{0, 0});
            const TypeParam* ptr_local =
                tt::sync_wait(mat.readwrite(local_index)).ptr(TileElementIndex{0, 0});

            EXPECT_NE(ptr_global, nullptr);
            EXPECT_EQ(ptr_global, ptr_local);

            const TypeParam* ptr_sub_global = [&]() {
              auto mat_sub = mat.subPipeline();
              return tt::sync_wait(mat_sub.readwrite(global_index)).ptr(TileElementIndex{0, 0});
            }();
            const TypeParam* ptr_sub_local = [&]() {
              auto mat_sub = mat.subPipeline();
              return tt::sync_wait(mat_sub.readwrite(local_index)).ptr(TileElementIndex{0, 0});
            }();

            EXPECT_EQ(ptr_sub_global, ptr_global);
            EXPECT_EQ(ptr_sub_local, ptr_global);
          }
        }
      }
    }
  }
}

TYPED_TEST(MatrixTest, LocalGlobalAccessRead) {
  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      comm::Index2D src_rank_index(std::min(1, comm_grid.size().rows() - 1),
                                   std::max(0, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);
      Matrix<TypeParam, Device::CPU> mat(std::move(distribution), tiles_compact);

      const Distribution& dist = mat.distribution();

      for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
        for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
          GlobalTileIndex global_index(i, j);
          comm::Index2D owner = dist.rankGlobalTile(global_index);

          if (dist.rankIndex() == owner) {
            LocalTileIndex local_index = dist.localTileIndex(global_index);

            const TypeParam* ptr_global =
                tt::sync_wait(mat.read(global_index)).get().ptr(TileElementIndex{0, 0});
            const TypeParam* ptr_local =
                tt::sync_wait(mat.read(local_index)).get().ptr(TileElementIndex{0, 0});

            EXPECT_NE(ptr_global, nullptr);
            EXPECT_EQ(ptr_global, ptr_local);

            const TypeParam* ptr_sub_global = [&]() {
              auto mat_sub = mat.subPipeline();
              return tt::sync_wait(mat_sub.read(global_index)).get().ptr(TileElementIndex{0, 0});
            }();
            const TypeParam* ptr_sub_local = [&]() {
              auto mat_sub = mat.subPipeline();
              return tt::sync_wait(mat_sub.read(local_index)).get().ptr(TileElementIndex{0, 0});
            }();

            EXPECT_EQ(ptr_sub_global, ptr_global);
            EXPECT_EQ(ptr_sub_local, ptr_global);

            const TypeParam* ptr_sub_const_global = [&]() {
              auto mat_sub = mat.subPipelineConst();
              return tt::sync_wait(mat_sub.read(global_index)).get().ptr(TileElementIndex{0, 0});
            }();
            const TypeParam* ptr_sub_const_local = [&]() {
              auto mat_sub = mat.subPipelineConst();
              return tt::sync_wait(mat_sub.read(local_index)).get().ptr(TileElementIndex{0, 0});
            }();

            EXPECT_EQ(ptr_sub_const_global, ptr_global);
            EXPECT_EQ(ptr_sub_const_local, ptr_global);
          }
        }
      }
    }
  }
}

TYPED_TEST(MatrixTest, ConstructorExisting) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Distribution distribution(size, test.block_size, test.tile_size, comm_grid.size(),
                                comm_grid.rank(), {0, 0});

      SizeType ld = std::max<SizeType>(1, distribution.localSize().rows());
      ColMajorLayout layout(std::move(distribution), ld);
      memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());

      Matrix<Type, Device::CPU> mat(layout, mem());

      CHECK_LAYOUT(mem(), layout, mat);

      {
        auto mat_sub = mat.subPipeline();
        CHECK_LAYOUT(mem(), layout, mat_sub);
      }

      {
        auto mat_sub_const = mat.subPipelineConst();
        CHECK_LAYOUT(mem(), layout, mat_sub_const);
      }
    }
  }
}

TYPED_TEST(MatrixTest, ConstructorExistingConst) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Distribution distribution(size, test.block_size, test.tile_size, comm_grid.size(),
                                comm_grid.rank(), {0, 0});

      SizeType ld = std::max<SizeType>(1, distribution.localSize().rows());
      ColMajorLayout layout(std::move(distribution), ld);
      memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());

      const Type* p = mem();
      Matrix<const Type, Device::CPU> mat(layout, p);

      CHECK_LAYOUT(mem(), layout, mat);

      {
        auto mat_sub_const = mat.subPipelineConst();
        CHECK_LAYOUT(mem(), layout, mat_sub_const);
      }
    }
  }
}

TYPED_TEST(MatrixTest, Dependencies) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /     \ ro4b /

      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto rosenders2a = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders2a));

      auto rosenders2b = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders2b));

      auto senders3 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders3));

      auto rosenders4a = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders4a));

      CHECK_MATRIX_SENDERS(true, senders1, senders0);
      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(true, rosenders2b, senders1);
      EXPECT_TRUE(checkSendersStep(rosenders2a.size(), rosenders2a));

      CHECK_MATRIX_SENDERS(false, senders3, rosenders2b);
      CHECK_MATRIX_SENDERS(true, senders3, rosenders2a);

      CHECK_MATRIX_SENDERS(true, rosenders4a, senders3);

      auto rosenders4b = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(rosenders4b.size(), rosenders4b));

      auto senders5 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders5));

      CHECK_MATRIX_SENDERS(false, senders5, rosenders4a);
      CHECK_MATRIX_SENDERS(true, senders5, rosenders4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesSubPipeline) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /     \ ro4b /
      //
      //             +--------+
      //            sub pipeline

      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto [rosenders2a, rosenders2b, senders3] = [&]() {
        auto mat_sub = mat.subPipeline();

        auto rosenders2a = getReadSendersUsingLocalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders2a));

        auto rosenders2b = getReadSendersUsingGlobalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders2b));

        auto senders3 = getReadWriteSendersUsingLocalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, senders3));

        return std::tuple(std::move(rosenders2a), std::move(rosenders2b), std::move(senders3));
      }();

      auto rosenders4a = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders4a));

      CHECK_MATRIX_SENDERS(true, senders1, senders0);
      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(true, rosenders2b, senders1);
      EXPECT_TRUE(checkSendersStep(rosenders2a.size(), rosenders2a));

      CHECK_MATRIX_SENDERS(false, senders3, rosenders2b);
      CHECK_MATRIX_SENDERS(true, senders3, rosenders2a);

      CHECK_MATRIX_SENDERS(true, rosenders4a, senders3);

      auto rosenders4b = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(rosenders4b.size(), rosenders4b));

      auto senders5 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders5));

      CHECK_MATRIX_SENDERS(false, senders5, rosenders4a);
      CHECK_MATRIX_SENDERS(true, senders5, rosenders4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesSubSubPipeline) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a ------- rw3 - ro4a ------- rw5
      //           \ ------ ro2b /     \ ----- ro4b /
      //
      //             +---------------------+
      //                  sub pipeline
      //
      //                    +-------+
      //                sub-sub pipeline
      //
      // NOTE: The above is the ideal case. The current implementation does not
      // merge read-only accesses between a pipeline and a sub-pipeline.

      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto [rosenders2a, rosenders2b, senders3, rosenders4a] = [&]() {
        auto mat_sub = mat.subPipeline();

        auto rosenders2a = getReadSendersUsingLocalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders2a));

        auto [rosenders2b, senders3] = [&]() {
          auto mat_sub_sub = mat_sub.subPipeline();

          auto rosenders2b = getReadSendersUsingGlobalIndex(mat_sub_sub);
          EXPECT_TRUE(checkSendersStep(0, rosenders2b));

          auto senders3 = getReadWriteSendersUsingLocalIndex(mat_sub_sub);
          EXPECT_TRUE(checkSendersStep(0, senders3));
          return std::tuple(std::move(rosenders2b), std::move(senders3));
        }();

        auto rosenders4a = getReadSendersUsingGlobalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders4a));

        return std::tuple(std::move(rosenders2a), std::move(rosenders2b), std::move(senders3),
                          std::move(rosenders4a));
      }();

      auto rosenders4b = getReadSendersUsingLocalIndex(mat);
      auto senders5 = getReadWriteSendersUsingGlobalIndex(mat);

      EXPECT_TRUE(checkSendersStep(0, senders1));
      CHECK_MATRIX_SENDERS(true, senders1, senders0);

      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(false, rosenders2b, senders1);

      EXPECT_TRUE(checkSendersStep(rosenders2a.size(), rosenders2a));
      CHECK_MATRIX_SENDERS(true, rosenders2b, rosenders2a);

      EXPECT_TRUE(checkSendersStep(0, senders3));
      CHECK_MATRIX_SENDERS(true, senders3, rosenders2b);

      EXPECT_TRUE(checkSendersStep(0, rosenders4a));
      CHECK_MATRIX_SENDERS(true, rosenders4a, senders3);

      EXPECT_TRUE(checkSendersStep(0, rosenders4b));
      CHECK_MATRIX_SENDERS(true, rosenders4b, rosenders4a);

      EXPECT_TRUE(checkSendersStep(0, senders5));
      CHECK_MATRIX_SENDERS(true, senders5, rosenders4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesSubPipelineConst) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /     \ ro4b /
      //
      //             +--+
      //         sub pipeline

      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto [rosenders2a, rosenders2b] = [&]() {
        auto mat_sub = mat.subPipelineConst();

        auto rosenders2a = getReadSendersUsingLocalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders2a));

        auto rosenders2b = getReadSendersUsingGlobalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders2b));

        return std::tuple(std::move(rosenders2a), std::move(rosenders2b));
      }();

      auto senders3 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders3));

      auto rosenders4a = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders4a));

      CHECK_MATRIX_SENDERS(true, senders1, senders0);
      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(true, rosenders2b, senders1);
      EXPECT_TRUE(checkSendersStep(rosenders2a.size(), rosenders2a));

      CHECK_MATRIX_SENDERS(false, senders3, rosenders2b);
      CHECK_MATRIX_SENDERS(true, senders3, rosenders2a);

      CHECK_MATRIX_SENDERS(true, rosenders4a, senders3);

      auto rosenders4b = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(rosenders4b.size(), rosenders4b));

      auto senders5 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders5));

      CHECK_MATRIX_SENDERS(false, senders5, rosenders4a);
      CHECK_MATRIX_SENDERS(true, senders5, rosenders4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesConst) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      Matrix<Type, Device::CPU> mat(std::move(distribution), tiles_compact);
      Matrix<const Type, Device::CPU>& const_mat = mat;

      auto rosenders1 = getReadSendersUsingGlobalIndex(const_mat);
      EXPECT_TRUE(checkSendersStep(rosenders1.size(), rosenders1));

      auto rosenders2 = getReadSendersUsingLocalIndex(const_mat);
      EXPECT_TRUE(checkSendersStep(rosenders2.size(), rosenders2));
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesConstSubPipelineConst) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      Distribution distribution(size, test.block_size, test.tile_size, comm_grid.size(),
                                comm_grid.rank(), {0, 0});
      Matrix<Type, Device::CPU> mat(std::move(distribution), tiles_compact);
      Matrix<const Type, Device::CPU>& const_mat = mat;

      auto rosenders1 = getReadSendersUsingGlobalIndex(const_mat);
      EXPECT_TRUE(checkSendersStep(rosenders1.size(), rosenders1));

      auto rosenders2 = [&]() {
        auto const_mat_sub = const_mat.subPipelineConst();
        return getReadSendersUsingLocalIndex(const_mat_sub);
      }();
      // NOTE: This is a limitation of the current implementation. Semantically
      // read-only access in sub-pipelines should be fused with read-only access
      // from the parent pipeline.
      EXPECT_TRUE(checkSendersStep(0, rosenders2));
      CHECK_MATRIX_SENDERS(true, rosenders2, rosenders1);

      auto rosenders3 = getReadSendersUsingLocalIndex(const_mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders3));
      CHECK_MATRIX_SENDERS(true, rosenders3, rosenders2);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesReferenceMix) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /    \ ro4b /

      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto rosenders2a = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders2a));

      auto rosenders2b = [&]() {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        auto rosenders2b = getReadSendersUsingLocalIndex(const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders2b));
        return rosenders2b;
      }();

      auto senders3 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders3));

      auto rosenders4a = [&]() {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        auto rosenders4a = getReadSendersUsingLocalIndex(const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders4a));
        return rosenders4a;
      }();

      CHECK_MATRIX_SENDERS(true, senders1, senders0);
      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(true, rosenders2b, senders1);
      EXPECT_TRUE(checkSendersStep(rosenders2a.size(), rosenders2a));

      CHECK_MATRIX_SENDERS(false, senders3, rosenders2b);
      CHECK_MATRIX_SENDERS(true, senders3, rosenders2a);

      CHECK_MATRIX_SENDERS(true, rosenders4a, senders3);

      auto rosenders4b = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(rosenders4b.size(), rosenders4b));

      auto senders5 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders5));

      CHECK_MATRIX_SENDERS(false, senders5, rosenders4a);
      CHECK_MATRIX_SENDERS(true, senders5, rosenders4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesReferenceMixSubPipeline) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a ------- rw3 - ro4a ------- rw5
      //           \ ------ ro2b /    \ ------ ro4b /
      //                    +--+        +--+
      //                     sub pipelines
      //
      // NOTE: The above is the ideal case. The current implementation does not
      // merge read-only accesses between a pipeline and a sub-pipeline.

      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto rosenders2a = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders2a));

      auto rosenders2b = [&]() {
        auto mat_sub = mat.subPipelineConst();
        auto rosenders2b = getReadSendersUsingLocalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders2b));
        return rosenders2b;
      }();

      auto senders3 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders3));

      auto rosenders4a = [&]() {
        auto mat_sub = mat.subPipelineConst();
        auto rosenders4a = getReadSendersUsingLocalIndex(mat_sub);
        EXPECT_TRUE(checkSendersStep(0, rosenders4a));
        return rosenders4a;
      }();

      CHECK_MATRIX_SENDERS(true, senders1, senders0);

      EXPECT_TRUE(checkSendersStep(0, rosenders2a));
      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(true, rosenders2a, senders1);

      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(true, rosenders2b, rosenders2a);

      CHECK_MATRIX_SENDERS(true, senders3, rosenders2b);

      CHECK_MATRIX_SENDERS(true, rosenders4a, senders3);

      auto rosenders4b = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders4b));

      auto senders5 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders5));

      CHECK_MATRIX_SENDERS(false, senders5, rosenders4a);
      CHECK_MATRIX_SENDERS(true, senders5, rosenders4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesPointerMix) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /    \ ro4b /

      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto rosenders2a = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders2a));

      auto rosenders2b = [&]() {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        auto rosenders2b = getReadSendersUsingGlobalIndex(*const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders2b));
        return rosenders2b;
      }();

      auto senders3 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders3));

      auto rosenders4a = [&]() {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        auto rosenders4a = getReadSendersUsingGlobalIndex(*const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders4a));
        return rosenders4a;
      }();

      CHECK_MATRIX_SENDERS(true, senders1, senders0);
      EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      CHECK_MATRIX_SENDERS(true, rosenders2b, senders1);
      EXPECT_TRUE(checkSendersStep(rosenders2a.size(), rosenders2a));

      CHECK_MATRIX_SENDERS(false, senders3, rosenders2b);
      CHECK_MATRIX_SENDERS(true, senders3, rosenders2a);

      CHECK_MATRIX_SENDERS(true, rosenders4a, senders3);

      auto rosenders4b = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(rosenders4b.size(), rosenders4b));

      auto senders5 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders5));

      CHECK_MATRIX_SENDERS(false, senders5, rosenders4a);
      CHECK_MATRIX_SENDERS(true, senders5, rosenders4b);
    }
  }
}

TYPED_TEST(MatrixTest, TileSize) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.tile_size, comm_grid);
      auto mat_sub = mat.subPipeline();
      auto mat_sub_const = mat.subPipelineConst();

      for (SizeType i = 0; i < mat.nrTiles().rows(); ++i) {
        SizeType mb = mat.tile_size().rows();
        SizeType ib = std::min(mb, mat.size().rows() - i * mb);
        for (SizeType j = 0; j < mat.nrTiles().cols(); ++j) {
          SizeType nb = mat.tile_size().cols();
          SizeType jb = std::min(nb, mat.size().cols() - j * nb);
          EXPECT_EQ(TileElementSize(ib, jb), mat.tile_size_of({i, j}));
          EXPECT_EQ(TileElementSize(ib, jb), mat_sub.tile_size_of({i, j}));
          EXPECT_EQ(TileElementSize(ib, jb), mat_sub_const.tile_size_of({i, j}));
        }
      }
    }
  }
}

struct TestLocalColMajor {
  GlobalElementSize size;
  TileElementSize tile_size;
  SizeType ld;
};

const std::vector<TestLocalColMajor> col_major_sizes_tests({
    {{10, 7}, {3, 4}, 10},  // packed ld
    {{10, 7}, {3, 4}, 11},  // padded ld
    {{6, 11}, {4, 3}, 6},   // packed ld
    {{6, 11}, {4, 3}, 7},   // padded ld
});

template <class T, Device D>
bool haveConstElements(const Matrix<T, D>&) {
  return false;
}

template <class T, Device D>
bool haveConstElements(const Matrix<const T, D>&) {
  return true;
}

TYPED_TEST(MatrixLocalTest, FromColMajor) {
  using Type = TypeParam;

  for (const auto& test : col_major_sizes_tests) {
    Distribution distribution(test.size, test.tile_size, {1, 1}, {0, 0}, {0, 0});
    ColMajorLayout layout(distribution, test.ld);
    memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());

    auto mat = create_matrix_from_col_major<Device::CPU>(test.size, test.tile_size, test.ld, mem());
    ASSERT_FALSE(haveConstElements(mat));
    CHECK_LAYOUT_LOCAL(mem(), layout, mat);

    {
      auto mat_sub = mat.subPipeline();
      ASSERT_FALSE(haveConstElements(mat_sub));
      CHECK_LAYOUT_LOCAL(mem(), layout, mat_sub);
    }

    {
      auto mat_sub_const = mat.subPipelineConst();
      ASSERT_TRUE(haveConstElements(mat_sub_const));
      CHECK_LAYOUT_LOCAL(mem(), layout, mat_sub_const);
    }
  }
}

TYPED_TEST(MatrixLocalTest, FromColMajorConst) {
  using Type = TypeParam;

  for (const auto& test : col_major_sizes_tests) {
    Distribution distribution(test.size, test.tile_size, {1, 1}, {0, 0}, {0, 0});
    ColMajorLayout layout(distribution, test.ld);
    memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());
    const Type* p = mem();

    auto mat = create_matrix_from_col_major<Device::CPU>(test.size, test.tile_size, test.ld, p);
    ASSERT_TRUE(haveConstElements(mat));
    CHECK_LAYOUT_LOCAL(mem(), layout, mat);

    {
      auto mat_sub_const = mat.subPipelineConst();
      ASSERT_TRUE(haveConstElements(mat_sub_const));
      CHECK_LAYOUT_LOCAL(mem(), layout, mat_sub_const);
    }
  }
}

TYPED_TEST(MatrixTest, FromColMajor) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});

        SizeType ld = distribution.localSize().rows() + 3;
        ColMajorLayout layout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());

        auto mat = create_matrix_from_col_major<Device::CPU>(size, test.tile_size, ld, comm_grid, mem());
        ASSERT_FALSE(haveConstElements(mat));
        CHECK_LAYOUT(mem(), layout, mat);

        {
          auto mat_sub = mat.subPipeline();
          ASSERT_FALSE(haveConstElements(mat_sub));
          CHECK_LAYOUT(mem(), layout, mat_sub);
        }

        {
          auto mat_sub_const = mat.subPipelineConst();
          ASSERT_TRUE(haveConstElements(mat_sub_const));
          CHECK_LAYOUT(mem(), layout, mat_sub_const);
        }
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::max(0, comm_grid.size().rows() - 1),
                               std::max(0, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), src_rank);

        SizeType ld = distribution.localSize().rows() + 3;
        ColMajorLayout layout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());

        auto mat = create_matrix_from_col_major<Device::CPU>(size, test.tile_size, ld, comm_grid,
                                                             src_rank, mem());
        ASSERT_FALSE(haveConstElements(mat));
        CHECK_LAYOUT(mem(), layout, mat);

        {
          auto mat_sub = mat.subPipeline();
          ASSERT_FALSE(haveConstElements(mat_sub));
          CHECK_LAYOUT(mem(), layout, mat_sub);
        }

        {
          auto mat_sub_const = mat.subPipelineConst();
          ASSERT_TRUE(haveConstElements(mat_sub_const));
          CHECK_LAYOUT(mem(), layout, mat_sub_const);
        }
      }
    }
  }
}

TYPED_TEST(MatrixTest, FromColMajorConst) {
  using Type = TypeParam;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});

        SizeType ld = distribution.localSize().rows() + 3;
        ColMajorLayout layout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());
        const Type* p = mem();

        auto mat = create_matrix_from_col_major<Device::CPU>(size, test.tile_size, ld, comm_grid, p);
        ASSERT_TRUE(haveConstElements(mat));
        CHECK_LAYOUT(mem(), layout, mat);

        {
          auto mat_sub_const = mat.subPipelineConst();
          ASSERT_TRUE(haveConstElements(mat_sub_const));
          CHECK_LAYOUT(mem(), layout, mat_sub_const);
        }
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::min(1, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), src_rank);

        SizeType ld = distribution.localSize().rows() + 3;
        ColMajorLayout layout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.min_mem_size());
        const Type* p = mem();

        auto mat =
            create_matrix_from_col_major<Device::CPU>(size, test.tile_size, ld, comm_grid, src_rank, p);
        ASSERT_TRUE(haveConstElements(mat));
        CHECK_LAYOUT(mem(), layout, mat);

        {
          auto mat_sub_const = mat.subPipelineConst();
          ASSERT_TRUE(haveConstElements(mat_sub_const));
          CHECK_LAYOUT(mem(), layout, mat_sub_const);
        }
      }
    }
  }
}

struct TestReshuffling {
  const GlobalElementSize size;
  const TileElementSize src_tilesize;
  const TileElementSize dst_tilesize;
};
std::vector<TestReshuffling> sizes_reshuffling_tests{
    TestReshuffling{{10, 10}, {3, 3}, {3, 3}},   // same shape
    TestReshuffling{{10, 5}, {5, 10}, {10, 2}},  // x2 | /5
    TestReshuffling{{26, 13}, {10, 3}, {5, 6}},  // /2 | x2
};

template <class T, Device Source, Device Destination>
void testReshuffling(const TestReshuffling& config, CommunicatorGrid& grid) {
  const auto& [size, src_tilesize, dst_tilesize] = config;
  const comm::Index2D origin_rank_src(std::max(0, grid.size().rows() - 1),
                                      std::max(0, grid.size().cols() - 1));
  matrix::Distribution dist_src(size, src_tilesize, grid.size(), grid.rank(), origin_rank_src);
  const comm::Index2D origin_rank_dst(
      std::min(grid.size().rows() - 1, dlaf::util::ceilDiv(grid.size().rows(), 2)),
      std::min(grid.size().cols() - 1, dlaf::util::ceilDiv(grid.size().cols(), 2)));
  matrix::Distribution dist_dst(size, dst_tilesize, grid.size(), grid.rank(), origin_rank_dst);

  auto fixedValues = [](const GlobalElementIndex index) { return T(index.row() * 1000 + index.col()); };

  matrix::Matrix<const T, Device::CPU> src_host = [dist_src, fixedValues]() {
    matrix::Matrix<T, Device::CPU> src_host(dist_src);
    matrix::util::set(src_host, fixedValues);
    return src_host;
  }();
  matrix::Matrix<T, Device::CPU> dst_host(dist_dst);

  {
    matrix::MatrixMirror<const T, Source, Device::CPU> src(src_host);
    matrix::MatrixMirror<T, Destination, Device::CPU> dst(dst_host);
    matrix::copy(src.get(), dst.get(), grid);
  }

  CHECK_MATRIX_EQ(fixedValues, dst_host);

  // Note: ensure that everything finishes before next call to Communicator::clone() that might block a
  // working thread (and if it is just one, it would deadlock)
  pika::wait();
}

TYPED_TEST(MatrixTest, CopyReshuffling) {
  for (auto& grid : this->commGrids()) {
    for (const auto& config : sizes_reshuffling_tests) {
      testReshuffling<TypeParam, Device::CPU, Device::CPU>(config, grid);
    }
  }
}

#ifdef DLAF_WITH_GPU
TYPED_TEST(MatrixTest, GPUCopyReshuffling) {
  for (auto& grid : this->commGrids()) {
    for (const auto& config : sizes_reshuffling_tests) {
      testReshuffling<TypeParam, Device::GPU, Device::GPU>(config, grid);
      testReshuffling<TypeParam, Device::CPU, Device::GPU>(config, grid);
      testReshuffling<TypeParam, Device::GPU, Device::CPU>(config, grid);
    }
  }
}
#endif

struct MatrixGenericTest : public TestWithCommGrids {};

TEST_F(MatrixGenericTest, SelectTilesReadonly) {
  using TypeParam = double;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      MatrixT mat{distribution, tiles_compact};

      // if this rank has no tiles locally, there's nothing interesting to do...
      if (distribution.localNrTiles().isEmpty())
        continue;

      const auto ncols = to_sizet(distribution.localNrTiles().cols());
      const LocalTileSize local_row_size{1, to_SizeType(ncols)};
      auto row0_range = common::iterate_range2d(local_row_size);

      // top left tile is selected in rw (i.e. exclusive access)
      auto sender_tl = mat.readwrite(LocalTileIndex{0, 0});

      // the entire first row is selected in ro
      auto senders_row = selectRead(mat, row0_range);
      EXPECT_EQ(ncols, senders_row.size());

      // eagerly start the tile senders, but don't release them
      std::vector<EagerVoidSender> void_senders_row;
      void_senders_row.reserve(senders_row.size());
      for (auto& s : senders_row) {
        void_senders_row.emplace_back(std::move(s));
      }

      // Since the top left tile has been selected two times, the group selection
      // would have all but the first tile ready...
      EXPECT_TRUE(checkSendersStep(1, void_senders_row, true));

      // ... until the first one will be released.
      tt::sync_wait(std::move(sender_tl));
      EXPECT_TRUE(checkSendersStep(ncols, void_senders_row));
    }
  }
}

TEST_F(MatrixGenericTest, SelectTilesReadonlySubPipeline) {
  using TypeParam = double;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      MatrixT mat(std::move(distribution), tiles_compact);
      auto mat_sub = mat.subPipeline();

      // if this rank has no tiles locally, there's nothing interesting to do...
      if (distribution.localNrTiles().isEmpty())
        continue;

      const auto ncols = to_sizet(distribution.localNrTiles().cols());
      const LocalTileSize local_row_size{1, to_SizeType(ncols)};
      auto row0_range = common::iterate_range2d(local_row_size);

      // top left tile is selected in rw (i.e. exclusive access)
      auto sender_tl = mat_sub.readwrite(LocalTileIndex{0, 0});

      // the entire first row is selected in ro
      auto senders_row = selectRead(mat_sub, row0_range);
      EXPECT_EQ(ncols, senders_row.size());

      // eagerly start the tile senders, but don't release them
      std::vector<EagerVoidSender> void_senders_row;
      void_senders_row.reserve(senders_row.size());
      for (auto& s : senders_row) {
        void_senders_row.emplace_back(std::move(s));
      }

      // Since the top left tile has been selected two times, the group selection
      // would have all but the first tile ready...
      EXPECT_TRUE(checkSendersStep(1, void_senders_row, true));

      // ... until the first one will be released.
      tt::sync_wait(std::move(sender_tl));
      EXPECT_TRUE(checkSendersStep(ncols, void_senders_row));
    }
  }
}

TEST_F(MatrixGenericTest, SelectTilesReadwrite) {
  using TypeParam = double;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      MatrixT mat(std::move(distribution), tiles_compact);

      // if this rank has no tiles locally, there's nothing interesting to do...
      if (distribution.localNrTiles().isEmpty())
        continue;

      const auto ncols = to_sizet(distribution.localNrTiles().cols());
      const LocalTileSize local_row_size{1, to_SizeType(ncols)};
      auto row0_range = common::iterate_range2d(local_row_size);

      // top left tile is selected in rw (i.e. exclusive access)
      auto sender_tl = mat.readwrite(LocalTileIndex{0, 0});

      // the entire first row is selected in rw
      auto senders_row = select(mat, row0_range);
      EXPECT_EQ(ncols, senders_row.size());

      // eagerly start the tile senders, but don't release them
      std::vector<EagerVoidSender> void_senders_row;
      void_senders_row.reserve(senders_row.size());
      for (auto& s : senders_row) {
        void_senders_row.emplace_back(std::move(s));
      }

      // Since the top left tile has been selected two times, the group selection
      // would have all but the first tile ready...
      EXPECT_TRUE(checkSendersStep(1, void_senders_row, true));

      // ... until the first one will be released.
      tt::sync_wait(std::move(sender_tl));
      EXPECT_TRUE(checkSendersStep(ncols, void_senders_row));
    }
  }
}

TEST_F(MatrixGenericTest, SelectTilesReadwriteSubPipeline) {
  using TypeParam = double;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      MatrixT mat(std::move(distribution), tiles_compact);
      auto mat_sub = mat.subPipeline();

      // if this rank has no tiles locally, there's nothing interesting to do...
      if (distribution.localNrTiles().isEmpty())
        continue;

      const auto ncols = to_sizet(distribution.localNrTiles().cols());
      const LocalTileSize local_row_size{1, to_SizeType(ncols)};
      auto row0_range = common::iterate_range2d(local_row_size);

      // top left tile is selected in rw (i.e. exclusive access)
      auto sender_tl = mat_sub.readwrite(LocalTileIndex{0, 0});

      // the entire first row is selected in rw
      auto senders_row = select(mat_sub, row0_range);
      EXPECT_EQ(ncols, senders_row.size());

      // eagerly start the tile senders, but don't release them
      std::vector<EagerVoidSender> void_senders_row;
      void_senders_row.reserve(senders_row.size());
      for (auto& s : senders_row) {
        void_senders_row.emplace_back(std::move(s));
      }

      // Since the top left tile has been selected two times, the group selection
      // would have all but the first tile ready...
      EXPECT_TRUE(checkSendersStep(1, void_senders_row, true));

      // ... until the first one will be released.
      tt::sync_wait(std::move(sender_tl));
      EXPECT_TRUE(checkSendersStep(ncols, void_senders_row));
    }
  }
}

// MatrixDestructor
//
// These tests checks that sender management on destruction is performed correctly. The behaviour is
// strictly related to the internal dependency management mechanism and generally is not affected by
// the element type of the matrix. For this reason, this kind of test will be carried out with just a
// (randomly chosen) element type.
//
// Note 1:
// In each task there is the last_task sender that must depend on the launched task. This is needed
// in order to being able to wait for it before the test ends, otherwise it may end after the test is
// already finished (and in case of failure it may not be presented correctly)
//
// Note 2:
// wait_guard is the time to wait in the launched task for assuring that Matrix d'tor has been called
// after going out-of-scope. This duration must be kept as low as possible in order to not waste time
// during tests, but at the same time it must be enough to let the "main" to arrive to the end of the
// scope.
//
// Note 3:
// The tests about lifetime of a Matrix built with user provided memory are not examples of good
// usage, but they are just meant to test that the Matrix does not wait on destruction for any left
// task on one of its tiles.

constexpr Device device = dlaf::Device::CPU;
using T = std::complex<float>;  // randomly chosen element type for matrix

// wait for guard to become true
auto try_waiting_guard = [](auto& guard) {
  const auto wait_guard = 20ms;

  for (int i = 0; i < 100 && !guard; ++i)
    std::this_thread::sleep_for(wait_guard);
};

// Create a single-element matrix
template <class T>
auto create_matrix() -> Matrix<T, device> {
  return {{1, 1}, {1, 1}};
}

// Create a single-element matrix with user-provided memory
template <class T>
auto create_matrix(T& data) -> Matrix<T, device> {
  return create_matrix_from_col_major<Device::CPU>({1, 1}, {1, 1}, 1, &data);
}

// Create a single-element const matrix with user-provided memory
template <class T>
auto create_const_matrix(const T& data) {
  return create_matrix_from_col_major<Device::CPU>({1, 1}, {1, 1}, 1, &data);
}

// Helper for waiting for guard and ensuring that is_exited_from_scope has been set
struct WaitGuardHelper {
  std::atomic<bool>& is_exited_from_scope;

  template <typename T>
  void operator()(T&&) {
    try_waiting_guard(is_exited_from_scope);
    EXPECT_TRUE(is_exited_from_scope);
  }
};

TEST(MatrixDestructor, NonConstAfterRead) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    auto matrix = create_matrix<T>();

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadWrite) {
  namespace ex = pika::execution::experimental;
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    auto matrix = create_matrix<T>();

    auto tile_sender = matrix.readwrite(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterRead_UserMemory) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = create_matrix<T>(data);

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadWrite_UserMemory) {
  namespace ex = pika::execution::experimental;
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = create_matrix<T>(data);

    auto tile_sender = matrix.readwrite(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, ConstAfterRead_UserMemory) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = create_const_matrix<T>(data);

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadSubPipeline) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    auto matrix = create_matrix<T>();
    auto matrix_sub = matrix.subPipeline();

    auto tile_sender = matrix_sub.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadSubPipelineConst) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    auto matrix = create_matrix<T>();
    auto matrix_sub = matrix.subPipelineConst();

    auto tile_sender = matrix_sub.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadWriteSubPipeline) {
  namespace ex = pika::execution::experimental;
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    auto matrix = create_matrix<T>();
    auto matrix_sub = matrix.subPipeline();

    auto tile_sender = matrix_sub.readwrite(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadSubPipeline_UserMemory) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = create_matrix<T>(data);
    auto matrix_sub = matrix.subPipeline();

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadSubPipelineConst_UserMemory) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = create_matrix<T>(data);
    auto matrix_sub = matrix.subPipelineConst();

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, NonConstAfterReadWriteSubPipeline_UserMemory) {
  namespace ex = pika::execution::experimental;
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = create_matrix<T>(data);
    auto matrix_sub = matrix.subPipeline();

    auto tile_sender = matrix_sub.readwrite(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructor, ConstAfterReadSubPipeline_UserMemory) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = create_const_matrix<T>(data);
    auto matrix_sub = matrix.subPipelineConst();

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST_F(MatrixGenericTest, SyncBarrier) {
  using TypeParam = double;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = global_test_size({test.m, test.n}, comm_grid.size());

      Distribution distribution(size, test.tile_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      MatrixT matrix{distribution, tiles_compact};

      const auto local_size = distribution.localNrTiles();
      const LocalTileIndex tile_tl(0, 0);
      const LocalTileIndex tile_br(std::max(SizeType(0), local_size.rows() - 1),
                                   std::max(SizeType(0), local_size.cols() - 1));

      const bool has_local = !local_size.isEmpty();

      // Note:
      // the guard is used to check that tasks before and after the barrier run sequentially and not
      // in parallel.
      // Indeed, two read calls one after the other would result in a parallel execution of their
      // tasks, while a barrier between them must assure that they will be run sequentially.
      std::atomic<bool> guard(false);

      // start a task (if it has at least a local part...otherwise there is no tile to work on)
      if (has_local)
        dlaf::internal::transformDetach(
            dlaf::internal::Policy<dlaf::Backend::MC>(),
            [&guard](auto&&) {
              std::this_thread::sleep_for(100ms);
              guard = true;
            },
            matrix.read(tile_tl));

      // everyone wait on its local part...
      // this means that it is possible to call it also on empty local matrices, they just don't
      // have anything to wait for
      matrix.waitLocalTiles();

      // after the sync barrier, start a task on a tile (another one/the same) expecting that
      // the previous task has been fully completed (and the dependency mechanism still works)
      if (has_local) {
        tt::sync_wait(dlaf::internal::transform(
            dlaf::internal::Policy<dlaf::Backend::MC>(), [&guard](auto&&) { EXPECT_TRUE(guard); },
            matrix.read(tile_tl)));
        tt::sync_wait(dlaf::internal::transform(
            dlaf::internal::Policy<dlaf::Backend::MC>(), [&guard](auto&&) { EXPECT_TRUE(guard); },
            matrix.read(tile_br)));
      }
    }
  }
}
struct CustomException final : public std::exception {};
inline auto throw_custom = [](auto) { throw CustomException{}; };

TEST(MatrixExceptionPropagation, RWDoesNotPropagateInRWAccess) {
  auto matrix = create_matrix<T>();

  auto s = matrix.readwrite(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.readwrite(LocalTileIndex(0, 0))));
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}

TEST(MatrixExceptionPropagation, RWDoesNotPropagateInReadAccess) {
  auto matrix = create_matrix<T>();

  auto s = matrix.readwrite(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.read(LocalTileIndex(0, 0))).get());
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}

TEST(MatrixExceptionPropagation, ReadDoesNotPropagateInRWAccess) {
  auto matrix = create_matrix<T>();

  auto s = matrix.read(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.readwrite(LocalTileIndex(0, 0))));
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}

TEST(MatrixExceptionPropagation, ReadDoesNotPropagateInReadAccess) {
  auto matrix = create_matrix<T>();

  auto s = matrix.read(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.read(LocalTileIndex(0, 0))).get());
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}
