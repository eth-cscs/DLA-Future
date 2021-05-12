//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/matrix.h"
#include "dlaf/util_math.h"
#include "dlaf_test/matrix/matrix_local.h"

#include <sstream>
#include <vector>

#include <gtest/gtest.h>

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_local.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf::test;
using namespace testing;

template <class T>
T value_preset(const GlobalElementIndex& index) {
  const auto i = index.row();
  const auto j = index.col();
  return TypeUtilities<T>::element(i + j / 1024., j - i / 128.);
}

struct TestSizes {
  GlobalElementSize size;
  TileElementSize block_size;
};

const std::vector<TestSizes> sizes_tests({
    {{15, 18}, {5, 9}},
    {{6, 6}, {2, 2}},
    {{3, 4}, {24, 15}},
    {{16, 24}, {3, 5}},
});

template <typename Type>
class MatrixLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixLocalTest, MatrixElementTypes);

TYPED_TEST(MatrixLocalTest, ConstructorAndShape) {
  for (const auto& test : sizes_tests) {
    const MatrixLocal<const TypeParam> mat(test.size, test.block_size);

    EXPECT_EQ(test.size, mat.size());
    EXPECT_EQ(test.block_size, mat.blockSize());

    const GlobalTileSize nrTiles{
        dlaf::util::ceilDiv(test.size.rows(), test.block_size.rows()),
        dlaf::util::ceilDiv(test.size.cols(), test.block_size.cols()),
    };
    EXPECT_EQ(nrTiles, mat.nrTiles());

    EXPECT_EQ(test.size.rows(), mat.ld());
  }
}

TYPED_TEST(MatrixLocalTest, Set) {
  constexpr auto error = TypeUtilities<TypeParam>::error;

  for (const auto& test : sizes_tests) {
    MatrixLocal<TypeParam> mat(test.size, test.block_size);

    set(mat, value_preset<TypeParam>);

    CHECK_MATRIX_NEAR(value_preset<TypeParam>, mat, error, error);
  }
}

TYPED_TEST(MatrixLocalTest, Copy) {
  constexpr auto error = TypeUtilities<TypeParam>::error;

  for (const auto& config : sizes_tests) {
    MatrixLocal<const TypeParam> source = [&config]() {
      MatrixLocal<TypeParam> source(config.size, config.block_size);
      set(source, value_preset<TypeParam>);
      return source;
    }();

    MatrixLocal<TypeParam> dest(config.size, config.block_size);

    copy(source, dest);

    CHECK_MATRIX_NEAR(source, dest, error, error);
  }
}

template <class T>
struct test_output {
  using element_t = T;
  auto operator()() const {
    MatrixLocal<const element_t> mat = [&]() {
      MatrixLocal<element_t> source({5, 4}, {2, 3});
      set(source, [ld = source.ld()](auto&& i) {
        const auto value = i.row() + i.col() * ld;
        return (value % 2 == 0) ? value : -value;
      });
      return source;
    }();

    const std::string output{
        "mat = np.zeros((5, 4), dtype=np.single)\n"
        "mat[0:2,0:3] = np.array([0,-1,-5,6,10,-11,], dtype=np.single).reshape(3, 2).T\n"
        "mat[2:4,0:3] = np.array([2,-3,-7,8,12,-13,], dtype=np.single).reshape(3, 2).T\n"
        "mat[4:5,0:3] = np.array([4,-9,14,], dtype=np.single).reshape(3, 1).T\n"
        "mat[0:2,3:4] = np.array([-15,16,], dtype=np.single).reshape(1, 2).T\n"
        "mat[2:4,3:4] = np.array([-17,18,], dtype=np.single).reshape(1, 2).T\n"
        "mat[4:5,3:4] = np.array([-19,], dtype=np.single).reshape(1, 1).T\n"};

    return std::make_pair(std::move(mat), output);
  }
};

template <class T>
struct test_output<std::complex<T>> {
  using element_t = std::complex<T>;

  auto operator()() const {
    MatrixLocal<const element_t> mat = [&]() {
      MatrixLocal<element_t> source({5, 4}, {2, 3});
      set(source, [ld = source.ld()](auto&& i) {
        return element_t(i.row(), i.col() % 2 == 0 ? i.col() : -i.col());
      });
      return source;
    }();

    const std::string output{
        "mat = np.zeros((5, 4), dtype=np.csingle)\n"
        "mat[0:2,0:3] = np.array([0+0j,1+0j,0-1j,1-1j,0+2j,1+2j,], dtype=np.csingle).reshape(3, 2).T\n"
        "mat[2:4,0:3] = np.array([2+0j,3+0j,2-1j,3-1j,2+2j,3+2j,], dtype=np.csingle).reshape(3, 2).T\n"
        "mat[4:5,0:3] = np.array([4+0j,4-1j,4+2j,], dtype=np.csingle).reshape(3, 1).T\n"
        "mat[0:2,3:4] = np.array([0-3j,1-3j,], dtype=np.csingle).reshape(1, 2).T\n"
        "mat[2:4,3:4] = np.array([2-3j,3-3j,], dtype=np.csingle).reshape(1, 2).T\n"
        "mat[4:5,3:4] = np.array([4-3j,], dtype=np.csingle).reshape(1, 1).T\n"};

    return std::make_pair(std::move(mat), output);
  }
};

TYPED_TEST(MatrixLocalTest, OutputNumpyForamt) {
  const auto test_config = test_output<TypeParam>{}();

  std::ostringstream stream_matrix_output;
  print(format::numpy{}, "mat", test_config.first, stream_matrix_output);

  EXPECT_EQ(test_config.second, stream_matrix_output.str());
}

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class MatrixLocalWithCommTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(MatrixLocalWithCommTest, MatrixElementTypes);

GlobalElementSize globalTestSize(const GlobalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

TYPED_TEST(MatrixLocalWithCommTest, AllGather) {
  constexpr auto error = TypeUtilities<TypeParam>::error;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& config : sizes_tests) {
      const GlobalElementSize size = globalTestSize(config.size, comm_grid.size());
      comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                   std::min(1, comm_grid.size().cols() - 1));
      Distribution distribution(size, config.block_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);

      Matrix<TypeParam, Device::CPU> source(std::move(distribution));
      set(source, value_preset<TypeParam>);

      auto dest = allGather<const TypeParam>(source, comm_grid);

      const auto& dist_src = source.distribution();
      for (const auto& ij_local : iterate_range2d(dist_src.localNrTiles())) {
        const auto ij_global = dist_src.globalTileIndex(ij_local);

        const auto& tile_src = source.read(ij_local).get();
        const auto& tile_dst = dest.tile_read(ij_global);

        CHECK_TILE_NEAR(tile_src, tile_dst, error, error);
      }
    }
  }
}
