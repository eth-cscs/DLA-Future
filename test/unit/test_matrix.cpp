//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix.h"
#include "dlaf/matrix/copy.h"

#include <vector>
#include "gtest/gtest.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_futures.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::comm;
using namespace dlaf_test;
using namespace testing;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new CommunicatorGrid6RanksEnvironment);

template <typename Type>
class MatrixLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixLocalTest, MatrixElementTypes);

template <typename Type>
class MatrixTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(MatrixTest, MatrixElementTypes);

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

TYPED_TEST(MatrixLocalTest, StaticAPI) {
  const Device device = Device::CPU;

  using matrix_t = Matrix<TypeParam, device>;

  static_assert(std::is_same<TypeParam, typename matrix_t::ElementType>::value, "wrong ElementType");
  static_assert(std::is_same<Tile<TypeParam, device>, typename matrix_t::TileType>::value,
                "wrong TileType");
  static_assert(std::is_same<Tile<const TypeParam, device>, typename matrix_t::ConstTileType>::value,
                "wrong ConstTileType");
}

TYPED_TEST(MatrixLocalTest, StaticAPIConst) {
  const Device device = Device::CPU;

  using const_matrix_t = Matrix<const TypeParam, device>;

  static_assert(std::is_same<TypeParam, typename const_matrix_t::ElementType>::value,
                "wrong ElementType");
  static_assert(std::is_same<Tile<TypeParam, device>, typename const_matrix_t::TileType>::value,
                "wrong TileType");
  static_assert(std::is_same<Tile<const TypeParam, device>,
                             typename const_matrix_t::ConstTileType>::value,
                "wrong ConstTileType");
}

TYPED_TEST(MatrixLocalTest, Constructor) {
  using Type = TypeParam;
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024., j - i / 128.);
  };

  for (const auto& test : sizes_tests) {
    Matrix<Type, Device::CPU> mat(test.size, test.block_size);

    EXPECT_EQ(Distribution(test.size, test.block_size), mat.distribution());

    set(mat, el);

    CHECK_MATRIX_EQ(el, mat);
  }
}

TYPED_TEST(MatrixTest, Constructor) {
  using Type = TypeParam;
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024., j - i / 128.);
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

      EXPECT_EQ(Distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0}),
                mat.distribution());

      set(mat, el);

      CHECK_MATRIX_EQ(el, mat);
    }
  }
}

TYPED_TEST(MatrixTest, ConstructorFromDistribution) {
  using Type = TypeParam;
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024., j - i / 128.);
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      comm::Index2D src_rank_index(std::max(0, comm_grid.size().rows() - 1),
                                   std::min(1, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);

      // Copy distribution for testing purpose.
      Distribution distribution_copy(distribution);

      Matrix<Type, Device::CPU> mat(std::move(distribution));

      EXPECT_EQ(distribution_copy, mat.distribution());

      set(mat, el);

      CHECK_MATRIX_EQ(el, mat);
    }
  }
}

/// Returns the memory index of the @p index element of the matrix.
///
/// @pre index should be valid, contained in @p distribution.size() and stored in the current rank.
std::size_t memoryIndex(const Distribution& distribution, const LayoutInfo& layout,
                        const GlobalElementIndex& index) {
  using dlaf::util::size_t::sum;
  using dlaf::util::size_t::mul;

  auto global_tile_index = distribution.globalTileIndex(index);
  auto tile_element_index = distribution.tileElementIndex(index);
  auto local_tile_index = distribution.localTileIndex(global_tile_index);
  std::size_t tile_offset = layout.tileOffset(local_tile_index);
  std::size_t element_offset =
      sum(tile_element_index.row(), mul(layout.ldTile(), tile_element_index.col()));
  return tile_offset + element_offset;
}

/// Returns true if the memory index is stored in distribution.rankIndex().
bool ownIndex(const Distribution& distribution, const GlobalElementIndex& index) {
  auto global_tile_index = distribution.globalTileIndex(index);
  return distribution.rankIndex() == distribution.rankGlobalTile(global_tile_index);
}

template <class T, Device device>
void checkDistributionLayout(T* p, const Distribution& distribution, const LayoutInfo& layout,
                             Matrix<T, device>& matrix) {
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

  auto ptr = [p, layout, distribution](const GlobalElementIndex& index) {
    return p + memoryIndex(distribution, layout, index);
  };
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

template <class T, Device device>
void checkDistributionLayout(T* p, const Distribution& distribution, const LayoutInfo& layout,
                             Matrix<const T, device>& matrix) {
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024., j - i / 128.);
  };

  CHECK_MATRIX_DISTRIBUTION(distribution, matrix);

  auto ptr = [p, layout, distribution](const GlobalElementIndex& index) {
    return p + memoryIndex(distribution, layout, index);
  };
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

template <class T, class Mat>
void checkLayoutLocal(T* p, const LayoutInfo& layout, Mat& matrix) {
  Distribution distribution(layout.size(), layout.blockSize());
  checkDistributionLayout(p, distribution, layout, matrix);
}

#define CHECK_DISTRIBUTION_LAYOUT(p, distribution, layout, mat) \
  do {                                                          \
    std::stringstream s;                                        \
    s << "Rank " << distribution.rankIndex();                   \
    SCOPED_TRACE(s.str());                                      \
    checkDistributionLayout(p, distribution, layout, mat);      \
  } while (0)

#define CHECK_LAYOUT_LOCAL(p, layout, mat)    \
  do {                                        \
    SCOPED_TRACE("Local (i.e. Rank (0, 0))"); \
    checkLayoutLocal(p, layout, mat);         \
  } while (0)

TYPED_TEST(MatrixTest, ConstructorFromDistributionLayout) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      comm::Index2D src_rank_index(std::min(1, comm_grid.size().rows() - 1),
                                   std::max(0, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      // Copy distribution for testing purpose.
      Distribution distribution_copy(distribution);

      Matrix<Type, Device::CPU> mat(std::move(distribution), layout);
      Type* ptr = nullptr;
      if (!mat.distribution().localSize().isEmpty()) {
        ptr = mat(LocalTileIndex(0, 0)).get().ptr();
      }

      CHECK_DISTRIBUTION_LAYOUT(ptr, distribution_copy, layout, mat);
    }
  }
}

TYPED_TEST(MatrixTest, LocalGlobalAccessOperatorCall) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      comm::Index2D src_rank_index(std::min(1, comm_grid.size().rows() - 1),
                                   std::max(0, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      Matrix<TypeParam, Device::CPU> mat(std::move(distribution), layout);
      const Distribution& dist = mat.distribution();

      for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
        for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
          GlobalTileIndex global_index(i, j);
          comm::Index2D owner = dist.rankGlobalTile(global_index);

          if (dist.rankIndex() == owner) {
            LocalTileIndex local_index = dist.localTileIndex(global_index);

            const TypeParam* ptr_global = mat(global_index).get().ptr(TileElementIndex{0, 0});
            const TypeParam* ptr_local = mat(local_index).get().ptr(TileElementIndex{0, 0});

            EXPECT_NE(ptr_global, nullptr);
            EXPECT_EQ(ptr_global, ptr_local);
          }
          else {
            EXPECT_THROW(mat(global_index), std::invalid_argument);
          }
        }
      }
    }
  }
}

TYPED_TEST(MatrixTest, LocalGlobalAccessRead) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      comm::Index2D src_rank_index(std::min(1, comm_grid.size().rows() - 1),
                                   std::max(0, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      Matrix<TypeParam, Device::CPU> mat(std::move(distribution), layout);
      const Distribution& dist = mat.distribution();

      for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
        for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
          GlobalTileIndex global_index(i, j);
          comm::Index2D owner = dist.rankGlobalTile(global_index);

          if (dist.rankIndex() == owner) {
            LocalTileIndex local_index = dist.localTileIndex(global_index);

            const TypeParam* ptr_global = mat.read(global_index).get().ptr(TileElementIndex{0, 0});
            const TypeParam* ptr_local = mat.read(local_index).get().ptr(TileElementIndex{0, 0});

            EXPECT_NE(ptr_global, nullptr);
            EXPECT_EQ(ptr_global, ptr_local);
          }
          else {
            EXPECT_THROW(mat.read(global_index), std::invalid_argument);
          }
        }
      }
    }
  }
}

struct ExistingLocalTestSizes {
  LocalElementSize size;
  TileElementSize block_size;
  SizeType ld;
  std::size_t row_offset;
  std::size_t col_offset;
};

const std::vector<ExistingLocalTestSizes> existing_local_tests({
    {{10, 7}, {3, 4}, 10, 3, 40},  // Column major layout
    {{10, 7}, {3, 4}, 11, 3, 44},  // with padding (ld)
    {{10, 7}, {3, 4}, 13, 4, 52},  // with padding (row)
    {{10, 7}, {3, 4}, 10, 3, 41},  // with padding (col)
    {{6, 11}, {4, 3}, 4, 12, 24},  // Tile layout
    {{6, 11}, {4, 3}, 5, 15, 30},  // with padding (ld)
    {{6, 11}, {4, 3}, 4, 13, 26},  // with padding (row)
    {{6, 11}, {4, 3}, 4, 12, 31},  // with padding (col)
    {{6, 11}, {4, 3}, 4, 12, 28},  // compressed col_offset
    {{0, 0}, {1, 1}, 1, 1, 1},
});

TYPED_TEST(MatrixLocalTest, ConstructorExisting) {
  using Type = TypeParam;

  for (const auto& test : existing_local_tests) {
    LayoutInfo layout(test.size, test.block_size, test.ld, test.row_offset, test.col_offset);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

    Matrix<Type, Device::CPU> mat(layout, mem());

    CHECK_LAYOUT_LOCAL(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixLocalTest, ConstructorExistingConst) {
  using Type = TypeParam;

  for (const auto& test : existing_local_tests) {
    LayoutInfo layout(test.size, test.block_size, test.ld, test.row_offset, test.col_offset);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

    const Type* p = mem();
    Matrix<const Type, Device::CPU> mat(layout, p);

    CHECK_LAYOUT_LOCAL(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixTest, ConstructorExisting) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

      // Copy distribution for testing purpose.
      Distribution distribution_copy(distribution);

      Matrix<Type, Device::CPU> mat(std::move(distribution), layout, mem());

      CHECK_DISTRIBUTION_LAYOUT(mem(), distribution_copy, layout, mat);
    }
  }
}

TYPED_TEST(MatrixTest, ConstructorExistingConst) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = colMajorLayout(distribution.localSize(), test.block_size,
                                         std::max<SizeType>(1, distribution.localSize().rows()));
      memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

      // Copy distribution for testing purpose.
      Distribution distribution_copy(distribution);

      const Type* p = mem();
      Matrix<const Type, Device::CPU> mat(std::move(distribution), layout, p);

      CHECK_DISTRIBUTION_LAYOUT(mem(), distribution_copy, layout, mat);
    }
  }
}

TYPED_TEST(MatrixTest, Dependencies) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // fut0 - fut1 - shfut2a - fut3 - shfut4a - fut5
      //             \ shfut2b /      \ shfut4b /

      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

      auto fut0 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

      auto fut1 = getFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut1));

      auto shfut2a = getSharedFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2a));

      auto shfut2b = getSharedFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2b));

      auto fut3 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      auto shfut4a = getSharedFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut4a));

      CHECK_MATRIX_FUTURES(true, fut1, fut0);
      EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      CHECK_MATRIX_FUTURES(true, shfut2b, fut1);
      EXPECT_TRUE(checkFuturesStep(shfut2a.size(), shfut2a));

      CHECK_MATRIX_FUTURES(false, fut3, shfut2b);
      CHECK_MATRIX_FUTURES(true, fut3, shfut2a);

      CHECK_MATRIX_FUTURES(true, shfut4a, fut3);

      auto shfut4b = getSharedFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(shfut4b.size(), shfut4b));

      auto fut5 = getFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      CHECK_MATRIX_FUTURES(false, fut5, shfut4a);
      CHECK_MATRIX_FUTURES(true, fut5, shfut4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesConst) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
      const Type* p = mem();
      Matrix<const Type, Device::CPU> mat(std::move(distribution), layout, p);
      auto shfut1 = getSharedFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(shfut1.size(), shfut1));

      auto shfut2 = getSharedFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(shfut2.size(), shfut2));
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesReferenceMix) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // fut0 - fut1 - shfut2a - fut3 - shfut4a - fut5
      //             \ shfut2b /      \ shfut4b /

      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

      auto fut0 = getFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

      auto fut1 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut1));

      auto shfut2a = getSharedFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2a));

      decltype(shfut2a) shfut2b;
      {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        shfut2b = getSharedFuturesUsingLocalIndex(const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      }

      auto fut3 = getFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      decltype(shfut2a) shfut4a;
      {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        shfut4a = getSharedFuturesUsingLocalIndex(const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut4a));
      }

      CHECK_MATRIX_FUTURES(true, fut1, fut0);
      EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      CHECK_MATRIX_FUTURES(true, shfut2b, fut1);
      EXPECT_TRUE(checkFuturesStep(shfut2a.size(), shfut2a));

      CHECK_MATRIX_FUTURES(false, fut3, shfut2b);
      CHECK_MATRIX_FUTURES(true, fut3, shfut2a);

      CHECK_MATRIX_FUTURES(true, shfut4a, fut3);

      auto shfut4b = getSharedFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(shfut4b.size(), shfut4b));

      auto fut5 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      CHECK_MATRIX_FUTURES(false, fut5, shfut4a);
      CHECK_MATRIX_FUTURES(true, fut5, shfut4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesPointerMix) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // fut0 - fut1 - shfut2a - fut3 - shfut4a - fut5
      //             \ shfut2b /      \ shfut4b /

      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

      auto fut0 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

      auto fut1 = getFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut1));

      auto shfut2a = getSharedFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2a));

      decltype(shfut2a) shfut2b;
      {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        shfut2b = getSharedFuturesUsingGlobalIndex(*const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      }

      auto fut3 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      decltype(shfut2a) shfut4a;
      {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        shfut4a = getSharedFuturesUsingGlobalIndex(*const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut4a));
      }

      CHECK_MATRIX_FUTURES(true, fut1, fut0);
      EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      CHECK_MATRIX_FUTURES(true, shfut2b, fut1);
      EXPECT_TRUE(checkFuturesStep(shfut2a.size(), shfut2a));

      CHECK_MATRIX_FUTURES(false, fut3, shfut2b);
      CHECK_MATRIX_FUTURES(true, fut3, shfut2a);

      CHECK_MATRIX_FUTURES(true, shfut4a, fut3);

      auto shfut4b = getSharedFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(shfut4b.size(), shfut4b));

      auto fut5 = getFuturesUsingGlobalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      CHECK_MATRIX_FUTURES(false, fut5, shfut4a);
      CHECK_MATRIX_FUTURES(true, fut5, shfut4b);
    }
  }
}

TYPED_TEST(MatrixTest, TileSize) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

      for (SizeType i = 0; i < mat.nrTiles().rows(); ++i) {
        SizeType mb = mat.blockSize().rows();
        SizeType ib = std::min(mb, mat.size().rows() - i * mb);
        for (SizeType j = 0; j < mat.nrTiles().cols(); ++j) {
          SizeType nb = mat.blockSize().cols();
          SizeType jb = std::min(nb, mat.size().cols() - j * nb);
          EXPECT_EQ(TileElementSize(ib, jb), mat.tileSize({i, j}));
        }
      }
    }
  }
}

struct TestLocalColMajor {
  LocalElementSize size;
  TileElementSize block_size;
  SizeType ld;
};

const std::vector<TestLocalColMajor> col_major_sizes_tests({
    {{10, 7}, {3, 4}, 10},  // packed ld
    {{10, 7}, {3, 4}, 11},  // padded ld
    {{6, 11}, {4, 3}, 6},   // packed ld
    {{6, 11}, {4, 3}, 7},   // padded ld
});

template <class T, Device device>
bool haveConstElements(const Matrix<T, device>&) {
  return false;
}

template <class T, Device device>
bool haveConstElements(const Matrix<const T, device>&) {
  return true;
}

TYPED_TEST(MatrixLocalTest, FromColMajor) {
  using Type = TypeParam;

  for (const auto& test : col_major_sizes_tests) {
    LayoutInfo layout = colMajorLayout(test.size, test.block_size, test.ld);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    auto mat = createMatrixFromColMajor<Device::CPU>(test.size, test.block_size, test.ld, mem());
    ASSERT_FALSE(haveConstElements(mat));

    CHECK_LAYOUT_LOCAL(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixLocalTest, FromColMajorConst) {
  using Type = TypeParam;

  for (const auto& test : col_major_sizes_tests) {
    LayoutInfo layout = colMajorLayout(test.size, test.block_size, test.ld);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    const Type* p = mem();
    auto mat = createMatrixFromColMajor<Device::CPU>(test.size, test.block_size, test.ld, p);
    ASSERT_TRUE(haveConstElements(mat));

    CHECK_LAYOUT_LOCAL(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixTest, FromColMajor) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});

        SizeType ld = distribution.localSize().rows() + 3;
        LayoutInfo layout = colMajorLayout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        auto mat = createMatrixFromColMajor<Device::CPU>(size, test.block_size, ld, comm_grid, mem());
        ASSERT_FALSE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::max(0, comm_grid.size().rows() - 1),
                               std::max(0, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), src_rank);

        SizeType ld = distribution.localSize().rows() + 3;
        LayoutInfo layout = colMajorLayout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        auto mat =
            createMatrixFromColMajor<Device::CPU>(size, test.block_size, ld, comm_grid, src_rank, mem());
        ASSERT_FALSE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
    }
  }
}

TYPED_TEST(MatrixTest, FromColMajorConst) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});

        SizeType ld = distribution.localSize().rows() + 3;
        LayoutInfo layout = colMajorLayout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        const Type* p = mem();
        auto mat = createMatrixFromColMajor<Device::CPU>(size, test.block_size, ld, comm_grid, p);
        ASSERT_TRUE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::min(1, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), src_rank);

        SizeType ld = distribution.localSize().rows() + 3;
        LayoutInfo layout = colMajorLayout(distribution, ld);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        const Type* p = mem();
        auto mat =
            createMatrixFromColMajor<Device::CPU>(size, test.block_size, ld, comm_grid, src_rank, p);
        ASSERT_TRUE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
    }
  }
}

struct TestLocalTile {
  LocalElementSize size;
  TileElementSize block_size;
  SizeType ld;
  SizeType tiles_per_col;
  bool is_basic;
};

const std::vector<TestLocalTile> tile_sizes_tests({
    {{10, 7}, {3, 4}, 3, 4, true},   // basic tile layout
    {{10, 7}, {3, 4}, 3, 7, false},  // padded tiles_per_col
    {{6, 11}, {4, 3}, 4, 2, true},   // basic tile layout
    {{6, 11}, {4, 3}, 5, 2, false},  // padded ld
});

TYPED_TEST(MatrixLocalTest, FromTile) {
  using Type = TypeParam;

  for (const auto& test : tile_sizes_tests) {
    LayoutInfo layout = tileLayout(test.size, test.block_size, test.ld, test.tiles_per_col);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    if (test.is_basic) {
      auto mat = createMatrixFromTile<Device::CPU>(test.size, test.block_size, mem());
      ASSERT_FALSE(haveConstElements(mat));

      CHECK_LAYOUT_LOCAL(mem(), layout, mat);
    }

    auto mat = createMatrixFromTile<Device::CPU>(test.size, test.block_size, test.ld, test.tiles_per_col,
                                                 mem());
    ASSERT_FALSE(haveConstElements(mat));

    CHECK_LAYOUT_LOCAL(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixLocalTest, FromTileConst) {
  using Type = TypeParam;

  for (const auto& test : tile_sizes_tests) {
    LayoutInfo layout = tileLayout(test.size, test.block_size, test.ld, test.tiles_per_col);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    const Type* p = mem();
    if (test.is_basic) {
      auto mat = createMatrixFromTile<Device::CPU>(test.size, test.block_size, p);
      ASSERT_TRUE(haveConstElements(mat));

      CHECK_LAYOUT_LOCAL(mem(), layout, mat);
    }

    auto mat =
        createMatrixFromTile<Device::CPU>(test.size, test.block_size, test.ld, test.tiles_per_col, p);
    ASSERT_TRUE(haveConstElements(mat));

    CHECK_LAYOUT_LOCAL(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixTest, FromTile) {
  using Type = TypeParam;

  using dlaf::util::ceilDiv;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      // Basic tile layout
      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
        LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, mem());
        ASSERT_FALSE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::max(0, comm_grid.size().rows() - 1),
                               std::max(0, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), src_rank);
        LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, src_rank, mem());
        ASSERT_FALSE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }

      // Advanced tile layout
      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});

        SizeType ld_tiles = test.block_size.rows();
        SizeType tiles_per_col =
            ceilDiv(distribution.localSize().rows(), distribution.blockSize().rows()) + 3;
        LayoutInfo layout = tileLayout(distribution, ld_tiles, tiles_per_col);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, ld_tiles, tiles_per_col,
                                                     comm_grid, mem());
        ASSERT_FALSE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::min(1, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), src_rank);

        SizeType ld_tiles = test.block_size.rows();
        SizeType tiles_per_col =
            ceilDiv(distribution.localSize().rows(), distribution.blockSize().rows()) + 1;
        LayoutInfo layout = tileLayout(distribution, ld_tiles, tiles_per_col);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, ld_tiles, tiles_per_col,
                                                     comm_grid, src_rank, mem());
        ASSERT_FALSE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
    }
  }
}

TYPED_TEST(MatrixTest, FromTileConst) {
  using Type = TypeParam;

  using dlaf::util::ceilDiv;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      // Basic tile layout
      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
        LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        const Type* p = mem();
        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, p);
        ASSERT_TRUE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::max(0, comm_grid.size().rows() - 1),
                               std::max(0, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), src_rank);
        LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        const Type* p = mem();
        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid, src_rank, p);
        ASSERT_TRUE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }

      // Advanced tile layout
      {
        // src_rank = {0, 0}
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});

        SizeType ld_tiles = test.block_size.rows();
        SizeType tiles_per_col =
            ceilDiv(distribution.localSize().rows(), distribution.blockSize().rows()) + 3;
        LayoutInfo layout = tileLayout(distribution, ld_tiles, tiles_per_col);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        const Type* p = mem();
        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, ld_tiles, tiles_per_col,
                                                     comm_grid, p);
        ASSERT_TRUE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
      {
        // specify src_rank
        comm::Index2D src_rank(std::min(1, comm_grid.size().rows() - 1),
                               std::min(1, comm_grid.size().cols() - 1));
        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), src_rank);

        SizeType ld_tiles = test.block_size.rows();
        SizeType tiles_per_col =
            ceilDiv(distribution.localSize().rows(), distribution.blockSize().rows()) + 1;
        LayoutInfo layout = tileLayout(distribution, ld_tiles, tiles_per_col);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

        const Type* p = mem();
        auto mat = createMatrixFromTile<Device::CPU>(size, test.block_size, ld_tiles, tiles_per_col,
                                                     comm_grid, src_rank, p);
        ASSERT_TRUE(haveConstElements(mat));

        CHECK_DISTRIBUTION_LAYOUT(mem(), distribution, layout, mat);
      }
    }
  }
}

TYPED_TEST(MatrixTest, CopyFrom) {
  using MemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::CPU>;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;
  using MatrixConstT = dlaf::Matrix<const TypeParam, Device::CPU>;

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

      MemoryViewT mem_dst(layout.minMemSize());
      MatrixT mat_dst = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid,
                                                          static_cast<TypeParam*>(mem_dst()));
      dlaf::matrix::util::set(mat_dst,
                              [](const auto&) { return TypeUtilities<TypeParam>::element(13, 26); });

      copy(mat_src_const, mat_dst);

      CHECK_MATRIX_NEAR(input_matrix, mat_dst, 0, TypeUtilities<TypeParam>::error);
    }
  }
}

// MatrixDestructorFutures
//
// These tests checks that futures management on destruction is performed correctly. The behaviour is
// strictly related to the future/shared_futures mechanism and generally is not affected by the
// element type of the matrix. For this reason, this kind of test will be carried out with just a
// (randomly chosen) element type.
//
// Note 1:
// In each task there is the last_task future that must depend on the launched task. This is needed
// in order to being able to wait for it before the test ends, otherwise it may end after the test is
// already finished (and in case of failure it may not be presented correctly)
//
// Note 2:
// WAIT_GUARD is the time to wait in the launched task for assuring that Matrix d'tor has been called
// after going out-of-scope. This duration must be kept as low as possible in order to not waste time
// during tests, but at the same time it must be enough to let the "main" to arrive to the end of the
// scope.

const auto WAIT_GUARD = std::chrono::milliseconds(10);
const auto device = dlaf::Device::CPU;
using TypeParam = std::complex<float>;  // randomly chosen element type for matrix

template <class T>
auto createMatrix() -> Matrix<T, device> {
  return {{1, 1}, {1, 1}};
}

template <class T>
auto createConstMatrix() {
  LayoutInfo layout({1, 1}, {1, 1}, 1, 1, 1);
  memory::MemoryView<T, device> mem(layout.minMemSize());
  const T* p = mem();

  return Matrix<const T, device>{layout, p};
}

TEST(MatrixDestructorFutures, NonConstAfterRead) {
  hpx::future<void> last_task;

  volatile int guard = 0;
  {
    auto matrix = createMatrix<TypeParam>();

    auto shared_future = matrix.read(LocalTileIndex(0, 0));
    last_task = shared_future.then(hpx::launch::async, [&guard](auto&&) {
      hpx::this_thread::sleep_for(WAIT_GUARD);
      EXPECT_EQ(0, guard);
    });
  }
  guard = 1;

  last_task.get();
}

TEST(MatrixDestructorFutures, NonConstAfterReadWrite) {
  hpx::future<void> last_task;

  volatile int guard = 0;
  {
    auto matrix = createMatrix<TypeParam>();

    auto future = matrix(LocalTileIndex(0, 0));
    last_task = future.then(hpx::launch::async, [&guard](auto&&) {
      hpx::this_thread::sleep_for(WAIT_GUARD);
      EXPECT_EQ(0, guard);
    });
  }
  guard = 1;

  last_task.get();
}

TEST(MatrixDestructorFutures, ConstAfterRead) {
  hpx::future<void> last_task;

  volatile int guard = 0;
  {
    auto matrix = createConstMatrix<TypeParam>();

    auto sf = matrix.read(LocalTileIndex(0, 0));
    last_task = sf.then(hpx::launch::async, [&guard](auto&&) {
      hpx::this_thread::sleep_for(WAIT_GUARD);
      EXPECT_EQ(0, guard);
    });
  }
  guard = 1;

  last_task.get();
}

struct CustomException final : public std::exception {};

TEST(MatrixExceptionPropagation, RWPropagatesInRWAccess) {
  auto matrix = createMatrix<TypeParam>();

  auto f =
      matrix(LocalTileIndex(0, 0)).then(hpx::util::unwrapping([](auto&&) { throw CustomException{}; }));

  EXPECT_THROW(matrix(LocalTileIndex(0, 0)).get(), dlaf::ContinuationException);
  EXPECT_THROW(f.get(), CustomException);
}

TEST(MatrixExceptionPropagation, RWPropagatesInReadAccess) {
  auto matrix = createMatrix<TypeParam>();

  auto f =
      matrix(LocalTileIndex(0, 0)).then(hpx::util::unwrapping([](auto&&) { throw CustomException{}; }));

  EXPECT_THROW(matrix.read(LocalTileIndex(0, 0)).get(), dlaf::ContinuationException);
  EXPECT_THROW(f.get(), CustomException);
}

TEST(MatrixExceptionPropagation, ReadDoesNotPropagateInRWAccess) {
  auto matrix = createMatrix<TypeParam>();

  auto f = matrix.read(LocalTileIndex(0, 0)).then(hpx::util::unwrapping([](auto&&) {
    throw CustomException{};
  }));

  EXPECT_NO_THROW(matrix(LocalTileIndex(0, 0)).get());
  EXPECT_THROW(f.get(), CustomException);
}

TEST(MatrixExceptionPropagation, ReadDoesNotPropagateInReadAccess) {
  auto matrix = createMatrix<TypeParam>();

  auto f = matrix.read(LocalTileIndex(0, 0)).then(hpx::util::unwrapping([](auto&&) {
    throw CustomException{};
  }));

  EXPECT_NO_THROW(matrix.read(LocalTileIndex(0, 0)).get());
  EXPECT_THROW(f.get(), CustomException);
}
