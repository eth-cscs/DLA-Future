//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"

#include <atomic>
#include <chrono>
#include <vector>

#include <gtest/gtest.h>
#include <pika/execution.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_senders.h"
#include "dlaf_test/util_types.h"

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
/// @pre index is contained in @p distribution.size(),
/// @pre index is stored in the current rank.
SizeType memoryIndex(const Distribution& distribution, const LayoutInfo& layout,
                     const GlobalElementIndex& index) {
  auto global_tile_index = distribution.globalTileIndex(index);
  auto tile_element_index = distribution.tileElementIndex(index);
  auto local_tile_index = distribution.localTileIndex(global_tile_index);
  SizeType tile_offset = layout.tileOffset(local_tile_index);
  SizeType element_offset = tile_element_index.row() + layout.ldTile() * tile_element_index.col();
  return tile_offset + element_offset;
}

/// Returns true if the memory index is stored in distribution.rankIndex().
bool ownIndex(const Distribution& distribution, const GlobalElementIndex& index) {
  auto global_tile_index = distribution.globalTileIndex(index);
  return distribution.rankIndex() == distribution.rankGlobalTile(global_tile_index);
}

template <class T, Device D>
void checkDistributionLayout(T* p, const Distribution& distribution, const LayoutInfo& layout,
                             Matrix<T, D>& matrix) {
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

template <class T, Device D>
void checkDistributionLayout(T* p, const Distribution& distribution, const LayoutInfo& layout,
                             Matrix<const T, D>& matrix) {
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
        ptr = tt::sync_wait(mat.readwrite(LocalTileIndex(0, 0))).ptr();
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

            const TypeParam* ptr_global =
                tt::sync_wait(mat.readwrite(global_index)).ptr(TileElementIndex{0, 0});
            const TypeParam* ptr_local =
                tt::sync_wait(mat.readwrite(local_index)).ptr(TileElementIndex{0, 0});

            EXPECT_NE(ptr_global, nullptr);
            EXPECT_EQ(ptr_global, ptr_local);
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

            const TypeParam* ptr_global =
                tt::sync_wait(mat.readwrite(global_index)).ptr(TileElementIndex{0, 0});
            const TypeParam* ptr_local =
                tt::sync_wait(mat.readwrite(local_index)).ptr(TileElementIndex{0, 0});

            EXPECT_NE(ptr_global, nullptr);
            EXPECT_EQ(ptr_global, ptr_local);
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
  SizeType row_offset;
  SizeType col_offset;
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
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /     \ ro4b /

      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

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
      auto rosenders1 = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(rosenders1.size(), rosenders1));

      auto rosenders2 = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(rosenders2.size(), rosenders2));
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesReferenceMix) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /    \ ro4b /

      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto rosenders2a = getReadSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders2a));

      decltype(rosenders2a) rosenders2b;
      {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        rosenders2b = getReadSendersUsingLocalIndex(const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      }

      auto senders3 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders3));

      decltype(rosenders2a) rosenders4a;
      {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        rosenders4a = getReadSendersUsingLocalIndex(const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders4a));
      }

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

TYPED_TEST(MatrixTest, DependenciesPointerMix) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // rw0 - rw1 - ro2a - rw3 - ro4a - rw5
      //           \ ro2b /    \ ro4b /

      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

      auto senders0 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(senders0.size(), senders0));

      auto senders1 = getReadWriteSendersUsingGlobalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders1));

      auto rosenders2a = getReadSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, rosenders2a));

      decltype(rosenders2a) rosenders2b;
      {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        rosenders2b = getReadSendersUsingGlobalIndex(*const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders2b));
      }

      auto senders3 = getReadWriteSendersUsingLocalIndex(mat);
      EXPECT_TRUE(checkSendersStep(0, senders3));

      decltype(rosenders2a) rosenders4a;
      {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        rosenders4a = getReadSendersUsingGlobalIndex(*const_mat);
        EXPECT_TRUE(checkSendersStep(0, rosenders4a));
      }

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

#if DLAF_WITH_GPU
TYPED_TEST(MatrixTest, GPUCopy) {
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

      copy(mat_src_const, mat_gpu1);
      copy(mat_gpu1, mat_gpu2);
      copy(mat_gpu2, mat_dst);

      CHECK_MATRIX_NEAR(input_matrix, mat_dst, 0, TypeUtilities<TypeParam>::error);
    }
  }
}
#endif

struct MatrixGenericTest : public TestWithCommGrids {};

TEST_F(MatrixGenericTest, SelectTilesReadonly) {
  using TypeParam = double;
  using MemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::CPU>;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      MemoryViewT mem(layout.minMemSize());
      MatrixT mat = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid,
                                                      static_cast<TypeParam*>(mem()));

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
      std::vector<VoidSenderWithAtomicBool> void_senders_row;
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
  using MemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::CPU>;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      MemoryViewT mem(layout.minMemSize());
      MatrixT mat = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid,
                                                      static_cast<TypeParam*>(mem()));

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
      std::vector<VoidSenderWithAtomicBool> void_senders_row;
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

// MatrixDestructorSenders
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
auto createMatrix() -> Matrix<T, device> {
  return {{1, 1}, {1, 1}};
}

// Create a single-element matrix with user-provided memory
template <class T>
auto createMatrix(T& data) -> Matrix<T, device> {
  return createMatrixFromColMajor<Device::CPU>({1, 1}, {1, 1}, 1, &data);
}

// Create a single-element const matrix with user-provided memory
template <class T>
auto createConstMatrix(const T& data) {
  return createMatrixFromColMajor<Device::CPU>({1, 1}, {1, 1}, 1, &data);
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

TEST(MatrixDestructorSenders, NonConstAfterRead) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    auto matrix = createMatrix<T>();

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructorSenders, NonConstAfterReadWrite) {
  namespace ex = pika::execution::experimental;
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    auto matrix = createMatrix<T>();

    auto tile_sender = matrix.readwrite(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructorSenders, NonConstAfterRead_UserMemory) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = createMatrix<T>(data);

    auto tile_sender = matrix.read(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructorSenders, NonConstAfterReadWrite_UserMemory) {
  namespace ex = pika::execution::experimental;
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = createMatrix<T>(data);

    auto tile_sender = matrix.readwrite(LocalTileIndex(0, 0));
    last_task = std::move(tile_sender) |
                dlaf::internal::transform(dlaf::internal::Policy<dlaf::Backend::MC>(),
                                          WaitGuardHelper{is_exited_from_scope}) |
                ex::ensure_started();
  }
  is_exited_from_scope = true;

  tt::sync_wait(std::move(last_task));
}

TEST(MatrixDestructorSenders, ConstAfterRead_UserMemory) {
  ex::unique_any_sender<> last_task;

  std::atomic<bool> is_exited_from_scope{false};
  {
    T data;
    auto matrix = createConstMatrix<T>(data);

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
  using MemoryViewT = dlaf::memory::MemoryView<TypeParam, Device::CPU>;
  using MatrixT = dlaf::Matrix<TypeParam, Device::CPU>;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      MemoryViewT mem(layout.minMemSize());
      MatrixT matrix = createMatrixFromTile<Device::CPU>(size, test.block_size, comm_grid,
                                                         static_cast<TypeParam*>(mem()));

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

// TODO: Do these tests still make sense with the sender version? They do
// nothing special.
struct CustomException final : public std::exception {};
inline auto throw_custom = [](auto) { throw CustomException{}; };

TEST(MatrixExceptionPropagation, RWDoesNotPropagateInRWAccess) {
  auto matrix = createMatrix<T>();

  auto s = matrix.readwrite(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.readwrite(LocalTileIndex(0, 0))));
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}

TEST(MatrixExceptionPropagation, RWDoesNotPropagateInReadAccess) {
  auto matrix = createMatrix<T>();

  auto s = matrix.readwrite(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.read(LocalTileIndex(0, 0))).get());
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}

TEST(MatrixExceptionPropagation, ReadDoesNotPropagateInRWAccess) {
  auto matrix = createMatrix<T>();

  auto s = matrix.read(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.readwrite(LocalTileIndex(0, 0))));
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}

TEST(MatrixExceptionPropagation, ReadDoesNotPropagateInReadAccess) {
  auto matrix = createMatrix<T>();

  auto s = matrix.read(LocalTileIndex(0, 0)) | ex::then(throw_custom) | ex::ensure_started();

  EXPECT_NO_THROW(tt::sync_wait(matrix.read(LocalTileIndex(0, 0))).get());
  EXPECT_THROW(tt::sync_wait(std::move(s)), CustomException);
}
