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

#include "gtest/gtest.h"
#include "dlaf_test/util_matrix.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf_test;
using namespace matrix_test;
using namespace testing;

template <typename Type>
class MatrixTest : public ::testing::Test {};

TYPED_TEST_CASE(MatrixTest, MatrixElementTypes);

std::vector<GlobalElementSize> sizes({{31, 17}, {29, 41}, {0, 1}, {3, 0}});
std::vector<TileElementSize> block_sizes({{7, 11}, {13, 11}, {3, 3}});

TYPED_TEST(MatrixTest, Constructor) {
  using Type = TypeParam;
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + 0.001 * j, j - 0.01 * i);
  };

  for (const auto& size : sizes) {
    for (const auto& block_size : block_sizes) {
      Matrix<Type, Device::CPU> mat(size, block_size);

      EXPECT_EQ(MatrixBase(size, block_size), mat);

      set(mat, el);

      CHECK_MATRIX_EQ(el, mat);
    }
  }
}

template <class T, Device device>
void checkFromExisting(T* p, const matrix::LayoutInfo& layout, Matrix<T, device>& matrix) {
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + 0.001 * j, j - 0.01 * i);
  };
  auto el2 = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(-i + 0.001 * j, j + 0.01 * i);
  };
  auto ptr = [p, layout](const GlobalElementIndex& index) { return getPtr(p, layout, index); };
  const auto& size = layout.size();

  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      *ptr({i, j}) = el({i, j});
    }
  }

  EXPECT_EQ(MatrixBase(layout), matrix);
  CHECK_MATRIX_PTR(ptr, matrix);
  CHECK_MATRIX_EQ(el, matrix);

  set(matrix, el2);

  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      if (el2({i, j}) != *ptr({i, j})) {
        FAIL() << "Error at index (" << i << ", " << j << "): "
               << "expected " << el2({i, j}) << " == " << *ptr({i, j}) << std::endl;
      }
    }
  }
}

template <class T, Device device>
void checkFromExisting(T* p, const matrix::LayoutInfo& layout, Matrix<const T, device>& matrix) {
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + 0.001 * j, j - 0.01 * i);
  };
  auto ptr = [p, layout](const GlobalElementIndex& index) { return getPtr(p, layout, index); };
  const auto& size = layout.size();

  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      *ptr({i, j}) = el({i, j});
    }
  }

  EXPECT_EQ(MatrixBase(layout), matrix);
  CHECK_MATRIX_PTR(ptr, matrix);
  CHECK_MATRIX_EQ(el, matrix);
}

#define CHECK_FROM_EXISTING(p, layout, mat) \
  do {                                      \
    SCOPED_TRACE("");                       \
    checkFromExisting(p, layout, mat);      \
  } while (0)

std::vector<std::tuple<GlobalElementSize, TileElementSize, SizeType, std::size_t, std::size_t>> values(
    {{{31, 17}, {7, 10}, 31, 7, 341},     // Scalapack like layout
     {{31, 17}, {7, 11}, 33, 7, 363},     // with padding (ld)
     {{31, 17}, {7, 11}, 47, 11, 517},    // with padding (row)
     {{31, 17}, {7, 11}, 31, 7, 348},     // with padding (col)
     {{29, 41}, {13, 11}, 13, 143, 429},  // Tile like layout
     {{29, 41}, {13, 11}, 17, 183, 549},  // with padding (ld)
     {{29, 41}, {13, 11}, 13, 146, 438},  // with padding (row)
     {{29, 41}, {13, 11}, 13, 143, 436},  // with padding (col)
     {{29, 41}, {13, 11}, 13, 143, 419},  // compressed col_offset
     {{0, 0}, {1, 1}, 1, 1, 1}});

TYPED_TEST(MatrixTest, ConstructorExisting) {
  using Type = TypeParam;

  for (const auto& v : values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto row_offset = std::get<3>(v);
    auto col_offset = std::get<4>(v);

    matrix::LayoutInfo layout(size, block_size, ld, row_offset, col_offset);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

    Matrix<Type, Device::CPU> mat(layout, mem(), mem.size());

    CHECK_FROM_EXISTING(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixTest, ConstructorExistingConst) {
  using Type = TypeParam;

  for (const auto& v : values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto row_offset = std::get<3>(v);
    auto col_offset = std::get<4>(v);

    matrix::LayoutInfo layout(size, block_size, ld, row_offset, col_offset);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());

    const Type* p = mem();
    Matrix<const Type, Device::CPU> mat(layout, p, mem.size());

    CHECK_FROM_EXISTING(mem(), layout, mat);
  }
}

/// @brief Returns true if the first @p ready futures are ready.
/// @pre Future* should be future or shared_future
/// @pre 0 <= ready <= futures.size()
template <template <class> class Future, class T, Device device>
bool checkFuturesStep(size_t ready, const std::vector<Future<Tile<T, device>>>& futures) {
  assert(ready >= 0);
  assert(ready <= futures.size());

  for (std::size_t index = 0; index < ready; ++index) {
    if (!futures[index].is_ready())
      return false;
  }
  for (std::size_t index = ready; index < futures.size(); ++index) {
    if (futures[index].is_ready())
      return false;
  }
  return true;
}

/// @brief Checks if current[i] depends correctly on previous[i].
/// If get_ready == true it checks if current[i] is ready after previous[i] is used.
/// If get_ready == false it checks if current[i] is not ready after previous[i] is used.
/// @pre Future* should be future or shared_future
template <template <class> class Future1, template <class> class Future2, class T1, class T2,
          Device device>
void checkFutures(bool get_ready, const std::vector<Future1<Tile<T1, device>>>& current,
                  std::vector<Future2<Tile<T2, device>>>& previous) {
  assert(current.size() == previous.size());

  for (std::size_t index = 0; index < current.size(); ++index) {
    EXPECT_TRUE(checkFuturesStep(get_ready ? index : 0, current));
    previous[index].get();
    previous[index] = {};
  }

  EXPECT_TRUE(checkFuturesStep(get_ready ? current.size() : 0, current));
}

TYPED_TEST(MatrixTest, Dependencies) {
  using Type = TypeParam;

  for (const auto& size : sizes) {
    for (const auto& block_size : block_sizes) {
      Matrix<Type, Device::CPU> mat(size, block_size);

      auto fut0 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

      auto fut1 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut1));

      auto shfut2a = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2a));

      auto shfut2b = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2b));

      auto fut3 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      auto shfut4a = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut4a));

      checkFutures(true, fut1, fut0);
      EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      checkFutures(true, shfut2b, fut1);
      EXPECT_TRUE(checkFuturesStep(shfut2a.size(), shfut2a));

      checkFutures(false, fut3, shfut2b);
      checkFutures(true, fut3, shfut2a);

      checkFutures(true, shfut4a, fut3);

      auto shfut4b = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(shfut4b.size(), shfut4b));

      auto fut5 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      checkFutures(false, fut5, shfut4a);
      checkFutures(true, fut5, shfut4b);
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesConst) {
  using Type = TypeParam;

  for (const auto& size : sizes) {
    for (const auto& block_size : block_sizes) {
      matrix::LayoutInfo layout = tileLayout(size, block_size);
      memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
      const Type* p = mem();
      auto mat = createMatrixFromTile<const Type, Device::CPU>(size, block_size, p, mem.size());
      auto shfut1 = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(shfut1.size(), shfut1));

      auto shfut2 = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(shfut2.size(), shfut2));
    }
  }
}

std::vector<std::tuple<GlobalElementSize, TileElementSize, SizeType>> col_major_values({
    {{31, 17}, {7, 11}, 31},   // packed ld
    {{31, 17}, {7, 11}, 33},   // padded ld
    {{29, 41}, {13, 11}, 29},  // packed ld
    {{29, 41}, {13, 11}, 35},  // padded ld
});

TYPED_TEST(MatrixTest, FromColMajor) {
  using Type = TypeParam;

  for (const auto& v : col_major_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);

    matrix::LayoutInfo layout = colMajorLayout(size, block_size, ld);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    auto mat = createMatrixFromColMajor<Type, Device::CPU>(size, block_size, ld, mem(), mem.size());

    CHECK_FROM_EXISTING(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixTest, FromColMajorConst) {
  using Type = TypeParam;

  for (const auto& v : col_major_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);

    matrix::LayoutInfo layout = colMajorLayout(size, block_size, ld);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    const Type* p = mem();
    auto mat = createMatrixFromColMajor<const Type, Device::CPU>(size, block_size, ld, p, mem.size());

    CHECK_FROM_EXISTING(mem(), layout, mat);
  }
}

std::vector<std::tuple<GlobalElementSize, TileElementSize, SizeType, SizeType, bool>> tile_values({
    {{31, 17}, {7, 11}, 7, 5, true},     // basic tile layout
    {{31, 17}, {7, 11}, 11, 5, false},   // padded ld
    {{31, 17}, {7, 11}, 7, 7, false},    // padded ld
    {{29, 41}, {13, 11}, 13, 3, true},   // basic tile layout
    {{29, 41}, {13, 11}, 17, 3, false},  // padded ld
    {{29, 41}, {13, 11}, 13, 4, false},  // padded tiles_per_col
});

TYPED_TEST(MatrixTest, FromTile) {
  using Type = TypeParam;

  for (const auto& v : tile_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto tiles_per_col = std::get<3>(v);
    auto is_basic = std::get<4>(v);

    matrix::LayoutInfo layout = tileLayout(size, block_size, ld, tiles_per_col);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    if (is_basic) {
      auto mat = createMatrixFromTile<Type, Device::CPU>(size, block_size, mem(), mem.size());
      CHECK_FROM_EXISTING(mem(), layout, mat);
    }

    auto mat =
        createMatrixFromTile<Type, Device::CPU>(size, block_size, ld, tiles_per_col, mem(), mem.size());

    CHECK_FROM_EXISTING(mem(), layout, mat);
  }
}

TYPED_TEST(MatrixTest, FromTileConst) {
  using Type = TypeParam;

  for (const auto& v : tile_values) {
    auto size = std::get<0>(v);
    auto block_size = std::get<1>(v);
    auto ld = std::get<2>(v);
    auto tiles_per_col = std::get<3>(v);
    auto is_basic = std::get<4>(v);

    matrix::LayoutInfo layout = tileLayout(size, block_size, ld, tiles_per_col);
    memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
    const Type* p = mem();
    if (is_basic) {
      auto mat = createMatrixFromTile<const Type, Device::CPU>(size, block_size, p, mem.size());
      CHECK_FROM_EXISTING(mem(), layout, mat);
    }

    auto mat = createMatrixFromTile<const Type, Device::CPU>(size, block_size, ld, tiles_per_col, p,
                                                             mem.size());

    CHECK_FROM_EXISTING(mem(), layout, mat);
  }
}
