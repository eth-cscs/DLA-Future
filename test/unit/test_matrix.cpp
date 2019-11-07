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

TYPED_TEST(MatrixTest, StaticAPI) {
  const Device device = Device::CPU;

  using matrix_t = Matrix<TypeParam, device>;

  static_assert(std::is_same<TypeParam, typename matrix_t::ElementType>::value, "wrong ElementType");
  static_assert(std::is_same<Tile<TypeParam, device>, typename matrix_t::TileType>::value,
                "wrong TileType");
  static_assert(std::is_same<Tile<const TypeParam, device>, typename matrix_t::ConstTileType>::value,
                "wrong ConstTileType");
}

TYPED_TEST(MatrixTest, StaticAPIConst) {
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

template <class MatrixType>
struct TestMatrix : protected MatrixType {
  // Test function which compares matrix sizes and distribution.
  static bool compareBase(const MatrixBase& matrix_base, const MatrixType& matrix) {
    return matrix_base == matrix;
  }
};

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

      EXPECT_TRUE(
          (TestMatrix<Matrix<Type, Device::CPU>>::compareBase(MatrixBase(size, block_size), mat)));

      set(mat, el);

      CHECK_MATRIX_EQ(el, mat);
    }
  }
}

/// @brief Returns the memory index of the @p index element of the matrix.
/// @pre index should be a valid and contained in @p layout.size().
std::size_t memoryIndex(const matrix::LayoutInfo& layout, const GlobalElementIndex& index) {
  using util::size_t::sum;
  using util::size_t::mul;
  const auto& block_size = layout.blockSize();
  SizeType tile_i = index.row() / block_size.rows();
  SizeType tile_j = index.col() / block_size.cols();
  std::size_t tile_offset = layout.tileOffset({tile_i, tile_j});
  SizeType i = index.row() % block_size.rows();
  SizeType j = index.col() % block_size.cols();
  std::size_t element_offset = sum(i, mul(layout.ldTile(), j));
  return tile_offset + element_offset;
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
  auto ptr = [p, layout](const GlobalElementIndex& index) { return p + memoryIndex(layout, index); };
  const auto& size = layout.size();

  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      *ptr({i, j}) = el({i, j});
    }
  }

  EXPECT_TRUE((TestMatrix<Matrix<T, Device::CPU>>::compareBase(MatrixBase(layout), matrix)));
  CHECK_MATRIX_PTR(ptr, matrix);
  CHECK_MATRIX_EQ(el, matrix);

  set(matrix, el2);

  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      ASSERT_EQ(el2({i, j}), *ptr({i, j})) << "Error at index (" << i << ", " << j << ").";
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
  auto ptr = [p, layout](const GlobalElementIndex& index) { return p + memoryIndex(layout, index); };
  const auto& size = layout.size();

  for (SizeType j = 0; j < size.cols(); ++j) {
    for (SizeType i = 0; i < size.rows(); ++i) {
      *ptr({i, j}) = el({i, j});
    }
  }

  EXPECT_TRUE((TestMatrix<Matrix<const T, Device::CPU>>::compareBase(MatrixBase(layout), matrix)));
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

/// @brief Returns true if only the first @p futures are ready.
/// @pre Future should be a future or shared_future.
/// @pre 0 <= ready <= futures.size()
template <class Future>
bool checkFuturesStep(size_t ready, const std::vector<Future>& futures) {
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
/// @pre Future[1,2] should be a future or shared_future
template <class Future1, class Future2>
void checkFutures(bool get_ready, const std::vector<Future1>& current, std::vector<Future2>& previous) {
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
      // Dependencies graph:
      // fut0 - fut1 - shfut2a - fut3 - shfut4a - fut5
      //             \ shfut2b /      \ shfut4b /

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
      auto mat = createMatrixFromTile<Device::CPU>(size, block_size, p, mem.size());
      auto shfut1 = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(shfut1.size(), shfut1));

      auto shfut2 = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(shfut2.size(), shfut2));
    }
  }
}

TYPED_TEST(MatrixTest, DependenciesReferenceMix) {
  using Type = TypeParam;

  for (const auto& size : sizes) {
    for (const auto& block_size : block_sizes) {
      // Dependencies graph:
      // fut0 - fut1 - shfut2a - fut3 - shfut4a - fut5
      //             \ shfut2b /      \ shfut4b /

      Matrix<Type, Device::CPU> mat(size, block_size);

      auto fut0 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

      auto fut1 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut1));

      auto shfut2a = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2a));

      decltype(shfut2a) shfut2b;
      {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        shfut2b = getSharedFutures(const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      }

      auto fut3 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      decltype(shfut2a) shfut4a;
      {
        Matrix<const Type, Device::CPU>& const_mat = mat;
        shfut4a = getSharedFutures(const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut4a));
      }

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

TYPED_TEST(MatrixTest, DependenciesPointerMix) {
  using Type = TypeParam;

  for (const auto& size : sizes) {
    for (const auto& block_size : block_sizes) {
      // Dependencies graph:
      // fut0 - fut1 - shfut2a - fut3 - shfut4a - fut5
      //             \ shfut2b /      \ shfut4b /

      Matrix<Type, Device::CPU> mat(size, block_size);

      auto fut0 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

      auto fut1 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut1));

      auto shfut2a = getSharedFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut2a));

      decltype(shfut2a) shfut2b;
      {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        shfut2b = getSharedFutures(*const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut2b));
      }

      auto fut3 = getFutures(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut3));

      decltype(shfut2a) shfut4a;
      {
        Matrix<const Type, Device::CPU>* const_mat = &mat;
        shfut4a = getSharedFutures(*const_mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut4a));
      }

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
    auto mat = createMatrixFromColMajor<Device::CPU>(size, block_size, ld, mem(), mem.size());

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
    auto mat = createMatrixFromColMajor<Device::CPU>(size, block_size, ld, p, mem.size());

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
      auto mat = createMatrixFromTile<Device::CPU>(size, block_size, mem(), mem.size());
      CHECK_FROM_EXISTING(mem(), layout, mat);
    }

    auto mat = createMatrixFromTile<Device::CPU>(size, block_size, ld, tiles_per_col, mem(), mem.size());

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
      auto mat = createMatrixFromTile<Device::CPU>(size, block_size, p, mem.size());
      CHECK_FROM_EXISTING(mem(), layout, mat);
    }

    auto mat = createMatrixFromTile<Device::CPU>(size, block_size, ld, tiles_per_col, p, mem.size());

    CHECK_FROM_EXISTING(mem(), layout, mat);
  }
}

// MatrixDestructorFutures
//
// These tests checks that futures management on destruction is performed correctly. The behaviour is
// strictly related to the future/shared_futures mechanism and generally is not affected by the element
// type of the matrix. For this reason, this kind of test will be carried out with just a (randomly
// chosen) element type.
//
// Note 1:
// In each task there is the last_task future that must depend on the launched task. This is needed in
// order to being able to wait for it before the test ends, otherwise it may end after the test is
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
auto createConstMatrix() -> Matrix<T, device> {
  matrix::LayoutInfo layout({1, 1}, {1, 1}, 1, 1, 1);
  memory::MemoryView<T, device> mem(layout.minMemSize());
  const T* p = mem();

  return {layout, p, mem.size()};
}

TEST(MatrixDestructorFutures, NonConstAfterRead) {
  hpx::future<void> last_task;

  volatile int guard = 0;
  {
    auto matrix = createMatrix<TypeParam>();

    auto shared_future = matrix.read({0, 0});
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

    auto future = matrix({0, 0});
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
    auto matrix = createConstMatrix<const TypeParam>();

    auto sf = matrix.read({0, 0});
    last_task = sf.then(hpx::launch::async, [&guard](auto&&) {
      hpx::this_thread::sleep_for(WAIT_GUARD);
      EXPECT_EQ(0, guard);
    });
  }
  guard = 1;

  last_task.get();
}
