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
class MatrixLocalViewTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixLocalViewTest, MatrixElementTypes);

template <typename Type>
class MatrixViewTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return comm_grids;
  }
};

TYPED_TEST_SUITE(MatrixViewTest, MatrixElementTypes);

struct TestSizes {
  LocalElementSize size;
  TileElementSize block_size;
};

// TODO (Upper and Lower cases NOT yet implemented)
const std::vector<blas::Uplo> blas_uplos({blas::Uplo::General});

const std::vector<TestSizes> sizes_tests({
    {{0, 0}, {11, 13}},
    {{3, 0}, {1, 2}},
    {{0, 1}, {7, 32}},
    {{15, 18}, {5, 9}},
    {{6, 6}, {2, 2}},
    {{3, 4}, {24, 15}},
    {{7, 9}, {3, 5}},
});

GlobalElementSize globalTestSize(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

TYPED_TEST(MatrixLocalViewTest, StaticAPI) {
  const Device device = Device::CPU;

  using matrix_t = Matrix<TypeParam, device>;

  static_assert(std::is_same<TypeParam, typename matrix_t::ElementType>::value, "wrong ElementType");
  static_assert(std::is_same<Tile<TypeParam, device>, typename matrix_t::TileType>::value,
                "wrong TileType");
  static_assert(std::is_same<Tile<const TypeParam, device>, typename matrix_t::ConstTileType>::value,
                "wrong ConstTileType");
}

TYPED_TEST(MatrixLocalViewTest, StaticAPIConst) {
  const Device device = Device::CPU;

  using const_matrix_view_t = MatrixView<const TypeParam, device>;

  static_assert(std::is_same<TypeParam, typename const_matrix_view_t::ElementType>::value,
                "wrong ElementType");
  static_assert(std::is_same<Tile<TypeParam, device>, typename const_matrix_view_t::TileType>::value,
                "wrong TileType");
  static_assert(std::is_same<Tile<const TypeParam, device>,
                             typename const_matrix_view_t::ConstTileType>::value,
                "wrong ConstTileType");
}

// TODO (Upper and Lower cases NOT yet implemented)
template <template <class, Device> class MatrixType, class T, Device device>
void checkConstructor(MatrixType<T, device>& matrix) {
  static_assert(!std::is_const<T>::value, "Cannot use const matrices for RW views");
  const auto& distribution = matrix.distribution();
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(i + j / 1024., j - i / 128.);
  };
  auto el2 = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<T>::element(-3 - i + j / 1024., 1 - j + i / 256.);
  };

  for (const auto& uplo : blas_uplos) {
    set(matrix, el);  // set the elements before creating the view to avoid deadlocks.
    {
      MatrixView<T, Device::CPU> mat_view(uplo, matrix, true);
      CHECK_MATRIX_DISTRIBUTION(distribution, mat_view);
      CHECK_MATRIX_EQ(el, mat_view);
      set(mat_view, el2);
    }
    CHECK_MATRIX_EQ(el2, matrix);
  }

  set(matrix, el);  // set the elements before creating the view to avoid deadlocks.
  {
    auto mat_view = getView(matrix);
    CHECK_MATRIX_DISTRIBUTION(distribution, mat_view);
    CHECK_MATRIX_EQ(el, mat_view);
    set(mat_view, el2);
  }
  CHECK_MATRIX_EQ(el2, matrix);

  for (const auto& uplo : blas_uplos) {
    set(matrix, el);  // set the elements before creating the view to avoid deadlocks.
    {
      auto mat_view = getView(uplo, matrix);
      CHECK_MATRIX_DISTRIBUTION(distribution, mat_view);
      CHECK_MATRIX_EQ(el, mat_view);
      set(mat_view, el2);
    }
    CHECK_MATRIX_EQ(el2, matrix);
  }
}

#define CHECK_CONSTRUCTOR(matrix)                      \
  do {                                                 \
    std::stringstream s;                               \
    s << "Rank " << matrix.distribution().rankIndex(); \
    SCOPED_TRACE(s.str());                             \
    checkConstructor(matrix);                          \
  } while (0)

// TODO (Upper and Lower cases NOT yet implemented)
template <template <class, Device> class MatrixType, class T, Device device, class ElementGetter>
void checkConstructorConst(MatrixType<T, device>& matrix, ElementGetter el) {
  const auto& distribution = matrix.distribution();

  for (const auto& uplo : blas_uplos) {
    MatrixView<const T, Device::CPU> mat_view(uplo, matrix, true);
    CHECK_MATRIX_DISTRIBUTION(distribution, mat_view);
    CHECK_MATRIX_EQ(el, mat_view);
  }

  {
    auto mat_view = getConstView(matrix);
    CHECK_MATRIX_DISTRIBUTION(distribution, mat_view);
    CHECK_MATRIX_EQ(el, mat_view);
  }

  for (const auto& uplo : blas_uplos) {
    auto mat_view = getConstView(uplo, matrix);
    CHECK_MATRIX_DISTRIBUTION(distribution, mat_view);
    CHECK_MATRIX_EQ(el, mat_view);
  }

  CHECK_MATRIX_EQ(el, matrix);
}

#define CHECK_CONSTRUCTOR_CONST(matrix, el)            \
  do {                                                 \
    std::stringstream s;                               \
    s << "Rank " << matrix.distribution().rankIndex(); \
    SCOPED_TRACE(s.str());                             \
    checkConstructorConst(matrix, el);                 \
  } while (0)

TYPED_TEST(MatrixViewTest, ConstructorHelper) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      {
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

        CHECK_CONSTRUCTOR(mat);
      }
      {
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
        MatrixView<Type, Device::CPU> mat_view(blas::Uplo::General, mat, true);

        CHECK_CONSTRUCTOR(mat_view);
      }
    }
  }
}

TYPED_TEST(MatrixViewTest, ConstructorHelperConst) {
  using Type = TypeParam;
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j / 1024., j - i / 128.);
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      {
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
        set(mat, el);  // set the elements before creating the view to avoid deadlocks.

        CHECK_CONSTRUCTOR_CONST(mat, el);
      }
      {
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
        set(mat, el);  // set the elements before creating the view to avoid deadlocks.
        Matrix<const Type, Device::CPU>& mat_const = mat;

        CHECK_CONSTRUCTOR_CONST(mat_const, el);
      }
      {
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
        set(mat, el);  // set the elements before creating the view to avoid deadlocks.
        MatrixView<Type, Device::CPU> mat_view_const(blas::Uplo::General, mat, true);

        CHECK_CONSTRUCTOR_CONST(mat_view_const, el);
      }
      {
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
        set(mat, el);  // set the elements before creating the view to avoid deadlocks.
        MatrixView<const Type, Device::CPU> mat_view_const(blas::Uplo::General, mat, true);

        CHECK_CONSTRUCTOR_CONST(mat_view_const, el);
      }
    }
  }
}

TYPED_TEST(MatrixViewTest, LocalGlobalAccessOperatorCall) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      comm::Index2D src_rank_index(std::min(1, comm_grid.size().rows() - 1),
                                   std::max(0, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      Matrix<TypeParam, Device::CPU> mat(std::move(distribution), layout);

      auto run_test = [](auto&& mat_view) {
        const Distribution& dist = mat_view.distribution();

        for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
          for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
            GlobalTileIndex global_index(i, j);
            comm::Index2D owner = dist.rankGlobalTile(global_index);

            if (dist.rankIndex() == owner) {
              LocalTileIndex local_index = dist.localTileIndex(global_index);

              const TypeParam* ptr_global = mat_view(global_index).get().ptr(TileElementIndex{0, 0});
              const TypeParam* ptr_local = mat_view(local_index).get().ptr(TileElementIndex{0, 0});

              EXPECT_NE(ptr_global, nullptr);
              EXPECT_EQ(ptr_global, ptr_local);
            }
          }
        }
      };

      run_test(getView(mat));
    }
  }
}

TYPED_TEST(MatrixViewTest, LocalGlobalAccessRead) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      comm::Index2D src_rank_index(std::min(1, comm_grid.size().rows() - 1),
                                   std::max(0, comm_grid.size().cols() - 1));
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(),
                                src_rank_index);
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);

      Matrix<TypeParam, Device::CPU> mat(std::move(distribution), layout);

      auto run_test = [](auto&& mat_view) {
        const Distribution& dist = mat_view.distribution();

        for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
          for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
            GlobalTileIndex global_index(i, j);
            comm::Index2D owner = dist.rankGlobalTile(global_index);

            if (dist.rankIndex() == owner) {
              LocalTileIndex local_index = dist.localTileIndex(global_index);

              const TypeParam* ptr_global =
                  mat_view.read(global_index).get().ptr(TileElementIndex{0, 0});
              const TypeParam* ptr_local = mat_view.read(local_index).get().ptr(TileElementIndex{0, 0});

              EXPECT_NE(ptr_global, nullptr);
              EXPECT_EQ(ptr_global, ptr_local);
            }
          }
        }
      };

      run_test(getView(mat));
      run_test(getConstView(mat));
    }
  }
}

TYPED_TEST(MatrixViewTest, Dependencies) {
  using Type = TypeParam;
  // Dependencies graph Test1:
  // fut0 - shfut1a     - fut2(+0) -(*0) fut3
  //      \ shfut1b     /
  //      \ shfut1c(+0) /
  //      \ shfut1d(+0) /

  // Dependencies graph Test2:
  // fut0 - fut1 (+1) - shfut2a      -(*1) fut3
  //                  \ shfut2b (+1) /
  //                  \(#1) shfut2c  /

  // Notes:
  //  (+n) from matrix view mat_view_<n>
  //  (#n) The next task also depends on destruction of the view object mat_view_<n>
  //       or a call to doneWrite with the index of the given tile.
  //  (*n) The next task also depends on destruction of the view object mat_view_<n>
  //       or a call to done with the index of the given tile.

  auto test1 = [](auto& mat, bool check_done) {
    auto fut0 = getFuturesUsingLocalIndex(mat);
    EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

    auto shfut1a = getSharedFuturesUsingLocalIndex(mat);
    EXPECT_TRUE(checkFuturesStep(0, shfut1a));

    auto shfut1b = getSharedFuturesUsingLocalIndex(mat);
    EXPECT_TRUE(checkFuturesStep(0, shfut1b));

    auto mat_view_0 = std::make_unique<MatrixView<Type, Device::CPU>>(blas::Uplo::General, mat, true);
    auto shfut1c = getSharedFuturesUsingGlobalIndex(*mat_view_0);
    EXPECT_TRUE(checkFuturesStep(0, shfut1c));

    auto shfut1d = getSharedFuturesUsingLocalIndex(*mat_view_0);
    EXPECT_TRUE(checkFuturesStep(0, shfut1d));

    auto fut2 = getFuturesUsingLocalIndex(*mat_view_0);
    EXPECT_TRUE(checkFuturesStep(0, fut2));

    if (!check_done)
      mat_view_0 = nullptr;

    auto fut3 = getFuturesUsingLocalIndex(mat);
    EXPECT_TRUE(checkFuturesStep(0, fut3));

    CHECK_MATRIX_FUTURES(true, shfut1c, fut0);
    EXPECT_TRUE(checkFuturesStep(shfut1a.size(), shfut1a));
    EXPECT_TRUE(checkFuturesStep(shfut1b.size(), shfut1b));
    EXPECT_TRUE(checkFuturesStep(shfut1d.size(), shfut1d));

    CHECK_MATRIX_FUTURES(false, fut2, shfut1a);
    CHECK_MATRIX_FUTURES(false, fut2, shfut1d);
    CHECK_MATRIX_FUTURES(false, fut2, shfut1c);
    CHECK_MATRIX_FUTURES(true, fut2, shfut1b);

    if (check_done) {
      // The view is still available.
      CHECK_MATRIX_FUTURES(false, fut3, fut2);

      CHECK_MATRIX_FUTURES_DONE_WRITE(false, fut3, *mat_view_0);
      CHECK_MATRIX_FUTURES_DONE(true, fut3, *mat_view_0);
    }
    else {
      // The destructor of the view has already been called.
      CHECK_MATRIX_FUTURES(true, fut3, fut2);
    }
  };

  auto test2 = [](auto& mat, bool check_done) {
    auto fut0 = getFuturesUsingLocalIndex(mat);
    EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

    auto mat_view_1 = std::make_unique<MatrixView<Type, Device::CPU>>(blas::Uplo::General, mat, true);

    auto fut1 = getFuturesUsingLocalIndex(*mat_view_1);
    EXPECT_TRUE(checkFuturesStep(0, fut1));

    auto shfut2a = getSharedFuturesUsingGlobalIndex(*mat_view_1);
    EXPECT_TRUE(checkFuturesStep(0, shfut2a));

    auto shfut2b = getSharedFuturesUsingLocalIndex(*mat_view_1);
    EXPECT_TRUE(checkFuturesStep(0, shfut2b));

    if (!check_done)
      mat_view_1 = nullptr;

    auto shfut2c = getSharedFuturesUsingLocalIndex(mat);
    EXPECT_TRUE(checkFuturesStep(0, shfut2c));

    auto fut3 = getFuturesUsingLocalIndex(mat);
    EXPECT_TRUE(checkFuturesStep(0, fut3));

    CHECK_MATRIX_FUTURES(true, fut1, fut0);

    if (check_done) {
      CHECK_MATRIX_FUTURES(true, shfut2b, fut1);
      EXPECT_TRUE(checkFuturesStep(shfut2a.size(), shfut2a));

      CHECK_MATRIX_FUTURES_DONE_WRITE(true, shfut2c, *mat_view_1);
    }
    else {
      CHECK_MATRIX_FUTURES(true, shfut2c, fut1);  // The destructor of the view has already been called.
      EXPECT_TRUE(checkFuturesStep(shfut2a.size(), shfut2a));
      EXPECT_TRUE(checkFuturesStep(shfut2b.size(), shfut2b));
    }

    CHECK_MATRIX_FUTURES(false, fut3, shfut2a);
    CHECK_MATRIX_FUTURES(false, fut3, shfut2c);

    if (check_done) {
      // The view is still available.
      CHECK_MATRIX_FUTURES(false, fut3, shfut2b);

      CHECK_MATRIX_FUTURES_DONE(true, fut3, *mat_view_1);
    }
    else {
      // The destructor of the view has already been called.
      CHECK_MATRIX_FUTURES(true, fut3, shfut2b);
    }
  };

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      for (auto check_done : {true, false}) {
        GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

        test1(mat, check_done);
      }

      for (auto check_done : {true, false}) {
        GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
        MatrixView<Type, Device::CPU> mat_view(blas::Uplo::General, mat, true);

        test1(mat_view, check_done);
      }

      for (auto check_done : {true, false}) {
        GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);

        test2(mat, check_done);
      }

      for (auto check_done : {true, false}) {
        GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
        Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
        MatrixView<Type, Device::CPU> mat_view(blas::Uplo::General, mat, true);

        test2(mat_view, check_done);
      }
    }
  }
}

TYPED_TEST(MatrixViewTest, DependenciesConst) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      // Dependencies graph:
      // fut0 - shfut1a     -(*0) fut2 - shfut3a (+1) -(*1,2) fut4
      //      \ shfut1b(+0) /          \ shfut3b (+2) /
      //      \ shfut1c(+0) /
      //      \ shfut1d     /
      // Notes:
      //  (+n) from matrix view mat_view_<n>
      //  (*n) The next task also depends on destruction of the view object mat_view_<n>
      //       or a call to done with the index of the given tile.
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
      memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
      Matrix<Type, Device::CPU> mat(std::move(distribution), layout);

      auto fut0 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

      auto shfut1a = getSharedFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut1a));

      auto mat_view_0 =
          std::make_unique<MatrixView<const Type, Device::CPU>>(blas::Uplo::General, mat, true);
      auto shfut1b = getSharedFuturesUsingGlobalIndex(*mat_view_0);
      EXPECT_TRUE(checkFuturesStep(0, shfut1b));

      auto shfut1c = getSharedFuturesUsingLocalIndex(*mat_view_0);
      EXPECT_TRUE(checkFuturesStep(0, shfut1c));

      mat_view_0 = nullptr;

      auto shfut1d = getSharedFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, shfut1d));

      auto fut2 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut2));

      auto mat_view_1 =
          std::make_unique<MatrixView<const Type, Device::CPU>>(blas::Uplo::General, mat, true);
      auto shfut3a = getSharedFuturesUsingGlobalIndex(*mat_view_1);
      EXPECT_TRUE(checkFuturesStep(0, shfut3a));

      auto mat_view_2 =
          std::make_unique<MatrixView<const Type, Device::CPU>>(blas::Uplo::General, *mat_view_1, true);
      auto shfut3b = getSharedFuturesUsingLocalIndex(*mat_view_2);
      EXPECT_TRUE(checkFuturesStep(0, shfut3b));

      auto fut4 = getFuturesUsingLocalIndex(mat);
      EXPECT_TRUE(checkFuturesStep(0, fut4));

      CHECK_MATRIX_FUTURES(true, shfut1b, fut0);
      EXPECT_TRUE(checkFuturesStep(shfut1a.size(), shfut1a));
      EXPECT_TRUE(checkFuturesStep(shfut1c.size(), shfut1c));
      EXPECT_TRUE(checkFuturesStep(shfut1d.size(), shfut1d));
      CHECK_MATRIX_FUTURES(false, fut2, shfut1a);
      CHECK_MATRIX_FUTURES(false, fut2, shfut1d);
      CHECK_MATRIX_FUTURES(false, fut2, shfut1c);
      CHECK_MATRIX_FUTURES(true, fut2, shfut1b);  // mat_view_0 already destructed.

      CHECK_MATRIX_FUTURES(true, shfut3b, fut2);
      EXPECT_TRUE(checkFuturesStep(shfut3a.size(), shfut3a));

      CHECK_MATRIX_FUTURES(false, fut4, shfut3a);
      CHECK_MATRIX_FUTURES(false, fut4, shfut3b);
      CHECK_MATRIX_FUTURES_DONE(false, fut4, *mat_view_1);
      CHECK_MATRIX_FUTURES_DONE(true, fut4, *mat_view_2);
    }
  }
}

TYPED_TEST(MatrixViewTest, DependenciesMultiLevelConst) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      for (auto check_done : {true, false}) {
        // Test if multiple level shared_future are ready at the same time.
        // Dependencies graph:
        // fut0 - shfut1a (+3)      -(*1,2,3) fut2
        //       \ shfut1b (+2)     /
        //       \(#2) shfut1c (+1) /
        //       \(#1) shfut1d      /
        // Notes:
        //  (+n) from matrix view mat_view_<n>
        //  (#n) The next task also depends on destruction of the view object mat_view_<n>
        //       or a call to doneWrite with the index of the given tile.
        //  (*n) The next task also depends on destruction of the view object mat_view_<n>
        //       or a call to done with the index of the given tile.
        GlobalElementSize size = globalTestSize(test.size, comm_grid.size());

        Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
        LayoutInfo layout = tileLayout(distribution.localSize(), test.block_size);
        memory::MemoryView<Type, Device::CPU> mem(layout.minMemSize());
        Matrix<Type, Device::CPU> mat(std::move(distribution), layout);

        auto fut0 = getFuturesUsingLocalIndex(mat);
        EXPECT_TRUE(checkFuturesStep(fut0.size(), fut0));

        auto mat_view_1 =
            std::make_unique<MatrixView<Type, Device::CPU>>(blas::Uplo::General, mat, true);
        auto mat_view_2 =
            std::make_unique<MatrixView<Type, Device::CPU>>(blas::Uplo::General, *mat_view_1, true);
        auto const_mat_view_3 =
            std::make_unique<MatrixView<const Type, Device::CPU>>(blas::Uplo::General, *mat_view_2,
                                                                  true);

        auto shfut1a = getSharedFuturesUsingLocalIndex(*const_mat_view_3);
        EXPECT_TRUE(checkFuturesStep(0, shfut1a));

        auto shfut1b = getSharedFuturesUsingLocalIndex(*mat_view_2);
        EXPECT_TRUE(checkFuturesStep(0, shfut1b));

        auto shfut1c = getSharedFuturesUsingLocalIndex(*mat_view_1);
        EXPECT_TRUE(checkFuturesStep(0, shfut1c));

        if (!check_done) {
          mat_view_1 = nullptr;
          mat_view_2 = nullptr;
          const_mat_view_3 = nullptr;
        }

        auto shfut1d = getSharedFuturesUsingLocalIndex(mat);
        EXPECT_TRUE(checkFuturesStep(0, shfut1d));

        auto fut2 = getFuturesUsingLocalIndex(mat);
        EXPECT_TRUE(checkFuturesStep(0, fut2));

        CHECK_MATRIX_FUTURES(true, shfut1a, fut0);
        EXPECT_TRUE(checkFuturesStep(shfut1b.size(), shfut1b));

        if (check_done) {
          CHECK_MATRIX_FUTURES_DONE_WRITE(true, shfut1c, *mat_view_2);
          CHECK_MATRIX_FUTURES_DONE_WRITE(true, shfut1d, *mat_view_1);
        }
        else {
          EXPECT_TRUE(checkFuturesStep(shfut1c.size(), shfut1c));
          EXPECT_TRUE(checkFuturesStep(shfut1d.size(), shfut1d));
        }

        CHECK_MATRIX_FUTURES(false, fut2, shfut1b);
        CHECK_MATRIX_FUTURES(false, fut2, shfut1c);
        CHECK_MATRIX_FUTURES(false, fut2, shfut1d);

        if (check_done) {
          CHECK_MATRIX_FUTURES(false, fut2, shfut1a);  // Views are still available.
          CHECK_MATRIX_FUTURES_DONE(false, fut2, *mat_view_2);
          CHECK_MATRIX_FUTURES_DONE(false, fut2, *mat_view_1);
          CHECK_MATRIX_FUTURES_DONE(true, fut2, *const_mat_view_3);
        }
        else {
          CHECK_MATRIX_FUTURES(true, fut2, shfut1a);  // mat_view_0 already destructed.
        }
      }
    }
  }
}

// TODO (Upper and Lower cases NOT yet implemented)
// TYPED_TEST(MatrixViewTest, DependenciesUplo) {
// }

TYPED_TEST(MatrixViewTest, TileSize) {
  using Type = TypeParam;

  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes_tests) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Matrix<Type, Device::CPU> mat(size, test.block_size, comm_grid);
      auto run_test = [](auto&& mat_view) {
        for (SizeType i = 0; i < mat_view.nrTiles().rows(); ++i) {
          SizeType mb = mat_view.blockSize().rows();
          SizeType ib = std::min(mb, mat_view.size().rows() - i * mb);
          for (SizeType j = 0; j < mat_view.nrTiles().cols(); ++j) {
            SizeType nb = mat_view.blockSize().cols();
            SizeType jb = std::min(nb, mat_view.size().cols() - j * nb);
            EXPECT_EQ(TileElementSize(ib, jb), mat_view.tileSize({i, j}));
          }
        }
      };

      run_test(getView(mat));
      run_test(getConstView(mat));
    }
  }
}
