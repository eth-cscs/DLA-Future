//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/print_csv.h"
#include "dlaf/matrix/print_numpy.h"

#ifdef DLAF_WITH_CUDA
#include "dlaf/matrix/print_gpu.h"
#endif

#include <sstream>
#include <tuple>

#include <gtest/gtest.h>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/matrix/tile.h"

#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::test;

using matrix::test::createTile;

template <typename Type>
class MatrixOutputTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixOutputTest, dlaf::test::MatrixElementTypes);

#ifdef DLAF_WITH_CUDA
template <typename Type>
class MatrixOutputTestGPU : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixOutputTestGPU, dlaf::test::MatrixElementTypes);
#endif

template <class T>
struct test_tile_output {
  using element_t = T;

  static auto empty() {
    const TileElementSize size{0, 0};
    const SizeType ld = 1;
    auto mat = createTile<const element_t>([](auto&&) { return element_t(); }, size, ld);

    const std::string output{"np.array([], dtype=np.single).reshape(0, 0).T\n"};

    const std::string output_csv{""};

    return std::make_tuple(std::move(mat), output, output_csv);
  }

  static auto nonempty() {
    const TileElementSize size{3, 2};
    const SizeType ld = size.rows();
    auto mat = createTile<const element_t>(
        [ld](auto&& i) {
          const auto value = i.row() + i.col() * ld;
          return (value % 2 == 0) ? value : -value;
        },
        size, ld);

    const std::string output{"np.array([0,-1,2,-3,4,-5,], dtype=np.single).reshape(2, 3).T\n"};

    const std::string output_csv{"0,-3,\n"
                                 "-1,4,\n"
                                 "2,-5,\n"};

    return std::make_tuple(std::move(mat), output, output_csv);
  }
};

template <class T>
struct test_tile_output<std::complex<T>> {
  using element_t = std::complex<T>;

  static auto empty() {
    const TileElementSize size{0, 0};
    auto mat = createTile<const element_t>([](auto&&) { return element_t{}; }, size, 1);

    const std::string output{"np.array([], dtype=np.csingle).reshape(0, 0).T\n"};

    const std::string output_csv{""};

    return std::make_tuple(std::move(mat), output, output_csv);
  }

  static auto nonempty() {
    auto value_preset = [](auto&& i) {
      return element_t(i.row(), i.col() % 2 == 0 ? i.col() : -i.col());
    };

    const TileElementSize size{3, 2};
    auto mat = createTile<const element_t>(value_preset, size, size.rows());

    const std::string output{
        "np.array([0+0j,1+0j,2+0j,0-1j,1-1j,2-1j,], dtype=np.csingle).reshape(2, 3).T\n"};

    const std::string output_csv{"(0,0),(0,-1),\n"
                                 "(1,0),(1,-1),\n"
                                 "(2,0),(2,-1),\n"};

    return std::make_tuple(std::move(mat), output, output_csv);
  }
};

#ifdef DLAF_WITH_CUDA
template <class T>
matrix::Tile<T, Device::GPU> gpuTile(const matrix::Tile<const T, Device::CPU>& tile) {
  const auto size = tile.size();
  const auto ld = std::max<SizeType>(1, size.rows());
  matrix::Tile<T, Device::GPU> tile_gpu(size, memory::MemoryView<T, Device::GPU>(size.linear_size()),
                                        ld);

  matrix::internal::copy_o(tile, tile_gpu);
  return tile_gpu;
}
#endif

TYPED_TEST(MatrixOutputTest, NumpyFormatTile) {
  using test_output = test_tile_output<TypeParam>;
  for (auto get_test_config : {test_output::empty, test_output::nonempty}) {
    const auto config = get_test_config();
    auto& tile = std::get<0>(config);
    std::string output_np = std::get<1>(config);

    std::ostringstream stream_tile_output;
    print(format::numpy{}, tile, stream_tile_output);

    EXPECT_EQ(output_np, stream_tile_output.str());
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(MatrixOutputTestGPU, NumpyFormatTile) {
  using test_output = test_tile_output<TypeParam>;
  for (auto get_test_config : {test_output::empty, test_output::nonempty}) {
    const auto config = get_test_config();
    auto tile = gpuTile(std::get<0>(config));
    std::string output_np = std::get<1>(config);

    std::ostringstream stream_tile_output;
    print(format::numpy{}, tile, stream_tile_output);

    EXPECT_EQ(output_np, stream_tile_output.str());
  }
}
#endif

TYPED_TEST(MatrixOutputTest, CsvFormatTile) {
  using test_output = test_tile_output<TypeParam>;
  for (auto get_test_config : {test_output::empty, test_output::nonempty}) {
    const auto config = get_test_config();
    auto& tile = std::get<0>(config);
    std::string output_csv = std::get<2>(config);

    std::ostringstream stream_tile_output;
    print(format::csv{}, tile, stream_tile_output);

    EXPECT_EQ(output_csv, stream_tile_output.str());
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(MatrixOutputTestGPU, CsvFormatTile) {
  using test_output = test_tile_output<TypeParam>;
  for (auto get_test_config : {test_output::empty, test_output::nonempty}) {
    const auto config = get_test_config();
    auto tile = gpuTile(std::get<0>(config));
    std::string output_np = std::get<2>(config);

    std::ostringstream stream_tile_output;
    print(format::csv{}, tile, stream_tile_output);

    EXPECT_EQ(output_np, stream_tile_output.str());
  }
}
#endif

template <class T>
struct test_matrix_output {
  using element_t = T;

  static auto empty() {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Matrix<element_t, Device::CPU> source({0, 0}, {2, 3});
      return source;
    }();

    const std::string output{"mat = np.zeros((0, 0), dtype=np.single)\n"};

    const std::string output_csv{"mat\n"};

    return std::make_tuple(std::move(mat), output, output_csv);
  }

  static auto nonempty() {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Matrix<element_t, Device::CPU> source({5, 4}, {2, 3});
      matrix::test::set(source, [ld = source.size().rows()](auto&& i) {
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

    const std::string output_csv{"mat\n"
                                 "0,-5,10,-15,\n"
                                 "-1,6,-11,16,\n"
                                 "2,-7,12,-17,\n"
                                 "-3,8,-13,18,\n"
                                 "4,-9,14,-19,\n"};

    return std::make_tuple(std::move(mat), output, output_csv);
  }
};

template <class T>
struct test_matrix_output<std::complex<T>> {
  using element_t = std::complex<T>;

  static auto empty() {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Matrix<element_t, Device::CPU> source({0, 0}, {2, 3});
      return source;
    }();

    const std::string output{"mat = np.zeros((0, 0), dtype=np.csingle)\n"};

    const std::string output_csv{"mat\n"};

    return std::make_tuple(std::move(mat), output, output_csv);
  }

  static auto nonempty() {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Matrix<element_t, Device::CPU> source({5, 4}, {2, 3});
      matrix::test::set(source, [ld = source.size().rows()](auto&& i) {
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

    const std::string output_csv{"mat\n"
                                 "(0,0),(0,-1),(0,2),(0,-3),\n"
                                 "(1,0),(1,-1),(1,2),(1,-3),\n"
                                 "(2,0),(2,-1),(2,2),(2,-3),\n"
                                 "(3,0),(3,-1),(3,2),(3,-3),\n"
                                 "(4,0),(4,-1),(4,2),(4,-3),\n"};

    return std::make_tuple(std::move(mat), output, output_csv);
  }
};

TYPED_TEST(MatrixOutputTest, NumpyFormatMatrix) {
  using test_output = test_matrix_output<TypeParam>;
  for (auto get_test_config : {test_output::empty, test_output::nonempty}) {
    auto config = get_test_config();
    auto& matrix = std::get<0>(config);
    std::string output_np = std::get<1>(config);

    std::ostringstream stream_matrix_output;
    print(format::numpy{}, "mat", matrix, stream_matrix_output);

    EXPECT_EQ(output_np, stream_matrix_output.str());
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(MatrixOutputTestGPU, NumpyFormatMatrix) {
  using test_output = test_matrix_output<TypeParam>;
  for (auto get_test_config : {test_output::empty, test_output::nonempty}) {
    auto config = get_test_config();
    auto& matrix = std::get<0>(config);
    MatrixMirror<const TypeParam, Device::GPU, Device::CPU> matrix_d(matrix);

    std::string output_np = std::get<1>(config);

    std::ostringstream stream_matrix_output;
    print(format::numpy{}, "mat", matrix_d.get(), stream_matrix_output);

    EXPECT_EQ(output_np, stream_matrix_output.str());
  }
}
#endif

TYPED_TEST(MatrixOutputTest, CsvFormatMatrix) {
  using test_output = test_matrix_output<TypeParam>;
  for (auto get_test_config : {test_output::empty, test_output::nonempty}) {
    auto config = get_test_config();
    auto& matrix = std::get<0>(config);
    std::string output_csv = std::get<2>(config);

    std::ostringstream stream_matrix_output;
    print(format::csv{}, "mat", matrix, stream_matrix_output);

    EXPECT_EQ(output_csv, stream_matrix_output.str());
  }
}

template <class T>
struct test_matrix_dist_output {
  using element_t = T;

  comm::CommunicatorGrid comm_grid_;

  test_matrix_dist_output() : comm_grid_({MPI_COMM_WORLD}, {3, 2}, common::Ordering::ColumnMajor) {}

  auto empty() const {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Distribution distribution({0, 0}, {2, 3}, comm_grid_.size(), comm_grid_.rank(), {0, 0});
      Matrix<element_t, Device::CPU> source(std::move(distribution));
      return source;
    }();

    const std::string output{"M = np.zeros((0, 0), dtype=np.single)\n"};

    std::stringstream output_csv;
    output_csv << mat << std::endl;

    return std::make_tuple(std::move(mat), output, output_csv.str());
  }

  auto nonempty() const {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Distribution distribution({5, 4}, {2, 3}, comm_grid_.size(), comm_grid_.rank(), {0, 0});
      Matrix<element_t, Device::CPU> source(std::move(distribution));

      matrix::test::set(source, [ld = source.size().rows()](auto&& i) {
        const auto value = i.row() + i.col() * ld;
        return (value % 2 == 0) ? value : -value;
      });
      return source;
    }();

    std::string output{"M = np.zeros((5, 4), dtype=np.single)\n"};

    const auto linear_rank =
        comm_grid_.rank().row() + comm_grid_.rank().col() * comm_grid_.size().rows();

    if (0 == linear_rank)
      output += "M[0:2,0:3] = np.array([0,-1,-5,6,10,-11,], dtype=np.single).reshape(3, 2).T\n";
    else if (1 == linear_rank)
      output += "M[2:4,0:3] = np.array([2,-3,-7,8,12,-13,], dtype=np.single).reshape(3, 2).T\n";
    else if (2 == linear_rank)
      output += "M[4:5,0:3] = np.array([4,-9,14,], dtype=np.single).reshape(3, 1).T\n";
    else if (3 == linear_rank)
      output += "M[0:2,3:4] = np.array([-15,16,], dtype=np.single).reshape(1, 2).T\n";
    else if (4 == linear_rank)
      output += "M[2:4,3:4] = np.array([-17,18,], dtype=np.single).reshape(1, 2).T\n";
    else if (5 == linear_rank)
      output += "M[4:5,3:4] = np.array([-19,], dtype=np.single).reshape(1, 1).T\n";

    std::stringstream output_csv;
    output_csv << mat << std::endl;

    return std::make_tuple(std::move(mat), output, output_csv.str());
  }
};

template <class T>
struct test_matrix_dist_output<std::complex<T>> {
  using element_t = std::complex<T>;

  comm::CommunicatorGrid comm_grid_;

  test_matrix_dist_output() : comm_grid_({MPI_COMM_WORLD}, {3, 2}, common::Ordering::ColumnMajor) {}

  auto empty() {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Distribution distribution({0, 0}, {2, 3}, comm_grid_.size(), comm_grid_.rank(), {0, 0});
      Matrix<element_t, Device::CPU> source(std::move(distribution));
      return source;
    }();

    const std::string output{"M = np.zeros((0, 0), dtype=np.csingle)\n"};

    std::stringstream output_csv;
    output_csv << mat << std::endl;

    return std::make_tuple(std::move(mat), output, output_csv.str());
  }

  auto nonempty() {
    Matrix<const element_t, Device::CPU> mat = [&]() {
      Distribution distribution({5, 4}, {2, 3}, comm_grid_.size(), comm_grid_.rank(), {0, 0});
      Matrix<element_t, Device::CPU> source(std::move(distribution));

      matrix::test::set(source, [ld = source.size().rows()](auto&& i) {
        return element_t(i.row(), i.col() % 2 == 0 ? i.col() : -i.col());
      });
      return source;
    }();

    std::string o{"M = np.zeros((5, 4), dtype=np.csingle)\n"};

    const auto linear_rank =
        comm_grid_.rank().row() + comm_grid_.rank().col() * comm_grid_.size().rows();

    if (0 == linear_rank)
      o += "M[0:2,0:3] = np.array([0+0j,1+0j,0-1j,1-1j,0+2j,1+2j,], dtype=np.csingle).reshape(3, 2).T\n";
    else if (1 == linear_rank)
      o += "M[2:4,0:3] = np.array([2+0j,3+0j,2-1j,3-1j,2+2j,3+2j,], dtype=np.csingle).reshape(3, 2).T\n";
    else if (2 == linear_rank)
      o += "M[4:5,0:3] = np.array([4+0j,4-1j,4+2j,], dtype=np.csingle).reshape(3, 1).T\n";
    else if (3 == linear_rank)
      o += "M[0:2,3:4] = np.array([0-3j,1-3j,], dtype=np.csingle).reshape(1, 2).T\n";
    else if (4 == linear_rank)
      o += "M[2:4,3:4] = np.array([2-3j,3-3j,], dtype=np.csingle).reshape(1, 2).T\n";
    else if (5 == linear_rank)
      o += "M[4:5,3:4] = np.array([4-3j,], dtype=np.csingle).reshape(1, 1).T\n";

    std::stringstream output_csv;
    output_csv << mat << std::endl;

    return std::make_tuple(std::move(mat), o, output_csv.str());
  }
};

TYPED_TEST(MatrixOutputTest, NumpyFormatMatrixDist) {
  using test_output_t = test_matrix_dist_output<TypeParam>;
  test_output_t instance;

  for (auto get_test_config : {&test_output_t::empty, &test_output_t::nonempty}) {
    auto config = (instance.*get_test_config)();
    auto& matrix = std::get<0>(config);
    std::string output_np = std::get<1>(config);

    std::ostringstream stream_matrix_output;
    print(format::numpy{}, "M", matrix, stream_matrix_output);

    EXPECT_EQ(output_np, stream_matrix_output.str());
  }
}

#ifdef DLAF_WITH_CUDA
TYPED_TEST(MatrixOutputTestGPU, NumpyFormatMatrixDist) {
  using test_output_t = test_matrix_dist_output<TypeParam>;
  test_output_t instance;

  for (auto get_test_config : {&test_output_t::empty, &test_output_t::nonempty}) {
    auto config = (instance.*get_test_config)();
    auto& matrix = std::get<0>(config);
    MatrixMirror<const TypeParam, Device::GPU, Device::CPU> matrix_d(matrix);

    std::string output_np = std::get<1>(config);

    std::ostringstream stream_matrix_output;
    print(format::numpy{}, "M", matrix_d.get(), stream_matrix_output);

    EXPECT_EQ(output_np, stream_matrix_output.str());
  }
}
#endif

TYPED_TEST(MatrixOutputTest, CsvFormatMatrixDist) {
  using test_output_t = test_matrix_dist_output<TypeParam>;
  test_output_t instance;

  for (auto get_test_config : {&test_output_t::empty, &test_output_t::nonempty}) {
    auto config = (instance.*get_test_config)();
    auto& matrix = std::get<0>(config);
    std::string output_csv = std::get<2>(config);

    std::ostringstream stream_matrix_output;
    print(format::csv{}, "M", matrix, stream_matrix_output);

    EXPECT_EQ(output_csv, stream_matrix_output.str());
  }
}
