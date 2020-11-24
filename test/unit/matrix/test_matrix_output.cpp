//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/matrix/print_numpy.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include "dlaf_test/comm_grids/grids_6_ranks.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::matrix;

using dlaf::matrix::test::set;
using matrix::test::createTile;

::testing::Environment* const comm_grids_env =
    ::testing::AddGlobalTestEnvironment(new dlaf::test::CommunicatorGrid6RanksEnvironment);

template <typename Type>
class MatrixOutputLocalTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixOutputLocalTest, dlaf::test::MatrixElementTypes);

struct TestSizes {
  LocalElementSize size;
  TileElementSize block_size;
};

const std::vector<TestSizes> sizes({
    {{0, 0}, {2, 2}},
    {{6, 6}, {2, 2}},
    {{6, 6}, {3, 3}},
    {{8, 8}, {3, 3}},
    {{9, 7}, {3, 3}},
});

template <class T>
T pattern_values(const GlobalElementIndex& index) {
  SizeType i = index.row();
  SizeType j = index.col();
  return dlaf::test::TypeUtilities<T>::element(i + j, j - i);
}

template <class T>
struct numpy_type {
  static constexpr auto name = "single";

  static T deserialize(const std::string& value_str) {
    std::istringstream deserializer(value_str);
    T value;
    deserializer >> value;
    return value;
  }
};

template <class T>
struct test_tile_output {
  using element_t = T;

  static auto tile_empty() {
    const TileElementSize size{0, 0};
    const SizeType ld = 1;
    auto mat = createTile<const element_t>([](auto&&) { return T(); }, size, ld);

    const std::string expected_output{"np.array([], dtype=np.single).reshape(0, 0).T\n"};

    return std::make_pair(std::move(mat), expected_output);
  }

  static auto tile_nonempty() {
    const TileElementSize size{3, 2};
    const SizeType ld = size.rows();
    auto mat = createTile<const element_t>(
        [ld](auto&& i) {
          const auto value = i.row() + i.col() * ld;
          return (value % 2 == 0) ? value : -value;
        },
        size, ld);

    const std::string expected_output{"np.array([0,-1,2,-3,4,-5,], dtype=np.single).reshape(2, 3).T\n"};

    return std::make_pair(std::move(mat), expected_output);
  }
};

template <class T>
struct test_tile_output<std::complex<T>> {
  using element_t = std::complex<T>;

  static auto tile_empty() {
    const TileElementSize size{0, 0};
    auto mat = createTile<const element_t>([](auto&&) { return T(); }, size, 1);

    const std::string expected_output{"np.array([], dtype=np.csingle).reshape(0, 0).T\n"};

    return std::make_pair(std::move(mat), expected_output);
  }

  static auto tile_nonempty() {
    const TileElementSize size{3, 2};
    auto mat = createTile<
        const element_t>([](auto&&
                                i) { return element_t(i.row(), i.col() % 2 == 0 ? i.col() : -i.col()); },
                         size, size.rows());

    const std::string expected_output{
        "np.array([0+0j,1+0j,2+0j,0-1j,1-1j,2-1j,], dtype=np.csingle).reshape(2, 3).T\n"};

    return std::make_pair(std::move(mat), expected_output);
  }
};

TYPED_TEST(MatrixOutputLocalTest, NumpyFormatTile) {
  using test_output = test_tile_output<TypeParam>;
  for (auto get_test_config : {test_output::tile_empty, test_output::tile_nonempty}) {
    const auto config = get_test_config();

    std::ostringstream stream_tile_output;
    print(format::numpy{}, config.first, stream_tile_output);

    EXPECT_EQ(config.second, stream_tile_output.str());
  }
}

TYPED_TEST(MatrixOutputLocalTest, NumpyFormatMatrix) {
  for (const auto& sz : sizes) {
    Matrix<TypeParam, Device::CPU> mat(sz.size, sz.block_size);
    EXPECT_EQ(Distribution(sz.size, sz.block_size), mat.distribution());

    set(mat, pattern_values<TypeParam>);

    std::ostringstream stream_matrix_output;
    print(format::numpy{}, "mat", mat, stream_matrix_output);

    // EXPECT_TRUE(parseAndCheckMatrix(stream_matrix_output.str(), mat));
  }
}

template <class T>
class MatrixOutputTest : public ::testing::Test {
public:
  const auto& commGrids() {
    return dlaf::test::comm_grids;
  }
};

TYPED_TEST_SUITE(MatrixOutputTest, dlaf::test::MatrixElementTypes);

GlobalElementSize globalTestSize(const LocalElementSize& size, const comm::Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

TYPED_TEST(MatrixOutputTest, NumpyFormatMatrix) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      Matrix<TypeParam, Device::CPU> mat(std::move(distribution));

      set(mat, pattern_values<TypeParam>);

      std::ostringstream stream_matrix_output;
      print(format::numpy{}, "mat", mat, stream_matrix_output);

      // EXPECT_TRUE(parseAndCheckMatrix(stream_matrix_output.str(), mat));
    }
  }
}
