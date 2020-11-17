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
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace testing;

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
T el(const GlobalElementIndex& index) {
  SizeType i = index.row();
  SizeType j = index.col();
  return dlaf::test::TypeUtilities<T>::element(i + j, j - i);
};

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
struct numpy_type<std::complex<T>> {
  static constexpr auto name = "csingle";

  static std::complex<T> deserialize(const std::string& value_str) {
    const std::string regex_sign{"([\\+\\-])"};              // capture sign
    const std::string regex_real{"([0-9]+(?:\\.[0-9]+)?)"};  // <floating-point value>
    const std::regex regex_complex{"\\s*" + regex_sign + "?\\s*" + regex_real +  // <(+/-)><real>
                                   "\\s*" + regex_sign + "\\s*" + regex_real +   // <+/-><imag>
                                   "[jJ]"};                                      // j (or J)

    std::smatch matches;
    DLAF_ASSERT(std::regex_match(value_str, matches, regex_complex), value_str,
                "does not look like a complex number");

    const auto real = numpy_type<T>::deserialize(matches[1].str() + matches[2].str());
    const auto imag = numpy_type<T>::deserialize(matches[3].str() + matches[4].str());

    return {real, imag};
  }
};

template <class T>
::testing::AssertionResult parseAndCheckTile(const std::string numpy_output,
                                             const Tile<const T, Device::CPU>& tile) {
  const std::regex regex_syntax{"np.array\\(\\[(.*)\\]\\)"          // np.array([<values>])
                                ".reshape\\((\\d+), (\\d+)\\).T"};  // .reshape(<n>, <n>).T

  std::smatch parts;
  if (!std::regex_match(numpy_output, parts, regex_syntax))
    return ::testing::AssertionFailure() << "syntax not valid for " << numpy_output;

  const std::string values_section = parts[1].str();
  const auto size = transposed(TileElementSize{std::stoi(parts[2].str()), std::stoi(parts[3].str())});

  if (size != tile.size())
    return ::testing::AssertionFailure() << size << " " << tile.size();

  std::vector<T> values;
  const std::regex regex_csv{"[^,\\s][^,]*"};
  auto values_begin = std::sregex_iterator(values_section.begin(), values_section.end(), regex_csv,
                                           std::regex_constants::match_not_null);
  std::transform(values_begin, std::sregex_iterator(), std::back_inserter(values),
                 [](const auto& match) { return numpy_type<T>::deserialize(match.str()); });

  std::vector<T> expected_values;
  const auto tile_elements = iterate_range2d(tile.size());
  std::transform(tile_elements.begin(), tile_elements.end(), std::back_inserter(expected_values),
                 std::ref(tile));

  auto compare_values = [](const T& a, const T& b) {
    return std::abs(a - b) <= std::abs(std::numeric_limits<T>::epsilon());
  };

  if (!std::equal(expected_values.begin(), expected_values.end(), values.begin(), compare_values))
    return ::testing::AssertionFailure() << "values differ between expected and output";

  return ::testing::AssertionSuccess();
}

TYPED_TEST(MatrixOutputLocalTest, NumpyFormatTile) {
  for (const auto& sz : sizes) {
    Matrix<TypeParam, Device::CPU> mat(sz.size, sz.block_size);
    EXPECT_EQ(Distribution(sz.size, sz.block_size), mat.distribution());

    set(mat, el<TypeParam>);

    for (const auto& index : iterate_range2d(mat.nrTiles())) {
      const auto& tile = mat.read(index).get();

      std::ostringstream stream_tile_output;
      print_numpy(stream_tile_output, tile);

      EXPECT_TRUE(parseAndCheckTile(stream_tile_output.str(), tile));
    }
  }
}

template <class T>
::testing::AssertionResult parseAndCheckMatrix(const std::string numpy_output,
                                               Matrix<T, Device::CPU>& mat) {
  const auto dist = mat.distribution();
  std::stringstream stream(numpy_output);

  // look for definition and check it, in the first line
  std::string definition;
  std::getline(stream, definition);

  const std::regex regex_definition(
      "(.+) = np.zeros\\(\\(([0-9]+), ([0-9]+)\\)"  // <symbol> = np.zeros((<rows>, <cols>)
      ", dtype=np\\.(.+)\\)");                      // , dtype=np.<type>)
  std::smatch matches;
  if (!std::regex_match(definition, matches, regex_definition))
    return ::testing::AssertionFailure() << "'" << definition << "'\n"
                                         << "does not look like a numpy matrix definition";

  const std::string symbol = matches[1].str();
  const SizeType rows = std::stoi(matches[2].str());
  const SizeType cols = std::stoi(matches[3].str());

  const std::string matrix_type = matches[4].str();
  if (std::string(numpy_type<T>::name) != matrix_type)
    return ::testing::AssertionFailure() << std::string(numpy_type<T>::name) << " " << matrix_type;

  if (GlobalElementSize{rows, cols} != mat.size())
    return ::testing::AssertionFailure() << mat.size() << " " << rows << " " << cols;

  // each next line must contain an assignment: one for each tile of the matrix
  for (std::string line; std::getline(stream, line);) {
    const std::regex regex_assignment("([^\\[]+)\\[\\s*"           // <symbol>[
                                      "([0-9]+)\\s*:\\s*([0-9]+)"  // <i_beg>:<i_end>
                                      "\\s*,\\s*"                  // ,
                                      "([0-9]+)\\s*:\\s*([0-9]+)"  // <j_beg>:<j_end>
                                      "\\s*\\] = (.+)");           // ] = <tile>

    std::smatch matches;
    if (!std::regex_match(line, matches, regex_assignment))
      return ::testing::AssertionFailure() << line << " is not a valid assignment";

    const std::string assign_symbol = matches[1].str();
    if (symbol != assign_symbol)
      return ::testing::AssertionFailure() << "symbol mismatch:" << symbol << " vs " << assign_symbol;

    const SizeType i_beg = std::stoi(matches[2].str());
    const SizeType i_end = std::stoi(matches[3].str());
    const SizeType j_beg = std::stoi(matches[4].str());
    const SizeType j_end = std::stoi(matches[5].str());

    const GlobalElementIndex ij_global{i_beg, j_beg};
    if (dist.rankIndex() != dist.rankGlobalTile(dist.globalTileIndex(ij_global)))
      continue;

    const TileElementSize size_tile{i_end - i_beg, j_end - j_beg};

    const auto ij_local = dist.globalTileIndex(ij_global);

    const auto& tile = mat.read(ij_local).get();

    EXPECT_EQ(tile.size(), size_tile);
    EXPECT_TRUE(parseAndCheckTile(matches[6].str(), tile));
  }

  return ::testing::AssertionSuccess();
}

TYPED_TEST(MatrixOutputLocalTest, NumpyFormatMatrix) {
  for (const auto& sz : sizes) {
    Matrix<TypeParam, Device::CPU> mat(sz.size, sz.block_size);
    EXPECT_EQ(Distribution(sz.size, sz.block_size), mat.distribution());

    set(mat, el<TypeParam>);

    std::ostringstream stream_matrix_output;
    print_numpy(stream_matrix_output, mat, "mat");

    EXPECT_TRUE(parseAndCheckMatrix(stream_matrix_output.str(), mat));
  }
}

template <class T>
class MatrixOutputTest : public ::testing::Test {
public:
  const std::vector<CommunicatorGrid>& commGrids() {
    return dlaf::test::comm_grids;
  }
};

TYPED_TEST_SUITE(MatrixOutputTest, dlaf::test::MatrixElementTypes);

GlobalElementSize globalTestSize(const LocalElementSize& size, const Size2D& grid_size) {
  return {size.rows() * grid_size.rows(), size.cols() * grid_size.cols()};
}

TYPED_TEST(MatrixOutputTest, NumpyFormatMatrix) {
  for (const auto& comm_grid : this->commGrids()) {
    for (const auto& test : sizes) {
      GlobalElementSize size = globalTestSize(test.size, comm_grid.size());
      Distribution distribution(size, test.block_size, comm_grid.size(), comm_grid.rank(), {0, 0});
      Matrix<TypeParam, Device::CPU> mat(std::move(distribution));

      set(mat, el<TypeParam>);

      std::ostringstream stream_matrix_output;
      print_numpy(stream_matrix_output, mat, "mat");

      EXPECT_TRUE(parseAndCheckMatrix(stream_matrix_output.str(), mat));
    }
  }
}
