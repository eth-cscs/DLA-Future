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

#include "dlaf/matrix/print_numpy.h"
#include "dlaf/matrix_output.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/matrix/util_matrix_blas.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::comm;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf_test;
using namespace testing;

template <typename Type>
class MatrixOutputTest : public ::testing::Test {};

TYPED_TEST_SUITE(MatrixOutputTest, MatrixElementTypes);

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

GlobalElementSize globalTestSize(const LocalElementSize& size) {
  return {size.rows(), size.cols()};
}

TYPED_TEST(MatrixOutputTest, printElements) {
  using Type = float;
  auto el = [](const GlobalElementIndex& index) {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<Type>::element(i + j, j - i);
  };

  for (const auto& sz : sizes) {
    Matrix<Type, Device::CPU> mat(sz.size, sz.block_size);
    EXPECT_EQ(Distribution(sz.size, sz.block_size), mat.distribution());

    set(mat, el);

    std::cout << "Matrix mat " << mat << std::endl;
    std::cout << "Printing elements" << std::endl;
    printElements(mat);
  }
}

template <class T>
struct is_complex : public std::false_type {};

template <class T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <class T>
struct numpy_type {
  static constexpr auto name = "float";

  static T deserialize(const std::string& value_str) {
    std::istringstream deserializer(value_str);
    T value;
    deserializer >> value;
    return value;
  }
};

template <class T>
struct numpy_type<std::complex<T>> {
  static constexpr auto name = "complex";

  static std::complex<T> deserialize(const std::string& value_str) {
    const std::string regex_real = "(-?[0-9]+(?:\\.[0-9]+)?)";  // <floating-point value>
    const std::regex regex_complex{regex_real +                 // <real>
                                   "\\s*\\+\\s*" +              // +
                                   regex_real + "[jJ]"};        // <imag>j (or J)

    std::smatch matches;
    DLAF_ASSERT(std::regex_match(value_str, matches, regex_complex), value_str,
                " does not look like a complex number");

    const auto real = numpy_type<T>::deserialize(matches[1]);
    const auto imag = numpy_type<T>::deserialize(matches[2]);

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
  const std::regex regex_csv{"[^,\\s]+"};  // TODO this does not work very well
  auto values_begin = std::sregex_iterator(values_section.begin(), values_section.end(), regex_csv);
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

template <class T>
::testing::AssertionResult parseAndCheckMatrix(const std::string numpy_output,
                                               Matrix<T, Device::CPU>& mat) {
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

  if (rows != mat.size().rows() || cols != mat.size().cols())
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
    const TileElementSize size_tile{i_end - i_beg, j_end - j_beg};

    const auto ij_local = mat.distribution().globalTileIndex(ij_global);

    const auto& tile = mat.read(ij_local).get();

    EXPECT_EQ(tile.size(), size_tile);
    EXPECT_TRUE(parseAndCheckTile(matches[6].str(), tile));
  }

  return ::testing::AssertionSuccess();
}

TYPED_TEST(MatrixOutputTest, NumpyFormat) {
  auto el = [](const GlobalElementIndex& index) -> TypeParam {
    SizeType i = index.row();
    SizeType j = index.col();
    return TypeUtilities<TypeParam>::element(i + j, j - i);
  };

  for (const auto& sz : sizes) {
    Matrix<TypeParam, Device::CPU> mat(sz.size, sz.block_size);
    EXPECT_EQ(Distribution(sz.size, sz.block_size), mat.distribution());

    set(mat, el);

    std::ostringstream stream_matrix_output;
    print_numpy(stream_matrix_output, mat, "mat");

    EXPECT_TRUE(parseAndCheckMatrix(stream_matrix_output.str(), mat));

    for (const auto& index : iterate_range2d(mat.nrTiles())) {
      const auto& tile = mat.read(index).get();

      std::ostringstream stream_tile_output;
      print_numpy(stream_tile_output, tile);

      EXPECT_TRUE(parseAndCheckTile(stream_tile_output.str(), tile));
    }
  }
}
