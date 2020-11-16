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
struct numpy_type;

template <>
struct numpy_type<float> {
  static constexpr auto name = "float";
};

template <>
struct numpy_type<double> {
  static constexpr auto name = "float";
};

template <class T>
struct numpy_type<std::complex<T>> {
  static constexpr auto name = "complex";
};

template <class T>
T deserialize_value(const std::string& value_str) {
  std::istringstream deserializer(value_str);
  if (is_complex<T>::value) {
    char prefix_value[] = {"complex"};
    deserializer.read(prefix_value, sizeof prefix_value - 1);
    if (std::string("complex") != std::string(prefix_value))
      throw std::runtime_error("error during complex deserialization");
  }

  T value;
  deserializer >> value;
  return value;
}

template <class T>
bool compare_values(const T& a, const T& b) {
  return std::abs(a - b) <= std::abs(std::numeric_limits<T>::epsilon());
}

template <class MatrixValues>
::testing::AssertionResult parseAndCheckMatrix(const std::string numpy_output,
                                               const LocalElementSize& size, MatrixValues&& values) {
  using T = typename std::remove_const_t<decltype(values({0, 0}))>;

  std::stringstream stream(numpy_output);

  // look for definition and check it, in the first line
  std::string definition;
  std::getline(stream, definition);

  std::regex regex_definition("(.+) = np.zeros\\(\\(([0-9]+), ([0-9]+)\\), dtype=np\\.(.+)\\)");
  std::smatch matches;
  std::regex_match(definition, matches, regex_definition);

  const std::string symbol = matches[1].str();

  const SizeType rows = std::stoi(matches[2].str());
  const SizeType cols = std::stoi(matches[3].str());

  const std::string type = matches[4].str();
  if (std::string(numpy_type<T>::name) != type)
    return ::testing::AssertionFailure();

  if (rows != size.rows() || cols != size.cols())
    return ::testing::AssertionFailure() << size << " " << rows << " " << cols;

  // each next line must contain an assignment: one for each element of the matrix
  SizeType count_values = 0;
  for (std::string line; std::getline(stream, line); ++count_values) {
    const std::regex regex_assignment("([^\\[]+)\\[([0-9]+),([0-9]+)\\] = (.+)");

    std::smatch matches;
    if (!std::regex_match(line, matches, regex_assignment))
      return ::testing::AssertionFailure() << line << " is not a valid assignment";

    const std::string assign_symbol = matches[1].str();
    if (symbol != assign_symbol)
      return ::testing::AssertionFailure()
             << "defined " << symbol << " but assigning to " << assign_symbol;

    const SizeType i = std::stoi(matches[2].str());
    const SizeType j = std::stoi(matches[3].str());
    const T value = deserialize_value<T>(matches[4]);

    const auto expected_value = values({i, j});

    if (!compare_values(expected_value, value))
      return ::testing::AssertionFailure() << value << " " << expected_value << " " << matches[4].str();
  }

  if (count_values != size.rows() * size.cols())
    return ::testing::AssertionFailure()
           << "assigned " << count_values << " of " << size.rows() * size.cols();

  return ::testing::AssertionSuccess();
}

template <class T>
::testing::AssertionResult parseAndCheckTile(const std::string numpy_output,
                                             const Tile<const T, Device::CPU>& tile) {
  // np.array([complex(12,0), complex(13,-1), complex(13,1), complex(14,0), ]).reshape(2, 2).T
  const auto syntax = "np.array\\(\\[(.*)\\]\\).reshape\\((\\d+), (\\d+)\\).T";
  const std::regex regex_syntax(syntax);

  std::smatch syntax_components;
  if (!std::regex_match(numpy_output, syntax_components, regex_syntax))
    return ::testing::AssertionFailure() << "syntax not valid for " << numpy_output;

  const std::string values_section = syntax_components[1].str();
  const TileElementSize size{std::stoi(syntax_components[2].str()),
                             std::stoi(syntax_components[3].str())};

  if (size != tile.size())
    return ::testing::AssertionFailure() << size << " " << tile.size();

  const auto regex_real = "(?:-?(?:[0-9]+\\.)?[0-9]+)";
  const auto regex_complex = std::string("complex\\(") + regex_real + ",\\s*" + regex_real + "\\)";
  const std::regex regex_value{std::string("(") + regex_real + "|" + regex_complex + ")"};

  std::vector<T> values;
  auto values_begin = std::sregex_iterator(values_section.begin(), values_section.end(), regex_value);
  auto deserialize_match = [](const auto& match) { return deserialize_value<T>(match.str()); };
  std::transform(values_begin, std::sregex_iterator(), std::back_inserter(values), deserialize_match);

  std::vector<T> expected_values;
  const auto tile_elements = iterate_range2d(tile.size());
  std::transform(tile_elements.begin(), tile_elements.end(), std::back_inserter(expected_values),
                 std::ref(tile));

  // TODO equal is not the best choice, it would be better std::abs(a-b)
  if (!std::equal(std::begin(expected_values), std::end(expected_values), std::begin(values),
                  compare_values<T>))
    return ::testing::AssertionFailure() << "values differ between expected and output";

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

    EXPECT_TRUE(parseAndCheckMatrix(stream_matrix_output.str(), sz.size, el));

    for (const auto& index : iterate_range2d(mat.nrTiles())) {
      const auto& tile = mat.read(index).get();

      std::ostringstream stream_tile_output;
      print_numpy(stream_tile_output, tile);

      EXPECT_TRUE(parseAndCheckTile(stream_tile_output.str(), tile));
    }
  }
}
