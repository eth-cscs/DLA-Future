//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/tile.h"

namespace dlaf_test {
namespace tile_test {
using namespace dlaf;

/// @brief Sets the elements of the tile.
/// The (i, j)-element of the tile is set to el({i, j}) if op == NoTrans,
///                                          el({j, i}) if op == Trans,
///                                          conj(el({j, i})) if op == ConjTrans.
/// @pre el argument is an index of type const TileElementIndex&.
/// @pre el return type should be T.
template <class T, class Func>
void set(Tile<T, Device::CPU>& tile, Func el) {
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      TileElementIndex index(i, j);
      tile(index) = el(index);
    }
  }
}

/// @brief Sets the elements of the tile.
/// The (i, j)-element of the tile is set to el({i, j}) if op == NoTrans,
///                                          el({j, i}) if op == Trans,
///                                          conj(el({j, i})) if op == ConjTrans.
/// @pre el argument is an index of type const TileElementIndex&.
/// @pre el return type should be T.
template <class T>
void print(Tile<T, Device::CPU>& tile, int precision = 4, std::ostream& out = std::cout) {
  auto out_precision = out.precision();
  out.precision(precision);
  // sign + number + . + exponent (e+xxx)
  int base_width = 1 + precision + 1 + 5;

  int width = std::is_same<T, ComplexType<T>>::value ? 2 * base_width + 3 : base_width;

  std::cout << "(" << tile.size().rows() << ", " << tile.size().cols() << ") Tile:" << std::endl;
  for (SizeType i = 0; i < tile.size().rows(); ++i) {
    for (SizeType j = 0; j < tile.size().cols(); ++j) {
      TileElementIndex index(i, j);
      out << std::setw(width) << tile(index) << " ";
    }
    out << std::endl;
  }
  out.precision(out_precision);
}

namespace internal {
/// @brief Checks the elements of the tile.
/// comp(el({i, j}), (i, j)-element) is used to compare the elements.
/// err_message(el({i, j}), (i, j)-element) is printed for the first element which does not fulfill the comparison.
/// @pre el argument is an index of type const TileElementIndex&.
/// @pre comp should have two arguments and return true if the comparison is fulfilled and false otherwise.
/// @pre err_message should have two arguments and return a string.
/// @pre el return type should be the same as the type of the first argument of comp and of err_message.
/// @pre The second argument of comp should be either T, T& or const T&.
/// @pre The second argument of err_message should be either T, T& or const T&.
template <class T, class Func1, class Func2, class Func3>
void check(Tile<T, Device::CPU>& tile, Func1 el, Func2 comp, Func3 err_message, const char* file,
           const int line) {
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      TileElementIndex index(i, j);
      if (!comp(el(index), tile(index))) {
        ADD_FAILURE_AT(file, line) << "Error at index (" << i << ", " << j
                                   << "): " << err_message(el({i, j}), tile({i, j})) << std::endl;
        return;
      }
    }
  }
}
}

/// @brief Checks the elements of the tile (exact equality).
/// The (i, j)-element of the tile is compared to el({i, j}).
/// @pre el argument is an index of type const TileElementIndex&.
/// @pre el return type should be T.
template <class T, class Func>
void checkEQ(Tile<T, Device::CPU>& tile, Func el, const char* file, const int line) {
  auto err_message = [](T expected, T value) {
    std::stringstream s;
    s << "expected " << expected << " == " << value;
    return s.str();
  };
  internal::check(tile, el, std::equal_to<T>{}, err_message, file, line);
}
#define CHECK_TILE_EQ(el, tile) ::dlaf_test::tile_test::checkEQ(tile, el, __FILE__, __LINE__);

/// @brief Checks the pointers to the elements of the tile.
/// The pointer to (i, j)-element of the matrix is compared to ptr({i, j}).
/// @pre ptr argument is an index of type const TileElementIndex&.
/// @pre ptr return type should be T*.
template <class T, class Func>
void checkPtr(Tile<T, Device::CPU>& tile, Func ptr, const char* file, const int line) {
  auto comp = [](T* ptr, const T& value) { return ptr == &value; };
  auto err_message = [](T* expected, const T& value) {
    std::stringstream s;
    s << "expected " << expected << " == " << &value;
    return s.str();
  };
  internal::check(tile, ptr, comp, err_message, file, line);
}
#define CHECK_TILE_PTR(ptr, tile) ::dlaf_test::tile_test::checkPtr(tile, ptr, __FILE__, __LINE__);

/// @brief Checks the elements of the tile.
/// The (i, j)-element of the tile is compared to el({i, j}).
/// @pre el argument is an index of type const TileElementIndex&.
/// @pre el return type should be T.
/// @pre rel_err > 0.
/// @pre abs_err > 0.
template <class T, class Func>
void checkNear(Tile<T, Device::CPU>& tile, Func el, BaseType<T> rel_err, BaseType<T> abs_err,
               const char* file, const int line) {
  ASSERT_GT(rel_err, 0);
  ASSERT_GT(abs_err, 0);

  auto comp = [rel_err, abs_err](T expected, T value) {
    auto diff = std::abs(expected - value);
    auto abs_max = std::max(std::abs(expected), std::abs(value));

    return (diff < abs_err) || (diff / abs_max < rel_err);
  };
  auto err_message = [rel_err, abs_err](T expected, T value) {
    auto diff = std::abs(expected - value);
    auto abs_max = std::max(std::abs(expected), std::abs(value));

    std::stringstream s;
    s << "expected " << expected << " == " << value << " (Relative diff: " << diff / abs_max << " > "
      << rel_err << ", " << diff << " > " << abs_err << ")";
    return s.str();
  };
  internal::check(tile, el, comp, err_message, file, line);
}
#define CHECK_TILE_NEAR(el, tile, rel_err, abs_err) \
  ::dlaf_test::tile_test::checkNear(tile, el, rel_err, abs_err, __FILE__, __LINE__);

}
}
