//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/traits.h"

namespace dlaf {
namespace matrix {
namespace test {

/// Sets the elements of the tile.
///
/// The (i, j)-element of the tile is set to el({i, j}).
/// @pre el argument is an index of type const TileElementIndex& or TileElementIndex,
/// @pre el return type should be T.
template <class T, class ElementGetter,
          enable_if_signature_t<ElementGetter, T(const TileElementIndex&), int> = 0>
void set(const Tile<T, Device::CPU>& tile, ElementGetter el) {
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      TileElementIndex index(i, j);
      tile(index) = el(index);
    }
  }
}

/// Sets all the elements of the tile with given value.
///
/// @pre el is an element of a type U that can be assigned to T.
template <class T, class U, enable_if_convertible_t<U, T, int> = 0>
void set(const Tile<T, Device::CPU>& tile, U el) {
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      tile(TileElementIndex{i, j}) = el;
    }
  }
}

/// Print the elements of the tile.
template <class T>
void print(const Tile<T, Device::CPU>& tile, int precision = 4, std::ostream& out = std::cout) {
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

/// Create a MemoryView and initialize a Tile on it.
///
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created.
template <class T>
Tile<T, Device::CPU> createTile(const TileElementSize size, const SizeType ld) {
  memory::MemoryView<T, Device::CPU> support_mem(ld * size.cols());
  return Tile<T, Device::CPU>(size, std::move(support_mem), ld);
}

/// Create a tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created.
template <class T, class ElementGetter>
Tile<T, Device::CPU> createTile(ElementGetter val, const TileElementSize size, const SizeType ld) {
  auto tile = createTile<std::remove_const_t<T>>(size, ld);
  set(tile, val);
  return Tile<T, Device::CPU>(std::move(tile));
}

namespace internal {
/// Checks the elements of the tile.
///
/// comp(expected({i, j}), (i, j)-element) is used to compare the elements.
/// err_message(expected({i, j}), (i, j)-element) is printed for the first element which does not fulfill
/// the comparison.
/// @pre expected argument is an index of type const TileElementIndex&,
/// @pre comp should have two arguments and return true if the comparison is fulfilled and false otherwise,
/// @pre err_message should have two arguments and return a string,
/// @pre expected return type should be the same as the type of the first argument of comp and of err_message,
/// @pre The second argument of comp should be either T, T& or const T&,
/// @pre The second argument of err_message should be either T, T& or const T&.
template <class T, class ElementGetter, class ComparisonOp, class ErrorMessageGetter,
          std::enable_if_t<!std::is_convertible<ElementGetter, T>::value, int> = 0>
void check(ElementGetter&& expected, const Tile<const T, Device::CPU>& tile, ComparisonOp comp,
           ErrorMessageGetter err_message, const char* file, const int line) {
  for (SizeType j = 0; j < tile.size().cols(); ++j) {
    for (SizeType i = 0; i < tile.size().rows(); ++i) {
      TileElementIndex index(i, j);
      if (!comp(expected(index), tile(index))) {
        ADD_FAILURE_AT(file, line) << "Error at index (" << i << ", " << j
                                   << "): " << err_message(expected({i, j}), tile({i, j})) << std::endl;
        return;
      }
    }
  }
}

/// Checks the elements of the tile w.r.t. a fixed value.
///
/// comp(expected, (i, j)-element) is used to compare the elements.
/// err_message(expected, (i, j)-element) is printed for the first element which does not fulfill
/// the comparison.
/// @pre comp should have two arguments and return true if the comparison is fulfilled and false otherwise,
/// @pre err_message should have two arguments and return a string,
/// @pre expected type should be the same as the type of the first argument of comp and of err_message,
/// @pre the second argument of comp should be either T, T& or const T&,
/// @pre the second argument of err_message should be either T, T& or const T&.
template <class T, class U, class ComparisonOp, class ErrorMessageGetter,
          enable_if_convertible_t<U, T, int> = 0>
void check(U expected, const Tile<const T, Device::CPU>& tile, ComparisonOp comp,
           ErrorMessageGetter err_message, const char* file, const int line) {
  check([expected](TileElementIndex) { return expected; }, tile, comp, err_message, file, line);
}
}

/// Checks the elements of the tile (exact equality).
///
/// The (i, j)-element of the tile is compared to exp_el({i, j}).
/// @pre exp_el argument is an index of type const TileElementIndex&,
/// @pre exp_el return type should be T.
template <class T, class ElementGetter>
void checkEQ(ElementGetter&& exp_el, const Tile<const T, Device::CPU>& tile, const char* file,
             const int line) {
  auto err_message = [](T expected, T value) {
    std::stringstream s;
    s << "expected " << expected << " == " << value;
    return s.str();
  };
  internal::check(exp_el, tile, std::equal_to<T>{}, err_message, file, line);
}
#define CHECK_TILE_EQ(exp_el, tile) ::dlaf::matrix::test::checkEQ(exp_el, tile, __FILE__, __LINE__)

/// Checks the pointers to the elements of the tile.
///
/// The pointer to (i, j)-element of the matrix is compared to exp_ptr({i, j}).
/// @pre exp_ptr argument is an index of type const TileElementIndex&,
/// @pre exp_ptr return type should be T*.
template <class T, class PointerGetter>
void checkPtr(PointerGetter exp_ptr, const Tile<const T, Device::CPU>& tile, const char* file,
              const int line) {
  auto comp = [](const T* ptr, const T& value) { return ptr == &value; };
  auto err_message = [](const T* expected, const T& value) {
    std::stringstream s;
    s << "expected " << expected << " == " << &value;
    return s.str();
  };
  internal::check(exp_ptr, tile, comp, err_message, file, line);
}
#define CHECK_TILE_PTR(exp_ptr, tile) ::dlaf::matrix::test::checkPtr(exp_ptr, tile, __FILE__, __LINE__)

/// Checks the elements of the tile.
///
/// The (i, j)-element of the tile is compared to expected({i, j}).
/// @pre expected argument is an index of type const TileElementIndex&,
/// @pre expected return type should be T,
/// @pre rel_err >= 0,
/// @pre abs_err >= 0,
/// @pre rel_err > 0 || abs_err > 0.
template <class T, class ElementGetter>
void checkNear(ElementGetter&& expected, const Tile<const T, Device::CPU>& tile, BaseType<T> rel_err,
               BaseType<T> abs_err, const char* file, const int line) {
  ASSERT_GE(rel_err, 0);
  ASSERT_GE(abs_err, 0);
  ASSERT_TRUE(rel_err > 0 || abs_err > 0);

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
      << rel_err << ", Absolute diff: " << diff << " > " << abs_err << ")";
    return s.str();
  };
  internal::check(expected, tile, comp, err_message, file, line);
}
#define CHECK_TILE_NEAR(expected, tile, rel_err, abs_err) \
  ::dlaf::matrix::test::checkNear(expected, tile, rel_err, abs_err, __FILE__, __LINE__)
}
}
}
