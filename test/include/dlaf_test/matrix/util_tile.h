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
#include "dlaf/matrix/copy_tile.h"
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

  int width = std::is_same_v<T, ComplexType<T>> ? 2 * base_width + 3 : base_width;

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
template <class T, Device D = Device::CPU>
Tile<T, D> createTile(const TileElementSize size, const SizeType ld) {
  memory::MemoryView<T, D> support_mem(ld * size.cols());
  return Tile<T, D>(size, std::move(support_mem), ld);
}

namespace internal {
template <class T, Device D>
struct CreateTile;

template <class T>
struct CreateTile<T, Device::CPU> {
  template <class ElementGetter>
  static Tile<T, Device::CPU> createAndSet(ElementGetter val, const TileElementSize size,
                                           const SizeType ld) {
    auto tile = createTile<std::remove_const_t<T>, Device::CPU>(size, ld);
    set(tile, val);
    return Tile<T, Device::CPU>(std::move(tile));
  }
};

template <class T>
struct CreateTile<T, Device::GPU> {
  template <class ElementGetter>
  static Tile<T, Device::GPU> createAndSet(ElementGetter val, const TileElementSize size,
                                           const SizeType ld) {
    auto tile_host = CreateTile<T, Device::CPU>::createAndSet(val, size, ld);
    auto tile = createTile<std::remove_const_t<T>, Device::GPU>(size, ld);
    dlaf::matrix::internal::copy(tile_host, tile);
    return Tile<T, Device::GPU>(std::move(tile));
  }
};
}

/// Create a tile and fill with selected values
///
/// @pre val argument is an index of type const TileElementIndex&,
/// @pre val return type should be T;
/// @pre size is the dimension of the tile to be created (type: TileElementSize);
/// @pre ld is the leading dimension of the tile to be created.
template <class T, Device D = Device::CPU, class ElementGetter>
Tile<T, D> createTile(ElementGetter val, const TileElementSize size, const SizeType ld) {
  return internal::CreateTile<T, D>::createAndSet(val, size, ld);
}

/// Infinite size tile with fixed value
///
/// Return a callable that acts like a tile that always return @p value,
/// for any @p index given
template <class T>
auto fixedValueTile(T value) noexcept {
  return [=](TileElementIndex) noexcept { return value; };
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
          std::enable_if_t<!std::is_convertible_v<ElementGetter, T>, int> = 0>
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

#ifdef DLAF_WITH_CUDA
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
          std::enable_if_t<!std::is_convertible_v<ElementGetter, T>, int> = 0>
void check(ElementGetter&& expected, const Tile<const T, Device::GPU>& tile, ComparisonOp comp,
           ErrorMessageGetter err_message, const char* file, const int line) {
  auto tile_host = createTile<std::remove_const_t<T>, Device::CPU>(tile.size(), tile.ld());
  dlaf::matrix::internal::copy(tile, tile_host);
  check(std::forward<ElementGetter>(expected), tile_host, comp, err_message, file, line);
}
#endif

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
template <class T, Device D, class U, class ComparisonOp, class ErrorMessageGetter,
          enable_if_convertible_t<U, T, int> = 0>
void check(U expected, const Tile<const T, D>& tile, ComparisonOp comp, ErrorMessageGetter err_message,
           const char* file, const int line) {
  check([expected](TileElementIndex) { return expected; }, tile, comp, err_message, file, line);
}
}

/// Checks the elements of the tile (exact equality).
///
/// The (i, j)-element of the tile is compared to exp_el({i, j}).
/// @pre exp_el argument is an index of type const TileElementIndex&,
/// @pre exp_el return type should be T.
template <class T, Device D, class ElementGetter>
void checkEQ(ElementGetter&& exp_el, const Tile<const T, D>& tile, const char* file, const int line) {
  auto err_message = [](T expected, T value) {
    std::stringstream s;
    s.precision(17);
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
template <class T, Device D, class PointerGetter>
void checkPtr(PointerGetter&& exp_ptr, const Tile<const T, D>& tile, const char* file, const int line) {
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
template <class T, Device D, class ElementGetter>
void checkNear(ElementGetter&& expected, const Tile<const T, D>& tile, BaseType<T> rel_err,
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

template <class T>
void checkEvecsNearOrOpposite(Tile<const T, Device::CPU>& expected_evecs,
                              Tile<const T, Device::CPU>& actual_evecs, BaseType<T> rel_tol,
                              BaseType<T> abs_tol, const char* file, const int line) {
  ASSERT_GE(rel_tol, 0);
  ASSERT_GE(abs_tol, 0);
  ASSERT_TRUE(rel_tol > 0 || abs_tol > 0);

  // Eigenvectors are unqiue up to a sign.
  for (SizeType j = 0; j < actual_evecs.size().cols(); ++j) {
    BaseType<T> sign = 1.0;
    for (SizeType i = 0; i < actual_evecs.size().rows(); ++i) {
      TileElementIndex index(i, j);
      auto expected = expected_evecs(index);
      auto actual = actual_evecs(index);
      // If the first elements of the eigenvectors (expected and actual) are opposite, all remaining
      // elements have to be opposite but equal in magnitude for the two vectors to be considered equal.
      if (i == 0 && std::abs(expected + actual) < abs_tol) {
        sign = -1.0;
      }
      auto abs_err = std::abs(expected - sign * actual);
      auto rel_err = abs_err / std::max(std::abs(expected), std::abs(actual));
      if ((abs_err > abs_tol) && (rel_err > rel_tol)) {
        // clang-format off
        ADD_FAILURE_AT(file, line) << "Error at index " << index << "!\n"
          << "The actual value is "        << actual  << " but the expected value is "     << expected  << ".\n"
          << "The relative difference is " << rel_err << " but the relative tolerance is " << rel_tol   << ".\n"
          << "The absolute difference is " << abs_err << " but the absolute tolerance is " << abs_tol   << "."  << std::endl;
        // clang-format on
        return;
      }
    }
  }
}
#define CHECK_EVECS_NEAR_OR_OPPOSITE(expected_evecs, actual_evecs, rel_tol, abs_tol)             \
  ::dlaf::matrix::test::checkEvecsNearOrOpposite(expected_evecs, actual_evecs, rel_tol, abs_tol, \
                                                 __FILE__, __LINE__)

}
}
}
