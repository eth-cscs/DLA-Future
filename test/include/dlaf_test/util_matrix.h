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
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/layout_info.h"

namespace dlaf_test {
namespace matrix_test {
using namespace dlaf;

/// @brief Sets the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(Matrix<T, Device::CPU>& mat, ElementGetter el) {
  for (SizeType tile_j = 0; tile_j < mat.nrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < mat.nrTiles().rows(); ++tile_i) {
      auto tile = mat(LocalTileIndex(tile_i, tile_j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        SizeType j = tile_j * mat.blockSize().cols() + jj;
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
          SizeType i = tile_i * mat.blockSize().rows() + ii;
          tile({ii, jj}) = el({i, j});
        }
      }
    }
  }
}

/// @brief Returns a col-major ordered vector with the futures to the matrix tiles.
template <class T, Device device>
std::vector<hpx::future<Tile<T, device>>> getFutures(Matrix<T, device>& mat) {
  std::vector<hpx::future<Tile<T, device>>> result;
  result.reserve(util::size_t::mul(mat.nrTiles().rows(), mat.nrTiles().cols()));

  for (SizeType j = 0; j < mat.nrTiles().cols(); ++j) {
    for (SizeType i = 0; i < mat.nrTiles().rows(); ++i) {
      result.emplace_back(std::move(mat(LocalTileIndex(i, j))));
      EXPECT_TRUE(result.back().valid());
    }
  }
  return result;
}

/// @brief Returns a col-major ordered vector with the read-only shared-futures to the matrix tiles.
template <class T, Device device>
std::vector<hpx::shared_future<Tile<const T, device>>> getSharedFutures(Matrix<T, device>& mat) {
  std::vector<hpx::shared_future<Tile<const T, device>>> result;
  result.reserve(util::size_t::mul(mat.nrTiles().rows(), mat.nrTiles().cols()));

  for (SizeType j = 0; j < mat.nrTiles().cols(); ++j) {
    for (SizeType i = 0; i < mat.nrTiles().rows(); ++i) {
      result.emplace_back(mat.read(LocalTileIndex(i, j)));
      EXPECT_TRUE(result.back().valid());
    }
  }

  return result;
}

/// @brief Checks the elements of the matrix.
///
/// comp(expected({i, j}), (i, j)-element) is used to compare the elements.
/// err_message(expected({i, j}), (i, j)-element) is printed for the first element
/// that does not fulfill the comparison.
/// @pre expected argument is an index of type const GlobalElementIndex&.
/// @pre comp should have two arguments and return true if the comparison is fulfilled and false otherwise.
/// @pre err_message should have two arguments and return a string.
/// @pre expected return type should be the same as the type of the first argument of comp and of err_message.
/// @pre The second argument of comp should be either T, T& or const T&.
/// @pre The second argument of err_message should be either T, T& or const T&.
namespace internal {
template <class T, class ElementGetter, class ComparisonOp, class ErrorMessageGetter>
void check(Matrix<T, Device::CPU>& mat, ElementGetter expected, ComparisonOp comp,
           ErrorMessageGetter err_message, const char* file, const int line) {
  for (SizeType tile_j = 0; tile_j < mat.nrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < mat.nrTiles().rows(); ++tile_i) {
      auto& tile = mat.read(LocalTileIndex(tile_i, tile_j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        SizeType j = tile_j * mat.blockSize().cols() + jj;
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
          SizeType i = tile_i * mat.blockSize().rows() + ii;
          if (!comp(expected({i, j}), tile({ii, jj}))) {
            ADD_FAILURE_AT(file, line)
                << "Error at index (" << i << ", " << j
                << "): " << err_message(expected({i, j}), tile({ii, jj})) << std::endl;
            return;
          }
        }
      }
    }
  }
}
}

/// @brief Checks the elements of the matrix (exact equality).
///
/// The (i, j)-element of the matrix is compared to exp_el({i, j}).
/// @pre exp_el argument is an index of type const GlobalElementIndex&.
/// @pre exp_el return type should be T.
template <class T, class ElementGetter>
void checkEQ(Matrix<T, Device::CPU>& mat, ElementGetter exp_el, const char* file, const int line) {
  auto err_message = [](T expected, T value) {
    std::stringstream s;
    s << "expected " << expected << " == " << value;
    return s.str();
  };
  internal::check(mat, exp_el, std::equal_to<T>{}, err_message, file, line);
}
#define CHECK_MATRIX_EQ(mat, exp_el) ::dlaf_test::matrix_test::checkEQ(exp_el, mat, __FILE__, __LINE__);

/// @brief Checks the pointers to the elements of the matrix.
///
/// The pointer to (i, j)-element of the matrix is compared to ptr({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T*.
template <class T, class PointerGetter>
void checkPtr(Matrix<T, Device::CPU>& mat, PointerGetter exp_ptr, const char* file, const int line) {
  auto comp = [](T* ptr, const T& value) { return ptr == &value; };
  auto err_message = [](T* expected, const T& value) {
    std::stringstream s;
    s << "expected " << expected << " == " << &value;
    return s.str();
  };
  internal::check(mat, exp_ptr, comp, err_message, file, line);
}
#define CHECK_MATRIX_PTR(exp_ptr, mat) \
  ::dlaf_test::matrix_test::checkPtr(mat, exp_ptr, __FILE__, __LINE__);

}
}
