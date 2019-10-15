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
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/layout_info.h"

namespace dlaf_test {
namespace matrix_test {
using namespace dlaf;

/// @brief Returns the pointer to the index element of the matrix.
/// @pre index should be a valid and contained in layout.size().
template <class T>
T* getPtr(T* base_ptr, const matrix::LayoutInfo& layout, const GlobalElementIndex& index) {
  const auto& block_size = layout.blockSize();
  SizeType tile_i = index.row() / block_size.rows();
  SizeType tile_j = index.col() / block_size.cols();
  std::size_t offset = layout.tileOffset({tile_i, tile_j});
  SizeType i = index.row() % block_size.rows();
  SizeType j = index.col() % block_size.cols();
  return base_ptr + offset + i + layout.ldTile() * j;
}

/// @brief Sets the element of the matrix.
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T.
template <class T, class Func>
void set(Matrix<T, Device::CPU>& mat, Func el) {
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

/// @brief Returns a col-major ordered vector with the future to the matrix tiles.
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

/// @brief Returns a col-major ordered vector with the read-only shared-future to the matrix tiles.
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
/// comp(el({i, j}), (i, j)-element) is used to compare the elements.
/// err_message(el({i, j}), (i, j)-element) is printed for the first element which does not fulfill the comparison.
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre comp should have two arguments and return true if the comparison is fulfilled and false otherwise.
/// @pre err_message should have two arguments and return a string.
/// @pre el return type should be the same as the type of the first argument of comp adn of err_message.
/// @pre The second argument of comp should be either T, T& or const T&.
/// @pre The second argument of err_message should be either T, T& or const T&.
namespace internal {
template <class T, class Func1, class Func2, class Func3>
void check(Matrix<T, Device::CPU>& mat, Func1 el, Func2 comp, Func3 err_message, const char* file,
           const int line) {
  for (SizeType tile_j = 0; tile_j < mat.nrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < mat.nrTiles().rows(); ++tile_i) {
      auto& tile = mat.read(LocalTileIndex(tile_i, tile_j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        SizeType j = tile_j * mat.blockSize().cols() + jj;
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
          SizeType i = tile_i * mat.blockSize().rows() + ii;
          if (!comp(el({i, j}), tile({ii, jj}))) {
            ADD_FAILURE_AT(file, line) << "Error at index (" << i << ", " << j
                                       << "): " << err_message(el({i, j}), tile({ii, jj})) << std::endl;
            return;
          }
        }
      }
    }
  }
}
}

/// @brief Checks the elements of the matrix (exact equality).
/// The (i, j)-element of the matrix is compared to el({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T.
template <class T, class Func>
void checkEQ(Matrix<T, Device::CPU>& mat, Func el, const char* file, const int line) {
  auto comp = [](T expected, T value) { return expected == value; };
  auto err_message = [](T expected, T value) {
    std::stringstream s;
    s << "expected " << expected << " == " << value;
    return s.str();
  };
  internal::check(mat, el, comp, err_message, file, line);
}
#define CHECK_MATRIX_EQ(el, mat) ::dlaf_test::matrix_test::checkEQ(mat, el, __FILE__, __LINE__);

/// @brief Checks the pointers to the elements of the matrix.
/// The ipointer to (i, j)-element of the matrix is compared to ptr({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex&.
/// @pre el return type should be T*.
template <class T, class Func>
void checkPtr(Matrix<T, Device::CPU>& mat, Func ptr, const char* file, const int line) {
  auto comp = [](T* ptr, const T& value) { return ptr == &value; };
  auto err_message = [](T* expected, const T& value) {
    std::stringstream s;
    s << "expected " << expected << " == " << &value;
    return s.str();
  };
  internal::check(mat, ptr, comp, err_message, file, line);
}
#define CHECK_MATRIX_PTR(ptr, mat) ::dlaf_test::matrix_test::checkPtr(mat, ptr, __FILE__, __LINE__);

}
}
