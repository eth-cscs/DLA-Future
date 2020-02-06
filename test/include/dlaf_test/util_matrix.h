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

/// @file

#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "gtest/gtest.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
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
  const matrix::Distribution& dist = mat.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      auto tile = mat(LocalTileIndex(tile_i, tile_j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        SizeType j = dist.globalElementFromLocalTileAndTileElement<Coord::Col>(tile_j, jj);
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
          SizeType i = dist.globalElementFromLocalTileAndTileElement<Coord::Row>(tile_i, ii);
          tile({ii, jj}) = el({i, j});
        }
      }
    }
  }
}

/// @brief Returns a col-major ordered vector with the futures to the matrix tiles.
template <class T, Device device>
std::vector<hpx::future<Tile<T, device>>> getFuturesUsingLocalIndex(Matrix<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::future<Tile<T, device>>> result;
  result.reserve(util::size_t::mul(dist.localNrTiles().rows(), dist.localNrTiles().cols()));

  for (SizeType j = 0; j < dist.localNrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.localNrTiles().rows(); ++i) {
      result.emplace_back(std::move(mat(LocalTileIndex(i, j))));
      EXPECT_TRUE(result.back().valid());
    }
  }
  return result;
}

/// @brief Returns a col-major ordered vector with the futures to the matrix tiles.
template <class T, Device device>
std::vector<hpx::future<Tile<T, device>>> getFuturesUsingGlobalIndex(Matrix<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::future<Tile<T, device>>> result;
  result.reserve(util::size_t::mul(dist.localNrTiles().rows(), dist.localNrTiles().cols()));

  for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
      GlobalTileIndex global_index{i, j};
      comm::Index2D owner = dist.rankGlobalTile(global_index);

      if (dist.rankIndex() == owner) {
        result.emplace_back(std::move(mat(global_index)));
        EXPECT_TRUE(result.back().valid());
      }
    }
  }
  return result;
}

/// @brief Returns a col-major ordered vector with the read-only shared-futures to the matrix tiles.
template <class T, Device device>
std::vector<hpx::shared_future<Tile<const T, device>>> getSharedFuturesUsingLocal(
    Matrix<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::shared_future<Tile<const T, device>>> result;
  result.reserve(util::size_t::mul(dist.localNrTiles().rows(), dist.localNrTiles().cols()));

  for (SizeType j = 0; j < dist.localNrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.localNrTiles().rows(); ++i) {
      result.emplace_back(mat.read(LocalTileIndex(i, j)));
      EXPECT_TRUE(result.back().valid());
    }
  }

  return result;
}

/// @brief Returns a col-major ordered vector with the read-only shared-futures to the matrix tiles.
template <class T, Device device>
std::vector<hpx::shared_future<Tile<const T, device>>> getSharedFuturesUsingGlobal(
    Matrix<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::shared_future<Tile<const T, device>>> result;
  result.reserve(util::size_t::mul(dist.localNrTiles().rows(), dist.localNrTiles().cols()));

  for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
      GlobalTileIndex global_index{i, j};
      comm::Index2D owner = dist.rankGlobalTile(global_index);

      if (dist.rankIndex() == owner) {
        result.emplace_back(mat.read(global_index));
        EXPECT_TRUE(result.back().valid());
      }
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
void check(ElementGetter expected, Matrix<T, Device::CPU>& mat, ComparisonOp comp,
           ErrorMessageGetter err_message, const char* file, const int line) {
  const matrix::Distribution& dist = mat.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      auto& tile = mat.read(LocalTileIndex(tile_i, tile_j)).get();
      for (SizeType jj = 0; jj < tile.size().cols(); ++jj) {
        SizeType j = dist.globalElementFromLocalTileAndTileElement<Coord::Col>(tile_j, jj);
        for (SizeType ii = 0; ii < tile.size().rows(); ++ii) {
          SizeType i = dist.globalElementFromLocalTileAndTileElement<Coord::Row>(tile_i, ii);
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
void checkEQ(ElementGetter exp_el, Matrix<T, Device::CPU>& mat, const char* file, const int line) {
  auto err_message = [](T expected, T value) {
    std::stringstream s;
    s << "expected " << expected << " == " << value;
    return s.str();
  };
  internal::check(exp_el, mat, std::equal_to<T>{}, err_message, file, line);
}
#define CHECK_MATRIX_EQ(exp_el, mat) ::dlaf_test::matrix_test::checkEQ(exp_el, mat, __FILE__, __LINE__);

/// @brief Checks the pointers to the elements of the matrix.
///
/// The pointer to (i, j)-element of the matrix is compared to exp_ptr({i, j}).
/// @pre exp_ptr argument is an index of type const GlobalElementIndex&.
/// @pre exp_ptr return type should be T*.
template <class T, class PointerGetter>
void checkPtr(PointerGetter exp_ptr, Matrix<T, Device::CPU>& mat, const char* file, const int line) {
  auto comp = [](T* ptr, const T& value) { return ptr == &value; };
  auto err_message = [](T* expected, const T& value) {
    std::stringstream s;
    s << "expected " << expected << " == " << &value;
    return s.str();
  };
  internal::check(exp_ptr, mat, comp, err_message, file, line);
}
#define CHECK_MATRIX_PTR(exp_ptr, mat) \
  ::dlaf_test::matrix_test::checkPtr(exp_ptr, mat, __FILE__, __LINE__);

/// @brief Checks the elements of the matrix.
///
/// The (i, j)-element of the matrix is compared to expected({i, j}).
/// @pre expected argument is an index of type const GlobalElementIndex&.
/// @pre expected return type should be T.
/// @pre rel_err > 0.
/// @pre abs_err > 0.
template <class T, class ElementGetter>
void checkNear(ElementGetter expected, Matrix<T, Device::CPU>& mat, BaseType<T> rel_err,
               BaseType<T> abs_err, const char* file, const int line) {
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
  internal::check(expected, mat, comp, err_message, file, line);
}
#define CHECK_MATRIX_NEAR(expected, matrix, rel_err, abs_err) \
  ::dlaf_test::matrix_test::checkNear(expected, matrix, rel_err, abs_err, __FILE__, __LINE__);

}
}
