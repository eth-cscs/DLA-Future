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

#include <sstream>
#include <vector>
#include "gtest/gtest.h"
#include "dlaf/matrix.h"

namespace dlaf {
namespace matrix {
namespace test {

/// @brief Returns a col-major ordered vector with the futures to the matrix tiles.
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::future<Tile<T, device>>> getFuturesUsingLocalIndex(MatrixType<T, device>& mat) {
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
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::future<Tile<T, device>>> getFuturesUsingGlobalIndex(MatrixType<T, device>& mat) {
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
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::shared_future<Tile<const T, device>>> getSharedFuturesUsingLocalIndex(
    MatrixType<T, device>& mat) {
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
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::shared_future<Tile<const T, device>>> getSharedFuturesUsingGlobalIndex(
    MatrixType<T, device>& mat) {
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

/// @brief Returns true if only the first @p futures are ready.
/// @pre Future should be a future or shared_future.
/// @pre 0 <= ready <= futures.size()
template <class Future>
bool checkFuturesStep(size_t ready, const std::vector<Future>& futures) {
  assert(ready >= 0);
  assert(ready <= futures.size());

  for (std::size_t index = 0; index < ready; ++index) {
    if (!futures[index].is_ready())
      return false;
  }
  for (std::size_t index = ready; index < futures.size(); ++index) {
    if (futures[index].is_ready())
      return false;
  }
  return true;
}

/// @brief Checks if current[i] depends correctly on previous[i].
/// If get_ready == true it checks if current[i] is ready after previous[i] is used.
/// If get_ready == false it checks if current[i] is not ready after previous[i] is used.
/// @pre Future[1,2] should be a future or shared_future
template <class Future1, class Future2>
void checkFutures(bool get_ready, const std::vector<Future1>& current, std::vector<Future2>& previous) {
  assert(current.size() == previous.size());

  for (std::size_t index = 0; index < current.size(); ++index) {
    EXPECT_TRUE(checkFuturesStep(get_ready ? index : 0, current));
    previous[index].get();
    previous[index] = {};
  }

  EXPECT_TRUE(checkFuturesStep(get_ready ? current.size() : 0, current));
}

#define CHECK_MATRIX_FUTURES(get_ready, current, previous)            \
  do {                                                                \
    SCOPED_TRACE("");                                                 \
    ::dlaf::matrix::test::checkFutures(get_ready, current, previous); \
  } while (0)

/// @brief Checks if current[i] depends correctly on mat_view.done(index),
///
/// where index = LocalTileIndex(i % mat_view.localNrTiles.rows(), i / mat_view.localNrTiles.rows())
/// If get_ready == true it checks if current[i] is ready after previous[i] is used.
/// If get_ready == false it checks if current[i] is not ready after previous[i] is used.
/// @pre Future1 should be a future or shared_future
template <class Future1, class MatrixViewType>
void checkFuturesDone(bool get_ready, const std::vector<Future1>& current, MatrixViewType& mat_view) {
  const auto& nr_tiles = mat_view.distribution().localNrTiles();
  assert(current.size() == util::size_t::mul(nr_tiles.rows(), nr_tiles.cols()));

  for (std::size_t index = 0; index < current.size(); ++index) {
    EXPECT_TRUE(checkFuturesStep(get_ready ? index : 0, current));
    LocalTileIndex tile_index(index % nr_tiles.rows(), index / nr_tiles.rows());
    mat_view.done(tile_index);
  }

  EXPECT_TRUE(checkFuturesStep(get_ready ? current.size() : 0, current));
}

#define CHECK_MATRIX_FUTURES_DONE(get_ready, current, mat_view)           \
  do {                                                                    \
    SCOPED_TRACE("");                                                     \
    ::dlaf::matrix::test::checkFuturesDone(get_ready, current, mat_view); \
  } while (0)
}
}
}
