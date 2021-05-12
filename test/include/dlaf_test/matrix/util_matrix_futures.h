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

#include <sstream>
#include <vector>

#include <hpx/local/future.hpp>

#include "gtest/gtest.h"
#include "dlaf/matrix/matrix.h"

namespace dlaf {
namespace matrix {
namespace test {

/// Returns a col-major ordered vector with the futures to the matrix tiles.
///
/// The futures are created using the matrix method operator()(const LocalTileIndex&).
/// Note: This function is interchangeable with getFuturesUsingGlobalIndex.
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::future<Tile<T, device>>> getFuturesUsingLocalIndex(MatrixType<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::future<Tile<T, device>>> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

  for (SizeType j = 0; j < dist.localNrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.localNrTiles().rows(); ++i) {
      result.emplace_back(std::move(mat(LocalTileIndex(i, j))));
      EXPECT_TRUE(result.back().valid());
    }
  }
  return result;
}

/// Returns a col-major ordered vector with the futures to the matrix tiles.
///
/// The futures are created using the matrix method operator()(const GlobalTileIndex&).
/// Note: This function is interchangeable with getFuturesUsingLocalIndex.
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::future<Tile<T, device>>> getFuturesUsingGlobalIndex(MatrixType<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::future<Tile<T, device>>> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

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

/// Returns a col-major ordered vector with the read-only shared-futures to the matrix tiles.
///
/// The futures are created using the matrix method read(const LocalTileIndex&).
/// Note: This function is interchangeable with getSharedFuturesUsingGlobalIndex.
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::shared_future<Tile<const T, device>>> getSharedFuturesUsingLocalIndex(
    MatrixType<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::shared_future<Tile<const T, device>>> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

  for (SizeType j = 0; j < dist.localNrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.localNrTiles().rows(); ++i) {
      result.emplace_back(mat.read(LocalTileIndex(i, j)));
      EXPECT_TRUE(result.back().valid());
    }
  }

  return result;
}

/// Returns a col-major ordered vector with the read-only shared-futures to the matrix tiles.
///
/// The futures are created using the matrix method read(const GlobalTileIndex&).
/// Note: This function is interchangeable with getSharedFuturesUsingLocalIndex.
template <template <class, Device> class MatrixType, class T, Device device>
std::vector<hpx::shared_future<Tile<const T, device>>> getSharedFuturesUsingGlobalIndex(
    MatrixType<T, device>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<hpx::shared_future<Tile<const T, device>>> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

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

/// Returns true if only the @p first_n futures are ready (or the opposite).
///
/// @param invert if set to true it checks that all futures are ready except the first_n
///
/// @pre Future should be a future or shared_future,
/// @pre 0 <= ready <= futures.size().
template <class Future>
bool checkFuturesStep(size_t first_n, const std::vector<Future>& futures, bool invert = false) {
  DLAF_ASSERT_HEAVY(first_n <= futures.size(), first_n, futures.size());

  const bool first_n_status = !invert;

  for (std::size_t index = 0; index < first_n; ++index) {
    if (futures[index].is_ready() != first_n_status)
      return false;
  }
  for (std::size_t index = first_n; index < futures.size(); ++index) {
    if (futures[index].is_ready() == first_n_status)
      return false;
  }
  return true;
}

/// Checks if current[i] depends correctly on previous[i].
///
/// If get_ready == true it checks if current[i] is ready after previous[i] is used.
/// If get_ready == false it checks if current[i] is not ready after previous[i] is used.
/// @pre Future[1,2] should be a future or shared_future.
template <class Future1, class Future2>
void checkFutures(bool get_ready, const std::vector<Future1>& current, std::vector<Future2>& previous) {
  DLAF_ASSERT_HEAVY(current.size() == previous.size(), current.size(), previous.size());

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

/// Checks if current[i] depends correctly on mat_view.done(index),
///
/// where index = LocalTileIndex(i % mat_view.localNrTiles.rows(), i / mat_view.localNrTiles.rows())
/// If get_ready == true it checks if current[i] is ready after the call to mat_view.done(i).
/// If get_ready == false it checks if current[i] is not ready after the call to mat_view.done(i).
/// @pre Future should be a future or shared_future.
template <class Future, class MatrixViewType>
void checkFuturesDone(bool get_ready, const std::vector<Future>& current, MatrixViewType& mat_view) {
  const auto& nr_tiles = mat_view.distribution().localNrTiles();
  DLAF_ASSERT(static_cast<SizeType>(current.size()) == nr_tiles.linear_size(), current.size(),
              nr_tiles.linear_size());

  for (std::size_t index = 0; index < current.size(); ++index) {
    EXPECT_TRUE(checkFuturesStep(get_ready ? index : 0, current));
    LocalTileIndex tile_index(to_signed<LocalTileIndex::IndexType>(index) % nr_tiles.rows(),
                              to_signed<LocalTileIndex::IndexType>(index) / nr_tiles.rows());
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
