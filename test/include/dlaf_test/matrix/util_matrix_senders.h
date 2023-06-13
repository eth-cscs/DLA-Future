//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <sstream>
#include <vector>

#include <pika/execution.hpp>

#include <dlaf/matrix/matrix.h>

#include <gtest/gtest.h>

#include <dlaf_test/matrix/util_tile.h>

namespace dlaf::matrix::test {

/// Returns a col-major ordered vector with wrappers of senders to matrix tiles.
///
/// The senders are created using the matrix method readwrite(const
/// LocalTileIndex&). Note: This function is interchangeable with
/// getReadWriteSendersUsingGlobalIndex.
template <template <class, Device> class MatrixType, class T, Device D>
std::vector<EagerVoidSender> getReadWriteSendersUsingLocalIndex(MatrixType<T, D>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<EagerVoidSender> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

  for (SizeType j = 0; j < dist.localNrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.localNrTiles().rows(); ++i) {
      result.emplace_back(mat.readwrite(LocalTileIndex(i, j)));
    }
  }
  return result;
}

/// Returns a col-major ordered vector with senders to the matrix tiles.
///
/// The senders are created using the matrix method readwrite(const
/// GlobalTileIndex&).  Note: This function is interchangeable with
/// getReadWriteSendersUsingLocalIndex.
template <template <class, Device> class MatrixType, class T, Device D>
std::vector<EagerVoidSender> getReadWriteSendersUsingGlobalIndex(MatrixType<T, D>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<EagerVoidSender> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

  for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
      GlobalTileIndex global_index{i, j};
      comm::Index2D owner = dist.rankGlobalTile(global_index);

      if (dist.rankIndex() == owner) {
        result.emplace_back(mat.readwrite(global_index));
      }
    }
  }
  return result;
}

/// Returns a col-major ordered vector with read-only senders to the matrix tiles.
///
/// The senders are created using the matrix method read(const LocalTileIndex&).
/// Note: This function is interchangeable with getReadSendersUsingGlobalIndex.
template <template <class, Device> class MatrixType, class T, Device D>
std::vector<EagerVoidSender> getReadSendersUsingLocalIndex(MatrixType<T, D>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<EagerVoidSender> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

  for (SizeType j = 0; j < dist.localNrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.localNrTiles().rows(); ++i) {
      result.emplace_back(mat.read(LocalTileIndex(i, j)));
    }
  }

  return result;
}

/// Returns a col-major ordered vector with read-only senders to the matrix tiles.
///
/// The senders are created using the matrix method read(const GlobalTileIndex&).
/// Note: This function is interchangeable with getReadSendersUsingLocalIndex.
template <template <class, Device> class MatrixType, class T, Device D>
std::vector<EagerVoidSender> getReadSendersUsingGlobalIndex(MatrixType<T, D>& mat) {
  const matrix::Distribution& dist = mat.distribution();

  std::vector<EagerVoidSender> result;
  result.reserve(static_cast<std::size_t>(dist.localNrTiles().linear_size()));

  for (SizeType j = 0; j < dist.nrTiles().cols(); ++j) {
    for (SizeType i = 0; i < dist.nrTiles().rows(); ++i) {
      GlobalTileIndex global_index{i, j};
      comm::Index2D owner = dist.rankGlobalTile(global_index);

      if (dist.rankIndex() == owner) {
        result.emplace_back(mat.read(global_index));
      }
    }
  }

  return result;
}

/// Returns true if only the @p first_n senders are ready (or the opposite).
///
/// @param invert if set to true it checks that all senders are ready except the first_n
///
/// @pre 0 <= ready <= senders.size().
inline bool checkSendersStep(size_t first_n, const std::vector<EagerVoidSender>& senders,
                             bool invert = false) {
  DLAF_ASSERT_HEAVY(first_n <= senders.size(), first_n, senders.size());

  const bool first_n_status = !invert;

  for (std::size_t index = 0; index < first_n; ++index) {
    if (senders[index].is_ready() != first_n_status)
      return false;
  }
  for (std::size_t index = first_n; index < senders.size(); ++index) {
    if (senders[index].is_ready() == first_n_status)
      return false;
  }
  return true;
}

/// Checks if current[i] depends correctly on previous[i].
///
/// If get_ready == true it checks if current[i] is ready after previous[i] is used.
/// If get_ready == false it checks if current[i] is not ready after previous[i] is used.
void checkSenders(bool get_ready, const std::vector<EagerVoidSender>& current,
                  std::vector<EagerVoidSender>& previous) {
  DLAF_ASSERT_HEAVY(current.size() == previous.size(), current.size(), previous.size());

  for (std::size_t index = 0; index < current.size(); ++index) {
    EXPECT_TRUE(checkSendersStep(get_ready ? index : 0, current));
    std::move(previous[index]).get();
  }

  EXPECT_TRUE(checkSendersStep(get_ready ? current.size() : 0, current));
}

#define CHECK_MATRIX_SENDERS(get_ready, current, previous)            \
  do {                                                                \
    SCOPED_TRACE("");                                                 \
    ::dlaf::matrix::test::checkSenders(get_ready, current, previous); \
  } while (0)
}
