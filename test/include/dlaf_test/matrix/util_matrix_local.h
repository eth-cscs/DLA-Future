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

#include <gtest/gtest.h>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/copy_tile.h"

#include "dlaf_test/matrix/matrix_local.h"

namespace dlaf {
namespace matrix {
namespace test {

/// Sets the elements of the matrix.
///
/// The (i, j)-element of the matrix is set to el({i, j}).
/// @pre el argument is an index of type const GlobalElementIndex& or GlobalElementIndex,
/// @pre el return type should be T.
template <class T, class ElementGetter>
void set(const MatrixLocal<T>& matrix, ElementGetter el) {
  using dlaf::common::iterate_range2d;

  for (const auto& tile_index : iterate_range2d(matrix.size()))
    matrix(tile_index) = el(tile_index);
}

template <class T>
void copy(const MatrixLocal<const T>& source, MatrixLocal<T>& dest) {
  DLAF_ASSERT(source.size() == dest.size(), source.size(), dest.size());
  const auto linear_size = static_cast<std::size_t>(source.size().rows() * source.size().cols());
  std::copy(source.ptr(), source.ptr() + linear_size, dest.ptr());
}

template <class T>  // TODO add tile_selector predicate
void all_gather(Matrix<const T, Device::CPU>& source, MatrixLocal<T>& dest,
                comm::CommunicatorGrid comm_grid) {
  using namespace dlaf;
  const auto& dist_source = source.distribution();
  const auto rank = dist_source.rankIndex();
  for (const auto& ij_tile : iterate_range2d(dist_source.nrTiles())) {
    const auto owner = dist_source.rankGlobalTile(ij_tile);
    auto& dest_tile = dest.readwrite_tile(ij_tile);
    if (owner == rank) {
      const auto& source_tile = source.read(ij_tile).get();
      comm::sync::broadcast::send(comm_grid.fullCommunicator(), source_tile);
      copy(source_tile, dest_tile);
    }
    else {
      comm::sync::broadcast::receive_from(comm_grid.rankFullCommunicator(owner),
                                          comm_grid.fullCommunicator(), dest_tile);
    }
  }
}

template <class T>
void checkNear(const MatrixLocal<const T>& expected, const MatrixLocal<const T>& mat, BaseType<T> rel_err,
               BaseType<T> abs_err, const char* file, const int line) {
  ASSERT_GE(rel_err, 0);
  ASSERT_GE(abs_err, 0);
  ASSERT_TRUE(rel_err > 0 || abs_err > 0);

  DLAF_ASSERT(expected.size() == mat.size(), expected.size(), mat.size());

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

  //internal::check(expected, mat, comp, err_message, file, line);
  for (const auto& index : iterate_range2d(expected.size())) {
    if (!comp(*expected.ptr(index), *mat.ptr(index))) {
      ADD_FAILURE_AT(file, line)
        << "Error at index " << index
        << "): " << err_message(*expected.ptr(index), *mat.ptr(index)) << std::endl;
      return;
    }
  }
}

}
}
}
