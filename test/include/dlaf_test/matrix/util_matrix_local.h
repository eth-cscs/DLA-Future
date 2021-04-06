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
#include <type_traits>

#include <gtest/gtest.h>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/print_numpy.h"

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

/// Given a (possibly) distributed Matrix, collect all data full-size local matrix
///
/// Optionally, it is possible to specify the type of the return MatrixLocal (useful for const correctness)
template <class T>
MatrixLocal<T> allGather(Matrix<const T, Device::CPU>& source, comm::CommunicatorGrid comm_grid) {
  MatrixLocal<std::remove_const_t<T>> dest(source.size(), source.blockSize());

  const auto& dist_source = source.distribution();
  const auto rank = dist_source.rankIndex();

  for (const auto& ij_tile : iterate_range2d(source.nrTiles())) {
    const auto owner = dist_source.rankGlobalTile(ij_tile);
    auto& dest_tile = dest.tile(ij_tile);

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

  return MatrixLocal<T>(std::move(dest));
}

template <class T>
void print(format::numpy, std::string symbol, const MatrixLocal<const T>& matrix,
           std::ostream& os = std::cout) {
  using common::iterate_range2d;

  os << symbol << " = np.zeros(" << matrix.size() << ", dtype=np."
     << dlaf::matrix::internal::numpy_datatype<T>::typestring << ")\n";

  auto getTileTopLeft = [&](const GlobalTileIndex& ij) -> GlobalElementIndex {
    return {ij.row() * matrix.blockSize().rows(), ij.col() * matrix.blockSize().cols()};
  };

  for (const auto& ij : iterate_range2d(matrix.nrTiles())) {
    const auto& tile = matrix.tile_read(ij);

    const auto index_tl = getTileTopLeft(ij);

    os << symbol << "[" << index_tl.row() << ":" << index_tl.row() + tile.size().rows() << ","
       << index_tl.col() << ":" << index_tl.col() + tile.size().cols() << "] = ";

    print(format::numpy{}, tile, os);
  }
}
}
}
}
