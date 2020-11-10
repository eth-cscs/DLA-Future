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
void set(MatrixLocal<T>& matrix, ElementGetter el) {
  using dlaf::common::iterate_range2d;

  for (const auto& index : iterate_range2d(matrix.size()))
    *matrix.ptr(index) = el(index);
}

template <class T>  // TODO add tile_selector predicate
void all_gather(Matrix<const T, Device::CPU>& source, MatrixLocal<T>& dest,
                comm::CommunicatorGrid comm_grid) {
  using namespace dlaf;
  const auto& dist_source = source.distribution();
  const auto rank = dist_source.rankIndex();
  for (const auto& ij_tile : iterate_range2d(dist_source.nrTiles())) {
    const auto owner = dist_source.rankGlobalTile(ij_tile);
    auto& dest_tile = dest(ij_tile);
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

}
}
}
