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

#include <hpx/hpx.hpp>
#include <hpx/util/unwrap.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/common/vector.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/communication/sync/reduce.h"

namespace dlaf {
namespace internal {
namespace mc {

// Distributed implementation of Max Norm of a Lower Triangular Matrix
template <class T>
T norm_max(comm::CommunicatorGrid comm_grid, Matrix<const T, Device::CPU>& matrix) {
  using dlaf::common::internal::vector;
  using dlaf::common::make_data;
  using hpx::util::unwrapping;

  // TODO unwrapping can be skipped for optimization reasons
  auto max_f = unwrapping([](const auto&& values) {
    // some rank may not have any element
    if (values.size() == 0)
      return std::numeric_limits<T>::min();
    return *std::max_element(values.begin(), values.end());
  });

  const auto& distribution = matrix.distribution();

  vector<hpx::future<T>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each tile in local (lower triangular), create a task that finds the max element in the tile
  for (SizeType j_loc = 0; j_loc < distribution.localNrTiles().cols(); ++j_loc) {
    const SizeType j = distribution.template globalTileFromLocalTile<Coord::Col>(j_loc);
    const SizeType i_diag_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(j);

    for (SizeType i_loc = i_diag_loc; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
      auto current_tile_max =
          hpx::dataflow(unwrapping([is_diag = (i_loc == i_diag_loc)](auto&& tile) -> T {
                          T tile_max_value = std::abs(T{0});
                          for (SizeType j = 0; j < tile.size().cols(); ++j)
                            for (SizeType i = is_diag ? j : 0; i < tile.size().rows(); ++i)
                              tile_max_value = std::max(tile_max_value, tile({i, j}));
                          return tile_max_value;
                        }),
                        matrix.read(LocalTileIndex{i_loc, j_loc}));

      tiles_max.emplace_back(std::move(current_tile_max));
    }
  }

  // than it is necessary to reduce max values from all ranks into a single max value for the matrix

  auto local_max_value = hpx::dataflow(max_f, tiles_max).get();

  T max_value_rows;
  if (comm_grid.size().cols() > 1)
    dlaf::comm::sync::reduce(0, comm_grid.rowCommunicator(), MPI_MAX, make_data(&local_max_value, 1),
                             make_data(&max_value_rows, 1));
  else
    max_value_rows = local_max_value;

  T max_value;
  if (comm_grid.size().rows() > 1)
    dlaf::comm::sync::reduce(0, comm_grid.colCommunicator(), MPI_MAX, make_data(&max_value_rows, 1),
                             make_data(&max_value, 1));
  else
    max_value = max_value_rows;

  return max_value;
}

}
}
}
