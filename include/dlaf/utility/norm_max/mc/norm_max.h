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
#include <lapack.hh>

#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/types.h"

namespace dlaf {
namespace internal {
namespace mc {

// Distributed implementation of Max Norm of a Lower Triangular Matrix
template <class T>
dlaf::BaseType<T> norm_max(comm::CommunicatorGrid comm_grid, Matrix<const T, Device::CPU>& matrix) {
  using dlaf::common::internal::vector;
  using dlaf::common::make_data;
  using hpx::util::unwrapping;

  using NormT = dlaf::BaseType<T>;

  const auto& distribution = matrix.distribution();

  vector<hpx::future<NormT>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each tile in local (lower triangular), create a task that finds the max element in the tile
  for (SizeType j_loc = 0; j_loc < distribution.localNrTiles().cols(); ++j_loc) {
    const SizeType j = distribution.template globalTileFromLocalTile<Coord::Col>(j_loc);
    const SizeType i_diag_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(j);

    for (SizeType i_loc = i_diag_loc; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
      auto current_tile_max =
          hpx::dataflow(unwrapping([is_diag = (i_loc == i_diag_loc)](auto&& tile) -> NormT {
                          if (is_diag)
                            return lapack::lantr(lapack::Norm::Max, lapack::Uplo::Lower,
                                                 lapack::Diag::NonUnit, tile.size().cols(),
                                                 tile.size().rows(), tile.ptr(), tile.ld());
                          else
                            return lapack::lange(lapack::Norm::Max, tile.size().cols(),
                                                 tile.size().rows(), tile.ptr(), tile.ld());
                        }),
                        matrix.read(LocalTileIndex{i_loc, j_loc}));

      tiles_max.emplace_back(std::move(current_tile_max));
    }
  }

  // than it is necessary to reduce max values from all ranks into a single max value for the matrix

  // TODO unwrapping can be skipped for optimization reasons
  NormT local_max_value = hpx::dataflow(unwrapping([](const auto&& values) {
                                          if (values.size() == 0)
                                            return std::numeric_limits<NormT>::min();
                                          return *std::max_element(values.begin(), values.end());
                                        }),
                                        tiles_max)
                              .get();

  NormT max_value_rows;
  if (comm_grid.size().cols() > 1)
    dlaf::comm::sync::reduce(0, comm_grid.rowCommunicator(), MPI_MAX, make_data(&local_max_value, 1),
                             make_data(&max_value_rows, 1));
  else
    max_value_rows = local_max_value;

  NormT max_value;
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
