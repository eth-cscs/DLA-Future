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

#include <hpx/include/util.hpp>
#include <hpx/local/future.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace internal {
namespace mc {

// Compute max norm of the lower triangular part of the distributed matrix
// https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
//
// It just addresses tiles with elements belonging to the lower triangular part of the matrix
//
// Thanks to the nature of the max norm, it is valid for:
// - sy/he lower
// - tr lower non-unit
template <class T>
dlaf::BaseType<T> norm_max_L(comm::CommunicatorGrid comm_grid, comm::Index2D rank,
                             Matrix<const T, Device::CPU>& matrix) {
  using namespace dlaf::matrix;

  using dlaf::common::internal::vector;
  using dlaf::common::make_data;
  using hpx::util::unwrapping;

  using dlaf::tile::lange;
  using dlaf::tile::lantr;

  using NormT = dlaf::BaseType<T>;

  const auto& distribution = matrix.distribution();

  DLAF_ASSERT(square_size(matrix), matrix);
  DLAF_ASSERT(square_blocksize(matrix), matrix);

  vector<hpx::future<NormT>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each local tile in the (global) lower triangular matrix, create a task that finds the max element in the tile
  for (auto tile_wrt_local : iterate_range2d(distribution.localNrTiles())) {
    auto tile_wrt_global = distribution.globalTileIndex(tile_wrt_local);

    if (tile_wrt_global.row() < tile_wrt_global.col())
      continue;

    bool is_diag = tile_wrt_global.row() == tile_wrt_global.col();
    auto norm_max_f = unwrapping([is_diag](auto&& tile) noexcept->NormT {
      if (is_diag)
        return lantr(lapack::Norm::Max, blas::Uplo::Lower, blas::Diag::NonUnit, tile);
      else
        return lange(lapack::Norm::Max, tile);
    });
    auto current_tile_max = hpx::dataflow(norm_max_f, matrix.read(tile_wrt_local));

    tiles_max.emplace_back(std::move(current_tile_max));
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
  NormT max_value;
  dlaf::comm::sync::reduce(comm_grid.rankFullCommunicator(rank), comm_grid.fullCommunicator(), MPI_MAX,
                           make_data(&local_max_value, 1), make_data(&max_value, 1));

  return max_value;
}

}
}
}
