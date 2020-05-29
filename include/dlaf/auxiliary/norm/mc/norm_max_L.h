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

#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/types.h"

namespace dlaf {
namespace internal {
namespace mc {

// Compute max norm of the lower triangular part of the distributed matrix
// https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
//
// It just addresses tiles with elements belonging to the lower triangular part of the matrix
//
// Thanks to the nature of the max norm, it is valid for:
// - ge/sy/he lower
// - tr lower non-unit
template <class T>
dlaf::BaseType<T> norm_max_L(comm::CommunicatorGrid comm_grid, Matrix<const T, Device::CPU>& matrix) {
  using dlaf::common::internal::vector;
  using dlaf::common::make_data;
  using hpx::util::unwrapping;

  using dlaf::tile::lange;
  using dlaf::tile::lantr;

  using NormT = dlaf::BaseType<T>;

  const auto& distribution = matrix.distribution();

  DLAF_ASSERT_SIZE_SQUARE(matrix);
  DLAF_ASSERT_BLOCKSIZE_SQUARE(matrix);

  if (GlobalElementSize{0, 0} == matrix.size())
    return {0};

  vector<hpx::future<NormT>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each local tile in the (global) lower triangular matrix, create a task that finds the max element in the tile
  for (SizeType j_loc = 0; j_loc < distribution.localNrTiles().cols(); ++j_loc) {
    const SizeType j = distribution.template globalTileFromLocalTile<Coord::Col>(j_loc);
    const SizeType i_diag_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(j);

    for (SizeType i_loc = i_diag_loc; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
      const SizeType i = distribution.template globalTileFromLocalTile<Coord::Row>(i_loc);

      auto norm_max_f = unwrapping([is_diag = (j == i)](auto&& tile) noexcept->NormT {
        if (is_diag)
          return lantr(lapack::Norm::Max, blas::Uplo::Lower, blas::Diag::NonUnit, tile);
        else
          return lange(lapack::Norm::Max, tile);
      });
      auto current_tile_max = hpx::dataflow(norm_max_f, matrix.read(LocalTileIndex{i_loc, j_loc}));

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
