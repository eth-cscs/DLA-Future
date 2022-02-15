//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/auxiliary/norm/api.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace auxiliary {
namespace internal {

template <class T>
struct Norm<Backend::MC, Device::CPU, T> {
  static dlaf::BaseType<T> max_L(comm::CommunicatorGrid comm_grid, comm::Index2D rank,
                                 Matrix<const T, Device::CPU>& matrix);
};

// Compute max norm of the lower triangular part of the distributed matrix
// https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
//
// It just addresses tiles with elements belonging to the lower triangular part of the matrix
//
// Thanks to the nature of the max norm, it is valid for:
// - sy/he lower
// - tr lower non-unit
template <class T>
dlaf::BaseType<T> Norm<Backend::MC, Device::CPU, T>::max_L(comm::CommunicatorGrid comm_grid,
                                                           comm::Index2D rank,
                                                           Matrix<const T, Device::CPU>& matrix) {
  using namespace dlaf::matrix;
  namespace ex = pika::execution::experimental;

  using dlaf::common::internal::vector;
  using dlaf::common::make_data;
  using pika::unwrapping;

  using dlaf::tile::internal::lange;
  using dlaf::tile::internal::lantr;

  using NormT = dlaf::BaseType<T>;

  const auto& distribution = matrix.distribution();

  DLAF_ASSERT(square_size(matrix), matrix);
  DLAF_ASSERT(square_blocksize(matrix), matrix);

  vector<pika::future<NormT>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each local tile in the (global) lower triangular matrix, create a task that finds the max element in the tile
  for (auto tile_wrt_local : iterate_range2d(distribution.localNrTiles())) {
    auto tile_wrt_global = distribution.globalTileIndex(tile_wrt_local);

    if (tile_wrt_global.row() < tile_wrt_global.col())
      continue;

    bool is_diag = tile_wrt_global.row() == tile_wrt_global.col();
    auto norm_max_f = unwrapping([is_diag](auto&& tile) noexcept -> NormT {
      if (is_diag)
        return lantr(lapack::Norm::Max, blas::Uplo::Lower, blas::Diag::NonUnit, tile);
      else
        return lange(lapack::Norm::Max, tile);
    });
    auto current_tile_max =
        dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), std::move(norm_max_f),
                                  matrix.read_sender(tile_wrt_local)) |
        ex::make_future();

    tiles_max.emplace_back(std::move(current_tile_max));
  }

  // then it is necessary to reduce max values from all ranks into a single max value for the matrix

  auto max_element = [](std::vector<NormT>&& values) {
    DLAF_ASSERT(!values.empty(), "");
    return *std::max_element(values.begin(), values.end());
  };
  NormT local_max_value = tiles_max.empty()
                              ? std::numeric_limits<NormT>::min()
                              : pika::execution::experimental::when_all_vector(std::move(tiles_max)) |
                                    dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(),
                                                              std::move(max_element)) |
                                    pika::execution::experimental::sync_wait();
  NormT max_value;
  dlaf::comm::sync::reduce(comm_grid.rankFullCommunicator(rank), comm_grid.fullCommunicator(), MPI_MAX,
                           make_data(&local_max_value, 1), make_data(&max_value, 1));

  return max_value;
}

/// ---- ETI
#define DLAF_NORM_ETI(KWORD, DATATYPE) KWORD template struct Norm<Backend::MC, Device::CPU, DATATYPE>;

DLAF_NORM_ETI(extern, float)
DLAF_NORM_ETI(extern, double)
DLAF_NORM_ETI(extern, std::complex<float>)
DLAF_NORM_ETI(extern, std::complex<double>)

}
}
}
