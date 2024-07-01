//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <pika/execution.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/eigensolver/reduction_to_band/api.h>
#include <dlaf/matrix/matrix.h>

namespace dlaf::eigensolver::internal {

namespace ca_red2band {}

// Distributed implementation of reduction to band
template <Backend B, Device D, class T>
CARed2BandResult<T, D> CAReductionToBand<B, D, T>::call(comm::CommunicatorGrid& grid,
                                                        Matrix<T, D>& mat_a, const SizeType band_size) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rank_index();

  const SizeType nrefls = std::max<SizeType>(0, dist.size().cols() - band_size - 1);

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real
  // nor complex)
  DLAF_ASSERT(dist.block_size().cols() % band_size == 0, dist.block_size().cols(), band_size);

  // Note:
  // row-vector that is distributed over columns, but replicated over rows.
  // for historical reason it is stored and accessed as a column-vector.
  DLAF_ASSERT(dist.block_size().cols() % band_size == 0, dist.block_size().cols(), band_size);
  const matrix::Distribution dist_taus(GlobalElementSize(nrefls, 1),
                                       TileElementSize(dist.block_size().cols(), 1),
                                       comm::Size2D(dist.grid_size().cols(), 1),
                                       comm::Index2D(rank.col(), 0),
                                       comm::Index2D(dist.source_rank_index().col(), 0));
  Matrix<T, Device::CPU> mat_taus_1st(dist_taus);
  Matrix<T, Device::CPU> mat_taus_2nd(dist_taus);

  // Note:
  // row-panel distributed over columns, but replicated over rows
  const matrix::Distribution dist_hh_2nd(GlobalElementSize(dist.block_size().rows(), dist.size().cols()),
                                         dist.block_size(), comm::Size2D(1, dist.grid_size().cols()),
                                         comm::Index2D(0, rank.col()),
                                         comm::Index2D(0, dist.source_rank_index().col()));
  Matrix<T, D> mat_hh_2nd(dist_hh_2nd);

  if (nrefls == 0)
    return {std::move(mat_taus_1st), std::move(mat_taus_2nd), std::move(mat_hh_2nd)};

  return {std::move(mat_taus_1st), std::move(mat_taus_2nd), std::move(mat_hh_2nd)};
}
}  // namespace dlaf::eigensolver::internal
