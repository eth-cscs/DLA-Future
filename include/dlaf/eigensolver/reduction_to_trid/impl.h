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

#include <dlaf/common/index2d.h>
#include <dlaf/eigensolver/reduction_to_trid/api.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
Matrix<T, Device::CPU> ReductionToTrid<B, D, T>::call(Matrix<T, D>& mat_a) {
  using dlaf::matrix::Matrix;

  using common::iterate_range2d;

  const auto dist_a = mat_a.distribution();

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real nor complex)
  const SizeType nrefls = std::max<SizeType>(0, dist_a.size().rows() - 1 - 1);

  // Row-vector that is distributed over columns, but exists locally on all rows of the grid
  Matrix<T, Device::CPU> mat_taus(matrix::Distribution(GlobalElementSize(nrefls, 1),
                                                       TileElementSize(mat_a.blockSize().cols(), 1),
                                                       comm::Size2D(mat_a.commGridSize().cols(), 1),
                                                       comm::Index2D(mat_a.rankIndex().col(), 0),
                                                       comm::Index2D(mat_a.sourceRankIndex().col(), 0)));

  return mat_taus;
}

}
