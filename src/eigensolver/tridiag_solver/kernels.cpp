//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/kernels.h"

namespace dlaf::eigensolver::internal {

template <class T>
T cuppensDecomp(const matrix::Tile<T, Device::CPU>& top, const matrix::Tile<T, Device::CPU>& bottom) {
  TileElementIndex offdiag_idx{top.size().rows() - 1, 1};
  TileElementIndex top_idx{top.size().rows() - 1, 0};
  TileElementIndex bottom_idx{0, 0};
  const T offdiag_val = top(offdiag_idx);

  // Refence: Lapack working notes: LAWN 69, Serial Cuppen algorithm, Chapter 3
  //
  top(top_idx) -= std::abs(offdiag_val);
  bottom(bottom_idx) -= std::abs(offdiag_val);
  return offdiag_val;
}

DLAF_CPU_CUPPENS_DECOMP_ETI(, float);
DLAF_CPU_CUPPENS_DECOMP_ETI(, double);

}
