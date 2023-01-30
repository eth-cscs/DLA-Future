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

#include "dlaf/common/assert.h"
#include "dlaf/tune.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

// Returns max(1, getTuneParameters().eigensolver_min_band / nb * nb).
inline SizeType get1DBlockSize(const SizeType nb) noexcept {
  const SizeType nb_base = getTuneParameters().band_to_tridiag_1d_block_size_base;

  DLAF_ASSERT(nb >= 1, nb);
  DLAF_ASSERT(nb_base >= 1, nb_base);

  return std::max<SizeType>(1, nb_base / nb * nb);
}

}
