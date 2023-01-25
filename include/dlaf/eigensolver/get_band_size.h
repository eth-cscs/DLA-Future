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

// Returns the smallest divisor of nb larger than b_min = getTuneParameters().eigensolver_min_band.
// If nb is smaller than b_min returns nb.
inline SizeType getBandSize(const SizeType nb) noexcept {
  const SizeType b_min = getTuneParameters().eigensolver_min_band;

  DLAF_ASSERT(nb >= 1, nb);
  DLAF_ASSERT(b_min >= 2, b_min);

  for (SizeType div = nb / b_min; div > 1; --div) {
    if (nb % div == 0)
      return nb / div;
  }
  return nb;
}

}
