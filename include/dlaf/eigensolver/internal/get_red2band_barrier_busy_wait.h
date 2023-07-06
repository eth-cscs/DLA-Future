//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <chrono>

#include "dlaf/tune.h"

namespace dlaf::eigensolver::internal {

inline std::chrono::duration<double> getReductionToBandBarrierBusyWait() noexcept {
  return std::chrono::microseconds(getTuneParameters().red2band_barrier_busy_wait_us);
}

}
