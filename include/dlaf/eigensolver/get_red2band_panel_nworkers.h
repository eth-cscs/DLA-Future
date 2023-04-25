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

#include <algorithm>
#include <cmath>

#include <pika/runtime.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/tune.h"

namespace dlaf::eigensolver::internal {

inline size_t getReductionToBandPanelNWorkers() noexcept {
  const size_t nworkers = getTuneParameters().red2band_panel_nworkers;

  // Note: precautionarily we leave at least 1 thread "free" to do other stuff
  const size_t max_workers = pika::resource::get_thread_pool("default").get_os_thread_count() - 1;

  // 1 <= number of workers < max_workers
  return std::max<std::size_t>(1, std::min<std::size_t>(max_workers, nworkers));
}

}
