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

#include <pika/runtime.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/tune.h"

namespace dlaf::eigensolver::internal {

inline size_t getReductionToBandPanelNWorkers() noexcept {
  const size_t nworkers = getTuneParameters().red2band_panel_nworkers;

  DLAF_ASSERT(nworkers >= 1 &&
                  nworkers < pika::resource::get_thread_pool("default").get_os_thread_count(),
              nworkers);

  return nworkers;
}

}
