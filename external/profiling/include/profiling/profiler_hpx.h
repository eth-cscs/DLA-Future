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

#include "profiler.h"

namespace dlaf {
namespace profiling {
namespace hpx {

void init_thread_getter() {
  profiler::instance().set_thread_id_getter(::hpx::get_worker_thread_num);
}

}
}
}
