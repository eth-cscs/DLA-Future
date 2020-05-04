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

/// @file

#include "dlaf/common/assert.h"

#include <mpi.h>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/program_options/options_description.hpp>

namespace dlaf {
namespace comm {

/// Create a thread pool for MPI work and add (enabled) PUs on the first core.
///
/// Preconditions:
/// - there is more than 1 core available
/// - the function must be called before `hpx::init()`.
///
void init_mpi_pool(hpx::program_options::options_description& desc, int argc, char** argv) {
  hpx::resource::partitioner rp(desc, argc, argv);
  auto const& cores_arr = rp.numa_domains()[0].cores();
  DLAF_ASSERT(cores_arr.size() > 1);
  rp.create_thread_pool("mpi");
  rp.add_resource(cores_arr[0].pus(), "mpi");
}

}
}
