//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <dlaf/init.h>
#include <dlaf_c/init.h>

static bool dlaf_initialized = false;

void dlaf_initialize(int argc_pika, const char** argv_pika, int argc_dlaf, const char** argv_dlaf) {
  if (!dlaf_initialized) {
    pika::program_options::options_description desc("");
    desc.add(dlaf::getOptionsDescription());

    // pika initialization
    pika::init_params params;
    params.rp_callback = dlaf::initResourcePartitionerHandler;
    params.desc_cmdline = desc;
    pika::start(nullptr, argc_pika, argv_pika, params);

    // DLA-Future initialization
    dlaf::initialize(argc_dlaf, argv_dlaf);
    dlaf_initialized = true;

    pika::suspend();
  }
}

void dlaf_finalize() {
  pika::resume();
  pika::finalize();
  dlaf::finalize();
  pika::stop();

  dlaf_initialized = false;
}
