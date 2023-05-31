//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf_c/init.h>

#include <dlaf/init.h>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

static bool dlaf_initialized = false;

void dlaf_initialize(int argc, const char** argv){
  if(!dlaf_initialized){
    pika::program_options::options_description desc("");
    desc.add(dlaf::getOptionsDescription());

    // pika initialization
    pika::init_params params;
    params.rp_callback = dlaf::initResourcePartitionerHandler;
    params.desc_cmdline = desc;
    pika::start(nullptr, argc, argv, params);

    // DLA-Future initialization
    dlaf::initialize(argc, argv);
    dlaf_initialized = true;

    pika::suspend();
  }
}

void dlaf_finalize(){
  pika::resume();
  pika::finalize();
  dlaf::finalize();
  pika::stop();

  dlaf_initialized = false;
}
