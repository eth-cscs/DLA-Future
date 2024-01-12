//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
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

void dlaf_initialize(int argc_pika, const char** argv_pika, int argc_dlaf,
                     const char** argv_dlaf) noexcept {
  if (!dlaf_initialized) {
    pika::program_options::options_description desc("");
    desc.add(dlaf::getOptionsDescription());

    // pika initialization
    pika::init_params params;
    params.rp_callback = dlaf::initResourcePartitionerHandler;
    params.desc_cmdline = desc;
    // After pika 0.21.0 pika::start reports errors only by exception and returns void
#if PIKA_VERSION_FULL >= 0x001500
    pika::start(argc_pika, argv_pika, params);
#else
    auto pika_started = pika::start(nullptr, argc_pika, argv_pika, params);
    DLAF_ASSERT(pika_started, pika_started);
#endif

    // DLA-Future initialization
    dlaf::initialize(argc_dlaf, argv_dlaf);
    dlaf_initialized = true;

    pika::suspend();
  }
}

void dlaf_finalize() noexcept {
  if (dlaf_initialized) {
    pika::resume();
    pika::finalize();
    dlaf::finalize();
    auto pika_stopped = pika::stop();
    DLAF_ASSERT(pika_stopped == 0, pika_stopped);

    dlaf_initialized = false;
  }
}
