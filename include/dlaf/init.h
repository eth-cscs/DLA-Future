//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <iostream>

#include <hpx/include/resource_partitioner.hpp>
#include <hpx/program_options.hpp>

#ifdef DLAF_WITH_CUDA
#include <hpx/modules/async_cuda.hpp>
#endif

#include <dlaf/communication/mech.h>
#include <dlaf/types.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#endif

namespace dlaf {
/// DLA-Future configuration.
///
/// Holds configuration values that can be used to customize DLA-Future through
/// dlaf::initialize.
struct configuration {
  std::size_t num_np_cuda_streams_per_thread = 3;
  std::size_t num_hp_cuda_streams_per_thread = 3;
  std::size_t umpire_host_memory_pool_initial_bytes = 1 << 30;
  std::size_t umpire_device_memory_pool_initial_bytes = 1 << 30;
  std::string mpi_pool = "mpi";
  comm::MPIMech mpi_mech = comm::MPIMech::Polling;
};

std::ostream& operator<<(std::ostream& os, configuration const& cfg);

namespace internal {
configuration& getConfiguration();

#ifdef DLAF_WITH_CUDA
cuda::StreamPool getNpCudaStreamPool();
cuda::StreamPool getHpCudaStreamPool();
cublas::HandlePool getCublasHandlePool();
#endif
}

/// Returns the DLA-Future command-line options description.
hpx::program_options::options_description getOptionsDescription();

/// Initialize DLA-Future.
///
/// Should be called once before every dlaf::finalize call. This overload can
/// be used when the application is using hpx::program_options for its own
/// command-line parsing needs. The user is responsible for ensuring that
/// DLA-Future command-line options are parsed. The DLA-Future options can be
/// retrieved with dlaf::getOptionsDescription.
///
/// @param vm parsed command-line options as provided by the application entry point.
/// @param user_cfg user-provided default configuration. Takes precedence over
/// DLA-Future defaults.
void initialize(hpx::program_options::variables_map const& vm, configuration const& user_cfg = {});

/// Initialize DLA-Future.
///
/// Should be called once before every dlaf::finalize call.
///
/// @param argc as provided by the application entry point.
/// @param argv as provided by the application entry point.
/// @param user_cfg user-provided default configuration. Takes precedence over
/// DLA-Future defaults.
void initialize(int argc, const char* const argv[], configuration const& user_cfg = {});

/// Finalize DLA-Future.
///
/// Should be called once after every dlaf::initialize call.
void finalize();

/// Initialize the MPI pool.
///
///
void initResourcePartitionerHandler(hpx::resource::partitioner& rp,
                                    hpx::program_options::variables_map const& vm);
}
