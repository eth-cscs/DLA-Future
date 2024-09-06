//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <cstddef>
#include <iostream>
#include <string>

#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#ifdef DLAF_WITH_GPU
#include <pika/cuda.hpp>
#endif

#include <dlaf/types.h>

namespace dlaf {
/// DLA-Future configuration.
///
/// Holds configuration values that can be used to customize DLA-Future through
/// dlaf::initialize.
struct configuration {
  // NOTE: Remember to update the following if you add or change parameters below:
  // - The operator<< overload in init.cpp
  // - updateConfiguration in init.cpp to update the value from command line options and environment
  //   values
  // - getOptionsDescription to add a corresponding command line option
  std::size_t num_np_gpu_streams_per_thread = 3;
  std::size_t num_hp_gpu_streams_per_thread = 3;
  std::size_t num_gpu_blas_handles = 16;
  std::size_t num_gpu_lapack_handles = 16;
  std::size_t umpire_host_memory_pool_initial_bytes = 1 << 30;
  std::size_t umpire_host_memory_pool_next_bytes = 1 << 30;
  std::size_t umpire_host_memory_pool_alignment_bytes = 16;
  double umpire_host_memory_pool_coalescing_free_ratio = 1.0;
  double umpire_host_memory_pool_coalescing_reallocation_ratio = 1.0;
  std::size_t umpire_device_memory_pool_initial_bytes = 1 << 30;
  std::size_t umpire_device_memory_pool_next_bytes = 1 << 30;
  std::size_t umpire_device_memory_pool_alignment_bytes = 16;
  double umpire_device_memory_pool_coalescing_free_ratio = 1.0;
  double umpire_device_memory_pool_coalescing_reallocation_ratio = 1.0;
};

std::ostream& operator<<(std::ostream& os, const configuration& cfg);

namespace internal {
configuration& getConfiguration();

#ifdef DLAF_WITH_GPU
pika::cuda::experimental::cuda_pool getGpuPool();
#endif
}

/// Returns the DLA-Future command-line options description.
pika::program_options::options_description getOptionsDescription();

/// Initialize DLA-Future.
///
/// Should be called once before every dlaf::finalize call. This overload can
/// be used when the application is using pika::program_options for its own
/// command-line parsing needs. The user is responsible for ensuring that
/// DLA-Future command-line options are parsed. The DLA-Future options can be
/// retrieved with dlaf::getOptionsDescription.
///
/// @param vm parsed command-line options as provided by the application entry point.
/// @param user_cfg user-provided default configuration. Takes precedence over
/// DLA-Future defaults.
void initialize(const pika::program_options::variables_map& vm, const configuration& user_cfg = {});

/// Initialize DLA-Future.
///
/// Should be called once before every dlaf::finalize call.
///
/// @param argc as provided by the application entry point.
/// @param argv as provided by the application entry point.
/// @param user_cfg user-provided default configuration. Takes precedence over
/// DLA-Future defaults.
void initialize(int argc, const char* const argv[], const configuration& user_cfg = {});

/// Finalize DLA-Future.
///
/// Should be called once after every dlaf::initialize call.
void finalize();

/// RAII helper for dlaf::initialize and dlaf::finalize.
///
/// Calls dlaf::initialize on construction and dlaf::finalize on destruction.
struct [[nodiscard]] ScopedInitializer {
  ScopedInitializer(const pika::program_options::variables_map& vm, const configuration& user_cfg = {});
  ScopedInitializer(int argc, const char* const argv[], const configuration& user_cfg = {});
  ~ScopedInitializer();

  ScopedInitializer(ScopedInitializer&&) = delete;
  ScopedInitializer(const ScopedInitializer&) = delete;
  ScopedInitializer& operator=(ScopedInitializer&&) = delete;
  ScopedInitializer& operator=(const ScopedInitializer&) = delete;
};

/// Initialize the MPI pool.
///
///
void initResourcePartitionerHandler(pika::resource::partitioner& rp,
                                    const pika::program_options::variables_map& vm);
}
