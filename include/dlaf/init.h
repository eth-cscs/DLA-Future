//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020, ETH Zurich
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

#include <dlaf/types.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#endif

namespace dlaf {
struct configuration {
  std::size_t num_np_cuda_streams_per_thread = 3;
  std::size_t num_hp_cuda_streams_per_thread = 3;
};

std::ostream& operator<<(std::ostream& os, configuration const& cfg);

namespace internal {
bool& initialized();

template <Backend D>
struct Init {
  // Initialization and finalization does nothing by default. Behaviour can be
  // overridden for backends.
  static void initialize(configuration const&) {}
  static void finalize() {}
};

#ifdef DLAF_WITH_CUDA
void initializeNpCudaStreamPool(int device, std::size_t num_streams_per_thread);
void finalizeNpCudaStreamPool();
cuda::StreamPool getNpCudaStreamPool();

void initializeHpCudaStreamPool(int device, std::size_t num_streams_per_thread);
void finalizeHpCudaStreamPool();
cuda::StreamPool getHpCudaStreamPool();

void initializeCublasHandlePool();
void finalizeCublasHandlePool();
cublas::HandlePool getCublasHandlePool();

template <>
struct Init<Backend::GPU> {
  static void initialize(configuration const& cfg) {
    // TODO: Do we already want to expose this? Not properly supported by
    // backend (i.e. only one device at a time supported).
    const int device = 0;
    initializeNpCudaStreamPool(device, cfg.num_np_cuda_streams_per_thread);
    initializeHpCudaStreamPool(device, cfg.num_hp_cuda_streams_per_thread);
    initializeCublasHandlePool();
    hpx::cuda::experimental::detail::register_polling(hpx::resource::get_thread_pool("default"));
  }

  static void finalize() {
    finalizeNpCudaStreamPool();
    finalizeHpCudaStreamPool();
    finalizeCublasHandlePool();
    hpx::cuda::experimental::detail::unregister_polling(hpx::resource::get_thread_pool("default"));
  }
};
#endif

template <typename T>
void update_configuration(hpx::program_options::variables_map const& vm, T& var,
                          std::string const& cmdline_option, std::string const& env_var);
configuration get_configuration(hpx::program_options::variables_map const& vm,
                                configuration const& user_cfg);
}

hpx::program_options::options_description get_options_description();
void initialize(hpx::program_options::variables_map const& vm, configuration const& user_cfg = {});
void initialize(int argc, char** argv, configuration const& user_cfg = {});
void finalize();
}
