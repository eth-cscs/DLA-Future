//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/common/assert.h>
#include <dlaf/init.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#include <dlaf/cuda/executor.h>
#endif

#include <memory>

namespace dlaf {
std::ostream& operator<<(std::ostream& os, configuration const& cfg) {
  os << "num_np_cuda_streams_per_thread = " << cfg.num_np_cuda_streams_per_thread << std::endl;
  os << "num_hp_cuda_streams_per_thread = " << cfg.num_hp_cuda_streams_per_thread;

  return os;
}

namespace internal {
bool& initialized() {
  static bool i = false;
  return i;
}

#ifdef DLAF_WITH_CUDA
static std::unique_ptr<cuda::StreamPool> np_stream_pool{nullptr};

void initializeNpCudaStreamPool(int device, std::size_t num_streams_per_thread) {
  DLAF_ASSERT(!np_stream_pool, "");
  np_stream_pool = std::make_unique<cuda::StreamPool>(device, num_streams_per_thread,
                                                      hpx::threads::thread_priority::normal);
}

void finalizeNpCudaStreamPool() {
  DLAF_ASSERT(bool(np_stream_pool), "");
  np_stream_pool.reset();
}

cuda::StreamPool getNpCudaStreamPool() {
  DLAF_ASSERT(bool(np_stream_pool), "");
  return *np_stream_pool;
}

static std::unique_ptr<cuda::StreamPool> hp_stream_pool{nullptr};

void initializeHpCudaStreamPool(int device, std::size_t num_streams_per_thread) {
  DLAF_ASSERT(!hp_stream_pool, "");
  hp_stream_pool = std::make_unique<cuda::StreamPool>(device, num_streams_per_thread,
                                                      hpx::threads::thread_priority::high);
}

void finalizeHpCudaStreamPool() {
  DLAF_ASSERT(bool(hp_stream_pool), "");
  hp_stream_pool.reset();
}

cuda::StreamPool getHpCudaStreamPool() {
  DLAF_ASSERT(bool(hp_stream_pool), "");
  return *hp_stream_pool;
}

static std::unique_ptr<cublas::HandlePool> handle_pool{nullptr};

void initializeCublasHandlePool() {
  DLAF_ASSERT(!handle_pool, "");
  handle_pool = std::make_unique<cublas::HandlePool>(0, CUBLAS_POINTER_MODE_HOST);
}

void finalizeCublasHandlePool() {
  DLAF_ASSERT(bool(handle_pool), "");
  handle_pool.reset();
}

cublas::HandlePool getCublasHandlePool() {
  DLAF_ASSERT(bool(handle_pool), "");
  return *handle_pool;
}
#endif

template <>
inline void update_configuration(hpx::program_options::variables_map const& vm, std::size_t& var,
                                 std::string const& env_var, std::string const& cmdline_option) {
  const std::string dlaf_env_var = "DLAF_" + env_var;
  char* env_var_value = std::getenv(dlaf_env_var.c_str());
  if (env_var_value) {
    var = std::stoul(env_var_value);
  }

  const std::string dlaf_cmdline_option = "dlaf:" + cmdline_option;
  if (vm.count(dlaf_cmdline_option)) {
    var = vm[dlaf_cmdline_option].as<std::size_t>();
  }
}

configuration get_configuration(hpx::program_options::variables_map const& vm,
                                configuration const& user_cfg) {
  configuration cfg = user_cfg;

  update_configuration(vm, cfg.num_np_cuda_streams_per_thread, "NUM_NP_CUDA_STREAMS_PER_THREAD",
                       "num-np-cuda-streams-per-thread");
  update_configuration(vm, cfg.num_hp_cuda_streams_per_thread, "NUM_HP_CUDA_STREAMS_PER_THREAD",
                       "num-hp-cuda-streams-per-thread");

  return cfg;
}
}

hpx::program_options::options_description get_options_description() {
  hpx::program_options::options_description desc("DLA-Future options");

  desc.add_options()("dlaf:help", "Print help message");
  desc.add_options()("dlaf:print-config", "Print the DLA-Future configuration");
  desc.add_options()("dlaf:num-np-cuda-streams-per-thread", hpx::program_options::value<std::size_t>(),
                     "Number of normal priority CUDA streams per worker thread");
  desc.add_options()("dlaf:num-hp-cuda-streams-per-thread", hpx::program_options::value<std::size_t>(),
                     "Number of high priority CUDA streams per worker thread");

  return desc;
}

void initialize(hpx::program_options::variables_map const& vm, configuration const& user_cfg) {
  if (vm.count("dlaf:help") > 0) {
    std::cout << get_options_description() << std::endl;
  }

  auto cfg = internal::get_configuration(vm, user_cfg);

  if (vm.count("dlaf:print-config") > 0) {
    std::cout << cfg << std::endl;
  }

  DLAF_ASSERT(!internal::initialized(), "");
  internal::Init<Backend::MC>::initialize(cfg);
#ifdef DLAF_WITH_CUDA
  internal::Init<Backend::GPU>::initialize(cfg);
#endif
  internal::initialized() = true;
}

void initialize(int argc, char** argv, configuration const& user_cfg) {
  auto desc = get_options_description();

  hpx::program_options::variables_map vm;
  hpx::program_options::store(hpx::program_options::parse_command_line(argc, argv, desc), vm);
  hpx::program_options::notify(vm);

  initialize(vm, user_cfg);
}

void finalize() {
  DLAF_ASSERT(internal::initialized(), "");
  internal::Init<Backend::MC>::finalize();
#ifdef DLAF_WITH_CUDA
  internal::Init<Backend::GPU>::finalize();
#endif
  internal::initialized() = false;
}
}
