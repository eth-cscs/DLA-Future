//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <pika/async_mpi/mpi_future.hpp>
#include <pika/runtime.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/communication/error.h>
#include <dlaf/init.h>
#include <dlaf/memory/memory_chunk.h>

#ifdef DLAF_WITH_CUDA
#include <dlaf/cublas/executor.h>
#include <dlaf/cuda/executor.h>
#include <dlaf/cusolver/executor.h>
#endif

#include <cstdlib>
#include <iostream>
#include <memory>

namespace dlaf {
std::ostream& operator<<(std::ostream& os, configuration const& cfg) {
  os << "  num_np_cuda_streams_per_thread = " << cfg.num_np_cuda_streams_per_thread << std::endl;
  os << "  num_hp_cuda_streams_per_thread = " << cfg.num_hp_cuda_streams_per_thread << std::endl;
  os << "  umpire_host_memory_pool_initial_bytes = " << cfg.umpire_host_memory_pool_initial_bytes
     << std::endl;
  os << "  umpire_device_memory_pool_initial_bytes = " << cfg.umpire_device_memory_pool_initial_bytes
     << std::endl;
  os << "  mpi_pool = " << cfg.mpi_pool << std::endl;
  os << "  mpi_mech = " << cfg.mpi_mech << std::endl;
  return os;
}

namespace internal {
bool& initialized() {
  static bool i = false;
  return i;
}

template <Backend D>
struct Init {
  // Initialization and finalization does nothing by default. Behaviour can be
  // overridden for backends.
  static void initialize(configuration const&) {}
  static void finalize() {}
};

template <>
struct Init<Backend::MC> {
  static void initialize(configuration const& cfg) {
    memory::internal::initializeUmpireHostAllocator(cfg.umpire_host_memory_pool_initial_bytes);
    // TODO: Consider disabling polling in finalize()
    if (cfg.mpi_mech == comm::MPIMech::Polling) {
      pika::mpi::experimental::init(false, cfg.mpi_pool);
    }
  }

  static void finalize() {
    memory::internal::finalizeUmpireHostAllocator();
  }
};

#ifdef DLAF_WITH_CUDA
static std::unique_ptr<cuda::StreamPool> np_stream_pool{nullptr};

void initializeNpCudaStreamPool(int device, std::size_t num_streams_per_thread) {
  DLAF_ASSERT(!np_stream_pool, "");
  np_stream_pool = std::make_unique<cuda::StreamPool>(device, num_streams_per_thread,
                                                      pika::threads::thread_priority::normal);
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
                                                      pika::threads::thread_priority::high);
}

void finalizeHpCudaStreamPool() {
  DLAF_ASSERT(bool(hp_stream_pool), "");
  hp_stream_pool.reset();
}

cuda::StreamPool getHpCudaStreamPool() {
  DLAF_ASSERT(bool(hp_stream_pool), "");
  return *hp_stream_pool;
}

static std::unique_ptr<cublas::HandlePool> cublas_handle_pool{nullptr};

void initializeCublasHandlePool() {
  DLAF_ASSERT(!cublas_handle_pool, "");
  cublas_handle_pool = std::make_unique<cublas::HandlePool>(0, CUBLAS_POINTER_MODE_HOST);
}

void finalizeCublasHandlePool() {
  DLAF_ASSERT(bool(cublas_handle_pool), "");
  cublas_handle_pool.reset();
}

cublas::HandlePool getCublasHandlePool() {
  DLAF_ASSERT(bool(cublas_handle_pool), "");
  return *cublas_handle_pool;
}

static std::unique_ptr<cusolver::HandlePool> cusolver_handle_pool{nullptr};

void initializeCusolverHandlePool() {
  DLAF_ASSERT(!cusolver_handle_pool, "");
  cusolver_handle_pool = std::make_unique<cusolver::HandlePool>(0);
}

void finalizeCusolverHandlePool() {
  DLAF_ASSERT(bool(cusolver_handle_pool), "");
  cusolver_handle_pool.reset();
}

cusolver::HandlePool getCusolverHandlePool() {
  DLAF_ASSERT(bool(cusolver_handle_pool), "");
  return *cusolver_handle_pool;
}

static std::unique_ptr<pika::cuda::experimental::cuda_pool> cuda_pool{nullptr};

void initializeCudaPool(int device, std::size_t num_np_streams, std::size_t num_hp_streams) {
  DLAF_ASSERT(!cuda_pool, "");
  cuda_pool =
      std::make_unique<pika::cuda::experimental::cuda_pool>(device, num_np_streams, num_hp_streams);
}

void finalizeCudaPool() {
  DLAF_ASSERT(bool(cuda_pool), "");
  cuda_pool.reset();
}

pika::cuda::experimental::cuda_pool getCudaPool() {
  DLAF_ASSERT(bool(cuda_pool), "");
  return *cuda_pool;
}

template <>
struct Init<Backend::GPU> {
  static void initialize(configuration const& cfg) {
    const int device = 0;
    memory::internal::initializeUmpireDeviceAllocator(cfg.umpire_device_memory_pool_initial_bytes);
    initializeNpCudaStreamPool(device, cfg.num_np_cuda_streams_per_thread);
    initializeHpCudaStreamPool(device, cfg.num_hp_cuda_streams_per_thread);
    initializeCublasHandlePool();
    initializeCusolverHandlePool();
    initializeCudaPool(device, cfg.num_np_cuda_streams_per_thread, cfg.num_hp_cuda_streams_per_thread);
    pika::cuda::experimental::detail::register_polling(pika::resource::get_thread_pool("default"));
  }

  static void finalize() {
    memory::internal::finalizeUmpireDeviceAllocator();
    finalizeNpCudaStreamPool();
    finalizeHpCudaStreamPool();
    finalizeCublasHandlePool();
    finalizeCusolverHandlePool();
    finalizeCudaPool();
  }
};
#endif

template <class T>
struct parseFromString {
  static T call(const std::string& val) {
    return val;
  };
};

template <>
struct parseFromString<std::size_t> {
  static std::size_t call(const std::string& var) {
    return std::stoul(var);
  };
};

template <>
struct parseFromString<comm::MPIMech> {
  static comm::MPIMech call(const std::string& var) {
    if (var == "yielding") {
      return comm::MPIMech::Yielding;
    }
    else if (var == "polling") {
      return comm::MPIMech::Polling;
    }

    std::cout << "Unknown value for --mech=" << var << "!" << std::endl;
    std::terminate();
    return comm::MPIMech::Polling;  // unreachable
  };
};

template <class T>
struct parseFromCommandLine {
  static T call(pika::program_options::variables_map const& vm, const std::string& cmd_val) {
    return vm[cmd_val].as<T>();
  }
};

template <>
struct parseFromCommandLine<comm::MPIMech> {
  static comm::MPIMech call(pika::program_options::variables_map const& vm, const std::string& cmd_val) {
    return parseFromString<comm::MPIMech>::call(vm[cmd_val].as<std::string>());
  }
};

template <class T>
void updateConfigurationValue(pika::program_options::variables_map const& vm, T& var,
                              std::string const& env_var, std::string const& cmdline_option) {
  const std::string dlaf_env_var = "DLAF_" + env_var;
  char* env_var_value = std::getenv(dlaf_env_var.c_str());
  if (env_var_value) {
    var = parseFromString<T>::call(env_var_value);
  }

  const std::string dlaf_cmdline_option = "dlaf:" + cmdline_option;
  if (vm.count(dlaf_cmdline_option)) {
    var = parseFromCommandLine<T>::call(vm, dlaf_cmdline_option);
  }
}

void updateConfiguration(pika::program_options::variables_map const& vm, configuration& cfg) {
  updateConfigurationValue(vm, cfg.num_np_cuda_streams_per_thread, "NUM_NP_CUDA_STREAMS_PER_THREAD",
                           "num-np-cuda-streams-per-thread");
  updateConfigurationValue(vm, cfg.num_hp_cuda_streams_per_thread, "NUM_HP_CUDA_STREAMS_PER_THREAD",
                           "num-hp-cuda-streams-per-thread");
  updateConfigurationValue(vm, cfg.umpire_host_memory_pool_initial_bytes,
                           "UMPIRE_HOST_MEMORY_POOL_INITIAL_BYTES",
                           "umpire-host-memory-pool-initial-bytes");
  updateConfigurationValue(vm, cfg.umpire_device_memory_pool_initial_bytes,
                           "UMPIRE_DEVICE_MEMORY_POOL_INITIAL_BYTES",
                           "umpire-device-memory-pool-initial-bytes");
  updateConfigurationValue(vm, cfg.mpi_mech, "MPI_MECH", "mpi-mech");
  cfg.mpi_pool = (pika::resource::pool_exists("mpi")) ? "mpi" : "default";
}

configuration& getConfiguration() {
  static configuration cfg;
  return cfg;
}
}

pika::program_options::options_description getOptionsDescription() {
  pika::program_options::options_description desc("DLA-Future options");

  desc.add_options()("dlaf:help", "Print help message");
  desc.add_options()("dlaf:print-config", "Print the DLA-Future configuration");
  desc.add_options()("dlaf:num-np-cuda-streams-per-thread", pika::program_options::value<std::size_t>(),
                     "Number of normal priority CUDA streams per worker thread");
  desc.add_options()("dlaf:num-hp-cuda-streams-per-thread", pika::program_options::value<std::size_t>(),
                     "Number of high priority CUDA streams per worker thread");
  desc.add_options()("dlaf:umpire-host-memory-pool-initial-bytes",
                     pika::program_options::value<std::size_t>(),
                     "Number of bytes to preallocate for pinned host memory pool");
  desc.add_options()("dlaf:umpire-device-memory-pool-initial-bytes",
                     pika::program_options::value<std::size_t>(),
                     "Number of bytes to preallocate for device memory pool");
  desc.add_options()("dlaf:mpi-mech",
                     pika::program_options::value<std::string>()->default_value("yielding"),
                     "MPI mechanism ('yielding', 'polling')");
  desc.add_options()("dlaf:no-mpi-pool", pika::program_options::bool_switch(), "Disable the MPI pool.");

  return desc;
}

void initialize(pika::program_options::variables_map const& vm, configuration const& user_cfg) {
  bool should_exit = false;
  if (vm.count("dlaf:help") > 0) {
    should_exit = true;
    std::cout << getOptionsDescription() << std::endl;
  }

  configuration cfg = user_cfg;
  internal::updateConfiguration(vm, cfg);
  internal::getConfiguration() = cfg;

  if (vm.count("dlaf:print-config") > 0) {
    std::cout << "DLA-Future configuration options:" << std::endl;
    std::cout << cfg << std::endl;
    std::cout << std::endl;
  }

  if (should_exit) {
    std::exit(0);
  }

  DLAF_ASSERT(!internal::initialized(), "");
  internal::Init<Backend::MC>::initialize(cfg);
#ifdef DLAF_WITH_CUDA
  internal::Init<Backend::GPU>::initialize(cfg);
#endif
  internal::initialized() = true;
}

void initialize(int argc, const char* const argv[], configuration const& user_cfg) {
  auto desc = getOptionsDescription();

  pika::program_options::variables_map vm;
  pika::program_options::store(pika::program_options::parse_command_line(argc, argv, desc), vm);
  pika::program_options::notify(vm);

  initialize(vm, user_cfg);
}

void finalize() {
  DLAF_ASSERT(internal::initialized(), "");
  internal::Init<Backend::MC>::finalize();
#ifdef DLAF_WITH_CUDA
  internal::Init<Backend::GPU>::finalize();
#endif
  internal::getConfiguration() = {};
  internal::initialized() = false;
}

ScopedInitializer::ScopedInitializer(pika::program_options::variables_map const& vm,
                                     configuration const& user_cfg) {
  initialize(vm, user_cfg);
}

ScopedInitializer::ScopedInitializer(int argc, const char* const argv[], configuration const& user_cfg) {
  initialize(argc, argv, user_cfg);
}

ScopedInitializer::~ScopedInitializer() {
  finalize();
}

void initResourcePartitionerHandler(pika::resource::partitioner& rp,
                                    pika::program_options::variables_map const& vm) {
  // Don't create the MPI pool if the user disabled it
  if (vm["dlaf:no-mpi-pool"].as<bool>())
    return;

  // Don't create the MPI pool if there is a single process
  int ntasks;
  DLAF_MPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &ntasks));
  if (ntasks == 1)
    return;

  // Disable idle backoff on the MPI pool
  using pika::threads::policies::scheduler_mode;
  auto mode = scheduler_mode::default_mode;
  mode = scheduler_mode(mode & ~scheduler_mode::enable_idle_backoff);

  // Create a thread pool with a single core that we will use for all
  // communication related tasks
  rp.create_thread_pool("mpi", pika::resource::scheduling_policy::local_priority_fifo, mode);
  rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0], "mpi");
}
}
