//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

#include <pika/mpi.hpp>
#include <pika/runtime.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/communication/error.h>
#include <dlaf/init.h>
#include <dlaf/memory/memory_chunk.h>
#include <dlaf/tune.h>

namespace dlaf {
std::ostream& operator<<(std::ostream& os, const configuration& cfg) {
  // clang-format off
  os << "  num_np_gpu_streams_per_thread = " << cfg.num_np_gpu_streams_per_thread << std::endl;
  os << "  num_hp_gpu_streams_per_thread = " << cfg.num_hp_gpu_streams_per_thread << std::endl;
  os << "  num_gpu_blas_handles = " << cfg.num_gpu_blas_handles << std::endl;
  os << "  num_gpu_lapack_handles = " << cfg.num_gpu_lapack_handles << std::endl;
  os << "  umpire_host_memory_pool_initial_bytes = " << cfg.umpire_host_memory_pool_initial_bytes << std::endl;
  os << "  umpire_host_memory_pool_next_bytes = " << cfg.umpire_host_memory_pool_next_bytes << std::endl;
  os << "  umpire_host_memory_pool_alignment_bytes = " << cfg.umpire_host_memory_pool_alignment_bytes << std::endl;
  os << "  umpire_host_memory_pool_coalescing_free_ratio = " << cfg.umpire_host_memory_pool_coalescing_free_ratio << std::endl;
  os << "  umpire_host_memory_pool_coalescing_reallocation_ratio = " << cfg.umpire_host_memory_pool_coalescing_reallocation_ratio << std::endl;
  os << "  umpire_device_memory_pool_initial_bytes = " << cfg.umpire_device_memory_pool_initial_bytes << std::endl;
  os << "  umpire_device_memory_pool_next_bytes = " << cfg.umpire_device_memory_pool_next_bytes << std::endl;
  os << "  umpire_device_memory_pool_alignment_bytes = " << cfg.umpire_device_memory_pool_alignment_bytes << std::endl;
  os << "  umpire_device_memory_pool_coalescing_free_ratio = " << cfg.umpire_device_memory_pool_coalescing_free_ratio << std::endl;
  os << "  umpire_device_memory_pool_coalescing_reallocation_ratio = " << cfg.umpire_device_memory_pool_coalescing_reallocation_ratio << std::endl;
  // clang-format on
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
  static void initialize(const configuration&) {}
  static void finalize() {}
};

template <>
struct Init<Backend::MC> {
  static void initialize(const configuration& cfg) {
    memory::internal::initializeUmpireHostAllocator(
        cfg.umpire_host_memory_pool_initial_bytes, cfg.umpire_host_memory_pool_next_bytes,
        cfg.umpire_host_memory_pool_alignment_bytes, cfg.umpire_host_memory_pool_coalescing_free_ratio,
        cfg.umpire_host_memory_pool_coalescing_reallocation_ratio);
    // install mpi polling loop
    // pika::mpi::experimental::init(false);
    pika::mpi::experimental::init(false, true);
    pika::mpi::experimental::register_polling();
  }

  static void finalize() {
    memory::internal::finalizeUmpireHostAllocator();
  }
};

#ifdef DLAF_WITH_GPU
static std::unique_ptr<pika::cuda::experimental::cuda_pool> gpu_pool{nullptr};

void initializeGpuPool(int device, std::size_t num_np_streams, std::size_t num_hp_streams,
                       std::size_t num_blas_handles, std::size_t num_lapack_handles) {
  DLAF_ASSERT(!gpu_pool, "");
  // HIP currently requires not using hipStreamNonBlocking as some rocSOLVER
  // functions such as potrf are not safe to use with it (see
  // https://github.com/ROCmSoftwarePlatform/rocSOLVER/issues/436).
  gpu_pool =
      std::make_unique<pika::cuda::experimental::cuda_pool>(device, num_np_streams, num_hp_streams,
#if defined(DLAF_WITH_CUDA)
                                                            cudaStreamNonBlocking
#else
                                                            0
#endif
#if PIKA_VERSION_FULL >= 0x001D00  // >= 0.29.0
                                                            ,
                                                            num_blas_handles, num_lapack_handles
#endif
      );
#if PIKA_VERSION_FULL < 0x001D00  // < 0.29.0
  dlaf::internal::silenceUnusedWarningFor(num_blas_handles, num_lapack_handles);
#endif
}

void finalizeGpuPool() {
  DLAF_ASSERT(bool(gpu_pool), "");
  gpu_pool.reset();
}

pika::cuda::experimental::cuda_pool getGpuPool() {
  DLAF_ASSERT(bool(gpu_pool), "");
  return *gpu_pool;
}

template <>
struct Init<Backend::GPU> {
  static void initialize(const configuration& cfg) {
    const int device = 0;
    // setup polling on default pool, enable exceptions and init mpi internals
    pika::mpi::experimental::init(false, true);
    pika::mpi::experimental::register_polling();

    memory::internal::initializeUmpireDeviceAllocator(
        cfg.umpire_device_memory_pool_initial_bytes, cfg.umpire_device_memory_pool_initial_bytes,
        cfg.umpire_device_memory_pool_alignment_bytes, cfg.umpire_host_memory_pool_coalescing_free_ratio,
        cfg.umpire_host_memory_pool_coalescing_reallocation_ratio);
    memory::internal::initializeUmpireDeviceAllocator(cfg.umpire_device_memory_pool_initial_bytes, cfg.umpire_device_memory_pool_initial_bytes, cfg.umpire_device_memory_pool_alignment_bytes, cfg.umpire_host_memory_pool_coalescing_free_ratio, cfg.umpire_host_memory_pool_coalescing_reallocation_ratio);
    initializeGpuPool(device, cfg.num_np_gpu_streams_per_thread, cfg.num_hp_gpu_streams_per_thread,
                      cfg.num_gpu_blas_handles, cfg.num_gpu_lapack_handles);
    pika::cuda::experimental::detail::register_polling(
        pika::resource::get_thread_pool(pika::cuda::experimental::get_pool_name()));
  }

  static void finalize() {
    memory::internal::finalizeUmpireDeviceAllocator();
    finalizeGpuPool();
  }
};
#endif

template <class T>
struct parseFromString {
  static std::optional<T> call(const std::string& val) {
    return val;
  }
};

template <>
struct parseFromString<std::size_t> {
  static std::optional<std::size_t> call(const std::string& var) {
    return std::stoull(var);
  }
};

template <>
struct parseFromString<SizeType> {
  static std::optional<SizeType> call(const std::string& var) {
    return std::stoll(var);
  }
};

template <>
struct parseFromString<double> {
  static std::optional<SizeType> call(const std::string& var) {
    return std::stod(var);
  }
};

template <>
struct parseFromString<bool> {
  static std::optional<bool> call(const std::string& var) {
    if (is_one_of_ignore_case(var, {"ON", "TRUE", "YES", "1"}))
      return true;
    if (is_one_of_ignore_case(var, {"OFF", "FALSE", "NO", "0"}))
      return false;
    return std::nullopt;
  }

private:
  static bool is_one_of_ignore_case(std::string value, const std::array<std::string, 4>& values) {
    std::transform(value.cbegin(), value.cend(), value.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return (values.cend()) != std::find(values.cbegin(), values.cend(), value);
  }
};

template <class T>
struct parseFromCommandLine {
  static T call(const pika::program_options::variables_map& vm, const std::string& cmd_val) {
    return vm[cmd_val].as<T>();
  }
};

template <class T>
void updateConfigurationValue(const pika::program_options::variables_map& vm, T& var,
                              const std::string& env_var, const std::string& cmdline_option) {
  DLAF_ASSERT(env_var.find("DLAF") == std::string::npos, env_var);
  DLAF_ASSERT(cmdline_option.find("dlaf") == std::string::npos, cmdline_option);

  const std::string dlaf_env_var = "DLAF_" + env_var;
  char* env_var_value = std::getenv(dlaf_env_var.c_str());
  if (env_var_value) {
    if (auto parsed_value = parseFromString<T>::call(env_var_value)) {
      var = parsed_value.value();
    }
    else {
      std::cerr << "Environment variable " << dlaf_env_var << " has an invalid value (='"
                << env_var_value << "').\n";
      std::terminate();
    }
  }

  const std::string dlaf_cmdline_option = "dlaf:" + cmdline_option;
  if (vm.count(dlaf_cmdline_option)) {
    var = parseFromCommandLine<T>::call(vm, dlaf_cmdline_option);
  }
}

void warnUnusedConfigurationOption(const pika::program_options::variables_map& vm,
                                   const std::string& env_var, const std::string& cmdline_option,
                                   const std::string& reason) {
  DLAF_ASSERT(env_var.find("DLAF") == std::string::npos, env_var);
  DLAF_ASSERT(cmdline_option.find("dlaf") == std::string::npos, cmdline_option);

  const std::string dlaf_env_var = "DLAF_" + env_var;
  char* env_var_value = std::getenv(dlaf_env_var.c_str());
  if (env_var_value) {
    std::cerr << "[WARNING] Environment variable " << dlaf_env_var << " is set but will be ignored ("
              << reason << ")\n";
  }

  const std::string dlaf_cmdline_option = "dlaf:" + cmdline_option;
  if (vm.count(dlaf_cmdline_option) >= 1) {
    std::cerr << "[WARNING] Command line option " << dlaf_cmdline_option
              << " is set but will be ignored (" << reason << ")\n";
  }
}

void updateConfiguration(const pika::program_options::variables_map& vm, configuration& cfg) {
  // clang-format off
  updateConfigurationValue(vm, cfg.num_gpu_blas_handles, "NUM_GPU_BLAS_HANDLES", "num-gpu-blas-handles");
  updateConfigurationValue(vm, cfg.num_gpu_lapack_handles, "NUM_GPU_LAPACK_HANDLES", "num-gpu-lapack-handles");
#if PIKA_VERSION_FULL < 0x001D00  // < 0.29.0
  warnUnusedConfigurationOption(vm, "NUM_GPU_BLAS_HANDLES", "num-gpu-blas-handles", "only supported with pika 0.29.0 or newer");
  warnUnusedConfigurationOption(vm, "NUM_GPU_LAPACK_HANDLES", "num-gpu-lapack-handles", "only supported with pika 0.29.0 or newer");
#endif
  updateConfigurationValue(vm, cfg.num_np_gpu_streams_per_thread, "NUM_NP_GPU_STREAMS_PER_THREAD", "num-np-gpu-streams-per-thread");
  updateConfigurationValue(vm, cfg.num_hp_gpu_streams_per_thread, "NUM_HP_GPU_STREAMS_PER_THREAD", "num-hp-gpu-streams-per-thread");
  updateConfigurationValue(vm, cfg.umpire_host_memory_pool_initial_bytes, "UMPIRE_HOST_MEMORY_POOL_INITIAL_BYTES", "umpire-host-memory-pool-initial-bytes");
  updateConfigurationValue(vm, cfg.umpire_host_memory_pool_next_bytes, "UMPIRE_HOST_MEMORY_POOL_NEXT_BYTES", "umpire-host-memory-pool-next-bytes");
  updateConfigurationValue(vm, cfg.umpire_host_memory_pool_alignment_bytes, "UMPIRE_HOST_MEMORY_POOL_ALIGNMENT_BYTES", "umpire-host-memory-pool-alignment-bytes");
  updateConfigurationValue(vm, cfg.umpire_host_memory_pool_coalescing_free_ratio, "UMPIRE_HOST_MEMORY_POOL_COALESCING_FREE_RATIO", "umpire-host-memory-pool-coalescing-free-ratio");
  updateConfigurationValue(vm, cfg.umpire_host_memory_pool_coalescing_reallocation_ratio, "UMPIRE_HOST_MEMORY_POOL_COALESCING_REALLOCATION_RATIO", "umpire-host-memory-pool-coalescing-reallocation-ratio");
  updateConfigurationValue(vm, cfg.umpire_device_memory_pool_initial_bytes, "UMPIRE_DEVICE_MEMORY_POOL_INITIAL_BYTES", "umpire-device-memory-pool-initial-bytes");
  updateConfigurationValue(vm, cfg.umpire_device_memory_pool_next_bytes, "UMPIRE_DEVICE_MEMORY_POOL_NEXT_BYTES", "umpire-device-memory-pool-next-bytes");
  updateConfigurationValue(vm, cfg.umpire_device_memory_pool_alignment_bytes, "UMPIRE_DEVICE_MEMORY_POOL_ALIGNMENT_BYTES", "umpire-device-memory-pool-alignment-bytes");
  updateConfigurationValue(vm, cfg.umpire_device_memory_pool_coalescing_free_ratio, "UMPIRE_DEVICE_MEMORY_POOL_COALESCING_FREE_RATIO", "umpire-device-memory-pool-coalescing-free-ratio");
  updateConfigurationValue(vm, cfg.umpire_device_memory_pool_coalescing_reallocation_ratio, "UMPIRE_DEVICE_MEMORY_POOL_COALESCING_REALLOCATION_RATIO", "umpire-device-memory-pool-coalescing-reallocation-ratio");
  // clang-format on

  // Warn if not using MPI pool without --dlaf:no-mpi-pool
  // int mpi_initialized;
  // DLAF_MPI_CHECK_ERROR(MPI_Initialized(&mpi_initialized));
  // if (mpi_initialized) {
  //   int ntasks;
  //   DLAF_MPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &ntasks));
  //   if (ntasks != 1 && cfg.mpi_pool == "default" && !vm["dlaf:no-mpi-pool"].as<bool>()) {
  //     std::cerr << "Warning! DLA-Future is not using the \"mpi\" pika thread pool for "
  //                  "MPI communication but --dlaf:no-mpi-pool is not set. This may "
  //                  "indicate a bug in DLA-Future or pika. Performance may be degraded."
  //               << std::endl;
  //   }
  // }

  // update tune parameters
  //
  // NOTE: Environment variables should omit the DLAF_ prefix and command line options the dlaf: prefix.
  // These are added automatically by updateConfigurationValue.
  auto& param = getTuneParameters();
  updateConfigurationValue(vm, param.red2band_panel_nworkers, "RED2BAND_PANEL_NWORKERS",
                           "red2band-panel-nworkers");

  updateConfigurationValue(vm, param.red2band_barrier_busy_wait_us, "RED2BAND_BARRIER_BUSY_WAIT_US",
                           "red2band-barrier-busy-wait-us");

  updateConfigurationValue(vm, param.eigensolver_min_band, "EIGENSOLVER_MIN_BAND",
                           "eigensolver-min-band");

  updateConfigurationValue(vm, param.band_to_tridiag_1d_block_size_base,
                           "BAND_TO_TRIDIAG_1D_BLOCK_SIZE_BASE", "band-to-tridiag-1d-block-size-base");

  updateConfigurationValue(vm, param.debug_dump_cholesky_factorization_data,
                           "DEBUG_DUMP_CHOLESKY_FACTORIZATION_DATA", "");
  updateConfigurationValue(vm, param.debug_dump_generalized_eigensolver_data,
                           "DEBUG_DUMP_GENERALIZED_EIGENSOLVER_DATA", "");
  updateConfigurationValue(vm, param.debug_dump_generalized_to_standard_data,
                           "DEBUG_DUMP_GENERALIZED_TO_STANDARD_DATA", "");
  updateConfigurationValue(vm, param.debug_dump_eigensolver_data, "DEBUG_DUMP_EIGENSOLVER_DATA", "");
  updateConfigurationValue(vm, param.debug_dump_reduction_to_band_data,
                           "DEBUG_DUMP_REDUCTION_TO_BAND_DATA", "");
  updateConfigurationValue(vm, param.debug_dump_band_to_tridiagonal_data,
                           "DEBUG_DUMP_BAND_TO_TRIDIAGONAL_DATA", "");
  updateConfigurationValue(vm, param.debug_dump_tridiag_solver_data, "DEBUG_DUMP_TRIDIAG_SOLVER_DATA",
                           "");

  updateConfigurationValue(vm, param.tridiag_rank1_nworkers, "TRIDIAG_RANK1_NWORKERS",
                           "tridiag-rank1-nworkers");

  updateConfigurationValue(vm, param.tridiag_rank1_barrier_busy_wait_us,
                           "TRIDIAG_RANK1_BARRIER_BUSY_WAIT_US", "tridiag-rank1-barrier-busy-wait-us");

  updateConfigurationValue(vm, param.bt_band_to_tridiag_hh_apply_group_size,
                           "BT_BAND_TO_TRIDIAG_HH_APPLY_GROUP_SIZE",
                           "bt-band-to-tridiag-hh-apply-group-size");

  updateConfigurationValue(vm, param.communicator_grid_num_pipelines, "COMMUNICATOR_GRID_NUM_PIPELINES",
                           "communicator-grid-num-pipelines");
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
  desc.add_options()("dlaf:num-gpu-blas-handles", pika::program_options::value<std::size_t>(), "Number of GPU BLAS (cuBLAS/rocBLAS) handles");
  desc.add_options()("dlaf:num-gpu-lapack-handles", pika::program_options::value<std::size_t>(), "Number of GPU LAPACK (cuSOLVER/rocSOLVER) handles");
  desc.add_options()("dlaf:num-np-gpu-streams-per-thread", pika::program_options::value<std::size_t>(), "Number of normal priority GPU streams per worker thread");
  desc.add_options()("dlaf:num-hp-gpu-streams-per-thread", pika::program_options::value<std::size_t>(), "Number of high priority GPU streams per worker thread");
  desc.add_options()("dlaf:umpire-host-memory-pool-initial-bytes", pika::program_options::value<std::size_t>(), "Number of bytes to preallocate for pinned host memory pool");
  desc.add_options()("dlaf:umpire-host-memory-pool-next-bytes", pika::program_options::value<std::size_t>(), "Number of bytes to allocate in blocks after the first block for pinned host memory pool");
  desc.add_options()("dlaf:umpire-host-memory-pool-alignment-bytes", pika::program_options::value<std::size_t>(), "Alignment of allocations in bytes in pinned host memory pool");
  desc.add_options()("dlaf:umpire-host-memory-pool-coalescing-free-ratio", pika::program_options::value<double>(), "Required ratio of free memory in pinned host memory pool before performing coalescing of free blocks");
  desc.add_options()("dlaf:umpire-host-memory-pool-coalescing-reallocation-ratio", pika::program_options::value<double>(), "Ratio of current used memory in pinned host memory pool to use for reallocation of new blocks when coalescing free blocks");
  desc.add_options()("dlaf:umpire-device-memory-pool-initial-bytes", pika::program_options::value<std::size_t>(), "Number of bytes to preallocate for device memory pool");
  desc.add_options()("dlaf:umpire-device-memory-pool-next-bytes", pika::program_options::value<std::size_t>(), "Number of bytes to allocate in blocks after the first block for device memory pool");
  desc.add_options()("dlaf:umpire-device-memory-pool-alignment-bytes", pika::program_options::value<std::size_t>(), "Alignment of allocations in bytes in device memory pool");
  desc.add_options()("dlaf:umpire-device-memory-pool-coalescing-free-ratio", pika::program_options::value<double>(), "Required ratio of free memory in device memory pool before performing coalescing of free blocks");
  desc.add_options()("dlaf:umpire-device-memory-pool-coalescing-reallocation-ratio", pika::program_options::value<double>(), "Ratio of current used memory in device memory pool to use for reallocation of new blocks when coalescing free blocks");
  desc.add_options()("dlaf:no-mpi-pool", pika::program_options::bool_switch(), "Disable the MPI pool.");

  // Tune parameters command line options
  desc.add_options()(
      "dlaf:red2band-panel-nworkers", pika::program_options::value<std::size_t>(),
      "The maximum number of threads to use for computing the panel in the reduction to band algorithm.");
  desc.add_options()(
      "dlaf:red2band-barrier-busy-wait-us", pika::program_options::value<std::size_t>(),
      "The duration in microseconds to busy-wait in barriers in the reduction to band algorithm.");
  desc.add_options()(
      "dlaf:eigensolver-min-band", pika::program_options::value<SizeType>(),
      "The minimum value to start looking for a divisor of the block size. When larger than the block size, the block size will be used instead.");
  desc.add_options()(
      "dlaf:band-to-tridiag-1d-block-size-base", pika::program_options::value<SizeType>(),
      "The 1D block size for band_to_tridiagonal is computed as 1d_block_size_base / nb * nb. (The input matrix is distributed with a {nb x nb} block size.)");
  desc.add_options()(
      "dlaf:tridiag-rank1-nworkers", pika::program_options::value<std::size_t>(),
      "The maximum number of threads to use for computing rank1 problem solution in tridiagonal solver algorithm.");
  desc.add_options()(
      "dlaf:tridiag-rank1-barrier-busy-wait-us", pika::program_options::value<std::size_t>(),
      "The duration in microseconds to busy-wait in barriers when computing rank1 problem solution in the tridiagonal solver algorithm.");
  desc.add_options()(
      "dlaf:bt-band-to-tridiag-hh-apply-group-size", pika::program_options::value<SizeType>(),
      "The application of the HH reflector is splitted in smaller applications of group size reflectors.");
  desc.add_options()(
      "dlaf:communicator-grid-num-pipelines", pika::program_options::value<std::size_t>(),
      "The default number of row, column, and full communicator pipelines to initialize in CommunicatorGrid.");

  return desc;
}

void initialize(const pika::program_options::variables_map& vm, const configuration& user_cfg) {
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
    std::cout << "DLA-Future tune parameters at startup:" << std::endl;
    std::cout << getTuneParameters() << std::endl;
    std::cout << std::endl;
  }

  if (should_exit) {
    std::exit(0);
  }

  int mpi_initialized;
  DLAF_MPI_CHECK_ERROR(MPI_Initialized(&mpi_initialized));
  if (mpi_initialized) {
    int provided;
    DLAF_MPI_CHECK_ERROR(MPI_Query_thread(&provided));
    if (provided < MPI_THREAD_MULTIPLE) {
      std::cerr << "MPI must be initialized to `MPI_THREAD_MULTIPLE` for DLA-Future!\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  DLAF_ASSERT(!internal::initialized(), "");
  internal::Init<Backend::MC>::initialize(cfg);
#ifdef DLAF_WITH_GPU
  internal::Init<Backend::GPU>::initialize(cfg);
#endif
  internal::initialized() = true;
}

void initialize(int argc, const char* const argv[], const configuration& user_cfg) {
  auto desc = getOptionsDescription();

  pika::program_options::variables_map vm;
  pika::program_options::store(pika::program_options::parse_command_line(argc, argv, desc), vm);
  pika::program_options::notify(vm);

  initialize(vm, user_cfg);
}

void finalize() {
  DLAF_ASSERT(internal::initialized(), "");
  internal::Init<Backend::MC>::finalize();
#ifdef DLAF_WITH_GPU
  internal::Init<Backend::GPU>::finalize();
#endif
  internal::getConfiguration() = {};
  internal::initialized() = false;
}

ScopedInitializer::ScopedInitializer(const pika::program_options::variables_map& vm,
                                     const configuration& user_cfg) {
  initialize(vm, user_cfg);
}

ScopedInitializer::ScopedInitializer(int argc, const char* const argv[], const configuration& user_cfg) {
  initialize(argc, argv, user_cfg);
}

ScopedInitializer::~ScopedInitializer() {
  finalize();
}

void initResourcePartitionerHandler(pika::resource::partitioner&,
                                    const pika::program_options::variables_map& vm) {
  namespace mpi = pika::mpi::experimental;
  // Create the MPI pool if needed and unless the user disabled it
  mpi::pool_create_mode pool_mode = mpi::pool_create_mode::pika_decides;
  namespace mpi = pika::mpi::experimental;
  if (vm["dlaf:no-mpi-pool"].as<bool>())
    pool_mode = mpi::pool_create_mode::force_no_create;

  namespace mpix = pika::mpi::experimental;
  // create a pool for mpi if necessary
  mpix::create_pool(mpix::get_pool_name(), pool_mode);
}
}
