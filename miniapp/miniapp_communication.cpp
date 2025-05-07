//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include <blas/util.hh>
#include <mpi.h>

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <dlaf/auxiliary/norm.h>
#include <dlaf/blas/tile.h>
#include <dlaf/common/format_short.h>
#include <dlaf/common/timer.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/error.h>
#include <dlaf/communication/init.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/communication/kernels/broadcast.h>
#include <dlaf/communication/kernels/internal/all_reduce.h>
#include <dlaf/communication/kernels/internal/broadcast.h>
#include <dlaf/communication/kernels/internal/p2p.h>
#include <dlaf/communication/kernels/internal/reduce.h>
#include <dlaf/communication/kernels/p2p.h>
#include <dlaf/communication/kernels/reduce.h>
#include <dlaf/init.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace {
using pika::execution::experimental::start_detached;

using dlaf::Backend;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::LocalElementSize;
using dlaf::LocalTileIndex;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::comm::Size2D;
using dlaf::common::iterate_range2d;
using dlaf::common::Ordering;
using dlaf::internal::RequireContiguous;
using dlaf::matrix::local_matrix;
using dlaf::matrix::MatrixAllocation;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};
}

template <Device D, class T>
Matrix<T, D> get_matrix_on_device_sync(Matrix<const T, Device::CPU>& matrix_ref) {
  DLAF_ASSERT(local_matrix(matrix_ref), matrix_ref);
  Matrix<T, D> matrix(matrix_ref.distribution(), MatrixAllocation::ColMajor);
  copy(matrix_ref, matrix);
  matrix.waitLocalTiles();

  return matrix;
}

std::string string_default(std::string name, Device D) {
  std::stringstream s;
  s << name << " " << D << " Default";
  return s.str();
}

std::string string_full(std::string name, Device D, Device DComm, RequireContiguous require_contiguous) {
  std::stringstream s;
  s << name << " " << D << ":" << DComm;
  if (require_contiguous == RequireContiguous::Yes)
    s << " contiguous    ";
  else
    s << " non-contiguous";
  return s.str();
}

std::string string_full(std::string name, Device D, Device DComm,
                        RequireContiguous require_contiguous_send,
                        RequireContiguous require_contiguous_recv) {
  std::stringstream s;
  s << name << " " << D << ":" << DComm;
  if (require_contiguous_send == RequireContiguous::Yes)
    s << " contiguous-send    ";
  else
    s << " non-contiguous-send";
  if (require_contiguous_recv == RequireContiguous::Yes)
    s << " contiguous-recv    ";
  else
    s << " non-contiguous-recv";
  return s.str();
}

void output(int64_t run_index, std::string name, double elapsed_time, int rank, const Options& opts,
            Size2D grid_size) {
  if (rank == 0 && run_index >= 0) {
    std::cout << "[" << run_index << "]"
              << " " << name << " " << elapsed_time << "s"
              << " " << dlaf::internal::FormatShort{opts.type} << " " << opts.m << " " << opts.mb << " "
              << grid_size << " " << pika::get_os_thread_count() << " " << opts.backend << std::endl;
  }
}

template <class CommPipeline, class T, Device D, class F>
void benchmark_rw(CommPipeline& pcomm, Matrix<T, D>& matrix, F&& schedule_comm_function) {
  for (auto& index : iterate_range2d(matrix.distribution().local_nr_tiles())) {
    start_detached(schedule_comm_function(pcomm.exclusive(), matrix.readwrite(index)));
  }
}
template <class CommPipeline, class T, Device D, class F>
void benchmark_ro(CommPipeline& pcomm, Matrix<T, D>& matrix, F&& schedule_comm_function) {
  for (auto& index : iterate_range2d(matrix.distribution().local_nr_tiles())) {
    start_detached(schedule_comm_function(pcomm.exclusive(), matrix.read(index)));
  }
}
template <class CommPipeline, class T, Device DIn, Device DOut, class F>
void benchmark_ro_rw(CommPipeline& pcomm, Matrix<T, DIn>& matrix_in, Matrix<T, DOut>& matrix_out,
                     F&& schedule_comm_function) {
  for (auto& index : iterate_range2d(matrix_out.distribution().local_nr_tiles())) {
    start_detached(schedule_comm_function(pcomm.exclusive(), matrix_in.read(index),
                                          matrix_out.readwrite(index)));
  }
}

template <Backend B, class CommPipeline, class T>
void benchmark_all_reduce(int64_t run_index, const Options& opts, Communicator& world,
                          CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::scheduleAllReduce;

  constexpr Device D = DefaultDevice_v<B>;
  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);
  Matrix<T, D> matrix_out = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  auto allred = [](auto comm, auto ro_tile, auto rw_tile) {
    return scheduleAllReduce(std::move(comm), MPI_SUM, std::move(ro_tile), std::move(rw_tile));
  };
  benchmark_ro_rw(pcomm, matrix, matrix_out, allred);
  matrix.waitLocalTiles();
  matrix_out.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_default("All reduce", D);
  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Device DComm, Backend B, class CommPipeline, class T>
void benchmark_internal_all_reduce(int64_t run_index, const Options& opts, Communicator& world,
                                   CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::internal::scheduleAllReduce;

  constexpr Device D = DefaultDevice_v<B>;
  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);
  Matrix<T, D> matrix_out = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  auto allred = [](auto comm, auto ro_tile, auto rw_tile) {
    return scheduleAllReduce<DComm, DComm>(std::move(comm), MPI_SUM, std::move(ro_tile),
                                           std::move(rw_tile));
  };
  benchmark_ro_rw(pcomm, matrix, matrix_out, allred);

  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_full("All reduce", D, DComm, RequireContiguous::Yes);

  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Backend B, class CommPipeline, class T>
void benchmark_all_reduce_in_place(int64_t run_index, const Options& opts, Communicator& world,
                                   CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::scheduleAllReduce;

  constexpr Device D = DefaultDevice_v<B>;
  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  auto allred = [](auto comm, auto rw_tile) {
    return schedule_all_reduce_in_place(std::move(comm), MPI_SUM, std::move(rw_tile));
  };
  benchmark_rw(pcomm, matrix, allred);

  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_default("All reduce (in place)", D);
  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Device DComm, Backend B, class CommPipeline, class T>
void benchmark_internal_all_reduce_in_place(int64_t run_index, const Options& opts, Communicator& world,
                                            CommPipeline&& pcomm,
                                            Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::internal::scheduleAllReduce;

  constexpr Device D = DefaultDevice_v<B>;
  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);
  Matrix<T, D> matrix_out = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  auto allred = [](auto comm, auto ro_tile, auto rw_tile) {
    return scheduleAllReduce<DComm, DComm>(std::move(comm), MPI_SUM, std::move(ro_tile),
                                           std::move(rw_tile));
  };
  benchmark_ro_rw(pcomm, matrix, matrix_out, allred);

  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_full("All reduce (in place)", D, DComm, RequireContiguous::Yes);

  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Backend B, class CommPipeline, class T>
void benchmark_broadcast(int64_t run_index, const Options& opts, Communicator& world,
                         CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::schedule_bcast_recv;
  using dlaf::comm::schedule_bcast_send;

  constexpr Device D = DefaultDevice_v<B>;
  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  if (world.rank() == 0) {
    auto bcast = [](auto comm, auto ro_tile) {
      return schedule_bcast_send(std::move(comm), std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, bcast);
  }
  else {
    auto bcast = [](auto comm, auto rw_tile) {
      return schedule_bcast_recv(std::move(comm), 0, std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, bcast);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_default("Broadcast", D);
  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Device DComm, RequireContiguous require_contiguous_send,
          RequireContiguous require_contiguous_recv, Backend B, class CommPipeline, class T>
void benchmark_internal_broadcast(int64_t run_index, const Options& opts, Communicator& world,
                                  CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::internal::schedule_bcast_recv;
  using dlaf::comm::internal::schedule_bcast_send;

  constexpr Device D = DefaultDevice_v<B>;
  if constexpr (DComm != D && (require_contiguous_send != RequireContiguous::Yes ||
                               require_contiguous_recv != RequireContiguous::Yes)) {
    // Skip benchmark, as copy to a different device forces contiguous buffers.
    return;
  }

  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  if (world.rank() == 0) {
    auto bcast = [](auto comm, auto ro_tile) {
      return schedule_bcast_send<DComm, require_contiguous_send>(std::move(comm), std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, bcast);
  }
  else {
    auto bcast = [](auto comm, auto rw_tile) {
      return schedule_bcast_recv<DComm, require_contiguous_recv>(std::move(comm), 0, std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, bcast);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_full("Broadcast", D, DComm, require_contiguous_send, require_contiguous_recv);

  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Backend B, class CommPipeline, class T>
void benchmark_p2p(int64_t run_index, const Options& opts, Communicator& world, CommPipeline&& pcomm,
                   Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::schedule_recv;
  using dlaf::comm::schedule_send;

  constexpr Device D = DefaultDevice_v<B>;
  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  const auto rank_recv = pcomm.size() - 1;

  if (pcomm.rank() == 0) {
    auto p2p = [rank_recv](auto comm, auto ro_tile) {
      return schedule_send(std::move(comm), rank_recv, 0, std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, p2p);
  }
  else if (pcomm.rank() == rank_recv) {
    auto p2p = [](auto comm, auto rw_tile) {
      return schedule_recv(std::move(comm), 0, 0, std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, p2p);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_default("P2P", D);
  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Device DComm, RequireContiguous require_contiguous_send,
          RequireContiguous require_contiguous_recv, Backend B, class CommPipeline, class T>
void benchmark_internal_p2p(int64_t run_index, const Options& opts, Communicator& world,
                            CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::internal::schedule_recv;
  using dlaf::comm::internal::schedule_send;

  constexpr Device D = DefaultDevice_v<B>;
  if constexpr (DComm != D && (require_contiguous_send != RequireContiguous::Yes ||
                               require_contiguous_recv != RequireContiguous::Yes)) {
    // Skip benchmark, as copy to a different device forces contiguous buffers.
    return;
  }

  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  const auto rank_recv = pcomm.size() - 1;

  if (pcomm.rank() == 0) {
    auto p2p = [rank_recv](auto comm, auto ro_tile) {
      return schedule_send<DComm, require_contiguous_send>(std::move(comm), rank_recv, 0,
                                                           std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, p2p);
  }
  else if (pcomm.rank() == rank_recv) {
    auto p2p = [](auto comm, auto rw_tile) {
      return schedule_recv<DComm, require_contiguous_recv>(std::move(comm), 0, 0, std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, p2p);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_full("P2P", D, DComm, require_contiguous_send, require_contiguous_recv);

  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Backend B, class CommPipeline, class T>
void benchmark_reduce(int64_t run_index, const Options& opts, Communicator& world, CommPipeline&& pcomm,
                      Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::schedule_reduce_recv_in_place;
  using dlaf::comm::schedule_reduce_send;

  constexpr Device D = DefaultDevice_v<B>;
  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  if (world.rank() == 0) {
    auto reduce = [](auto comm, auto rw_tile) {
      return schedule_reduce_recv_in_place(std::move(comm), MPI_SUM, std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, reduce);
  }
  else {
    auto reduce = [](auto comm, auto ro_tile) {
      return schedule_reduce_send(std::move(comm), 0, MPI_SUM, std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, reduce);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_default("Reduce", D);
  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

template <Device DComm, Backend B, class CommPipeline, class T>
void benchmark_internal_reduce(int64_t run_index, const Options& opts, Communicator& world,
                               CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::internal::schedule_reduce_recv_in_place;
  using dlaf::comm::internal::schedule_reduce_send;

  constexpr Device D = DefaultDevice_v<B>;

  Matrix<T, D> matrix = get_matrix_on_device_sync<D>(matrix_ref);

  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  dlaf::common::Timer<> timeit;

  if (world.rank() == 0) {
    auto reduce = [](auto comm, auto rw_tile) {
      return schedule_reduce_recv_in_place<DComm>(std::move(comm), MPI_SUM, std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, reduce);
  }
  else {
    auto reduce = [](auto comm, auto ro_tile) {
      return schedule_reduce_send<DComm>(std::move(comm), 0, MPI_SUM, std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, reduce);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
  auto t = timeit.elapsed();

  std::string s = string_full("Reduce", D, DComm, RequireContiguous::Yes, RequireContiguous::Yes);

  output(run_index, std::move(s), t, world.rank(), opts, pcomm.size_2d());
}

struct communicationMiniapp {
  template <Backend B, typename T>
  static void run(const Options& opts) {
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstHostMatrixType = Matrix<const T, Device::CPU>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    if (opts.local || world.size() == 1) {
      std::cout << "Single rank. Nothing to measure." << std::endl;
      return;
    }

    // Allocate memory for the matrix
    LocalElementSize matrix_size(opts.m, opts.m);
    TileElementSize block_size(opts.mb, opts.mb);

    ConstHostMatrixType matrix_ref = [matrix_size, block_size]() {
      using dlaf::matrix::util::set_random;

      HostMatrixType random(matrix_size, block_size);
      set_random(random);

      return random;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      benchmark_all_reduce<B>(run_index, opts, world, comm_grid.full_communicator_pipeline(),
                              matrix_ref);
      benchmark_internal_all_reduce<Device::CPU, B>(run_index, opts, world,
                                                    comm_grid.full_communicator_pipeline(), matrix_ref);
#ifdef DLAF_WITH_MPI_GPU_AWARE
      benchmark_internal_all_reduce<Device::GPU, B>(run_index, opts, world,
                                                    comm_grid.full_communicator_pipeline(), matrix_ref);
#endif

      benchmark_all_reduce_in_place<B>(run_index, opts, world, comm_grid.full_communicator_pipeline(),
                                       matrix_ref);
      benchmark_internal_all_reduce_in_place<Device::CPU, B>(run_index, opts, world,
                                                             comm_grid.full_communicator_pipeline(),
                                                             matrix_ref);
#ifdef DLAF_WITH_MPI_GPU_AWARE
      benchmark_internal_all_reduce_in_place<Device::GPU, B>(run_index, opts, world,
                                                             comm_grid.full_communicator_pipeline(),
                                                             matrix_ref);
#endif

      benchmark_broadcast<B>(run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::No, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::Yes, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::No, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::Yes, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
#ifdef DLAF_WITH_MPI_GPU_AWARE
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::No, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::Yes, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::No, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::Yes, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
#endif

      benchmark_p2p<B>(run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_p2p<Device::CPU, RequireContiguous::No, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_p2p<Device::CPU, RequireContiguous::Yes, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_p2p<Device::CPU, RequireContiguous::No, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_p2p<Device::CPU, RequireContiguous::Yes, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
#ifdef DLAF_WITH_MPI_GPU_AWARE
      benchmark_internal_p2p<Device::GPU, RequireContiguous::No, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_p2p<Device::GPU, RequireContiguous::Yes, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_p2p<Device::GPU, RequireContiguous::No, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_p2p<Device::GPU, RequireContiguous::Yes, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
#endif

      benchmark_reduce<B>(run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_reduce<Device::CPU, B>(run_index, opts, world,
                                                comm_grid.full_communicator_pipeline(), matrix_ref);
#ifdef DLAF_WITH_MPI_GPU_AWARE
      benchmark_internal_reduce<Device::GPU, B>(run_index, opts, world,
                                                comm_grid.full_communicator_pipeline(), matrix_ref);
#endif
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  pika::scoped_finalize pika_finalizer;
  dlaf::ScopedInitializer init(vm);

  const Options opts(vm);
  dlaf::miniapp::dispatchMiniapp<communicationMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_communication [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size", value<SizeType>()   ->default_value(4096), "Matrix size")
    ("block-size",  value<SizeType>()   ->default_value( 256), "Block cyclic distribution size")
  ;
  // clang-format on

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  return pika::init(pika_main, argc, argv, p);
}
