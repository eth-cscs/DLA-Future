//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdlib>
#include <iostream>

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
#include <dlaf/communication/kernels/broadcast.h>
#include <dlaf/communication/kernels/internal/broadcast.h>
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
using dlaf::GlobalElementSize;
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

template <Backend B, class CommPipeline, class T>
void benchmark_broadcast(int64_t run_index, const Options& opts, Communicator& world,
                         CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::scheduleRecvBcast;
  using dlaf::comm::scheduleSendBcast;
  Matrix<T, DefaultDevice_v<B>> matrix(matrix_ref.distribution());  // TODO Fix Scalapack layout.
  copy(matrix_ref, matrix);
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  dlaf::common::Timer<> timeit;

  if (world.rank() == 0) {
    auto bcast = [](auto comm, auto ro_tile) {
      return scheduleSendBcast(std::move(comm), std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, bcast);
  }
  else {
    auto bcast = [](auto comm, auto rw_tile) {
      return scheduleRecvBcast(std::move(comm), 0, std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, bcast);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  output(run_index, "Broadcast Default", timeit.elapsed(), world.rank(), opts, pcomm.size_2d());
}

template <Device comm_device, RequireContiguous require_contiguous_send,
          RequireContiguous require_contiguous_recv, Backend B, class CommPipeline, class T>
void benchmark_internal_broadcast(int64_t run_index, const Options& opts, Communicator& world,
                                  CommPipeline&& pcomm, Matrix<const T, Device::CPU>& matrix_ref) {
  using dlaf::comm::internal::scheduleRecvBcast;
  using dlaf::comm::internal::scheduleSendBcast;
  Matrix<T, DefaultDevice_v<B>> matrix(matrix_ref.distribution());  // TODO Fix Scalapack layout.
  copy(matrix_ref, matrix);
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  dlaf::common::Timer<> timeit;

  if (world.rank() == 0) {
    auto bcast = [](auto comm, auto ro_tile) {
      return scheduleSendBcast<comm_device, require_contiguous_send>(std::move(comm),
                                                                     std::move(ro_tile));
    };
    benchmark_ro(pcomm, matrix, bcast);
  }
  else {
    auto bcast = [](auto comm, auto rw_tile) {
      return scheduleRecvBcast<comm_device, require_contiguous_recv>(std::move(comm), 0,
                                                                     std::move(rw_tile));
    };
    benchmark_rw(pcomm, matrix, bcast);
  }
  matrix.waitLocalTiles();
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  std::stringstream s;
  s << "Broadcast " << DefaultDevice_v<B> << ":" << comm_device;
  if (require_contiguous_send == RequireContiguous::Yes)
    s << " contiguous-send    ";
  else
    s << " non-contiguous-send";
  if (require_contiguous_recv == RequireContiguous::Yes)
    s << " contiguous-recv    ";
  else
    s << " non-contiguous-recv";

  output(run_index, s.str(), timeit.elapsed(), world.rank(), opts, pcomm.size_2d());
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
    GlobalElementSize matrix_size(opts.m, opts.m);
    TileElementSize block_size(opts.mb, opts.mb);

    ConstHostMatrixType matrix_ref = [matrix_size, block_size, &comm_grid]() {
      using dlaf::matrix::util::set_random;

      HostMatrixType random(matrix_size, block_size, comm_grid);  // TODO Fix Scalapack layout.
      set_random(random);

      return random;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      benchmark_broadcast<B>(run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::No, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::Yes, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::No, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::CPU, RequireContiguous::Yes, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
#ifdef DLAF_WITH_MPI_GPU_SUPPORT
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::No, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::Yes, RequireContiguous::No, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::No, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
      benchmark_internal_broadcast<Device::GPU, RequireContiguous::Yes, RequireContiguous::Yes, B>(
          run_index, opts, world, comm_grid.full_communicator_pipeline(), matrix_ref);
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
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}
