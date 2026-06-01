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

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <dlaf/common/format_short.h>
#include <dlaf/common/timer.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/init.h>
#include <dlaf/init.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/types.h>

namespace {
using dlaf::Backend;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::GlobalElementSize;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;
  SizeType mb_dst;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()),
        mb_dst(vm["block-size-dst"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);
    DLAF_ASSERT(mb <= m, mb, m);
    DLAF_ASSERT(mb_dst > 0, mb_dst);
    DLAF_ASSERT(mb_dst <= m, mb_dst, m);
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};
}

struct RedistributionMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    constexpr Device device = DefaultDevice_v<backend>;

    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstHostMatrixType = Matrix<const T, Device::CPU>;
    using DeviceMatrixType = Matrix<T, device>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Create random reference matrix on the host
    ConstHostMatrixType matrix_ref = [&comm_grid, &opts]() {
      const GlobalElementSize matrix_size(opts.m, opts.m);
      const TileElementSize block_size_src(opts.mb, opts.mb);

      using dlaf::matrix::util::set_random;

      HostMatrixType random(matrix_size, block_size_src, comm_grid);
      set_random(random);

      return random;
    }();

    const auto matrix_size = matrix_ref.size();
    const auto block_size_src = matrix_ref.blockSize();
    const TileElementSize block_size_dst(opts.mb_dst, opts.mb_dst);

    const auto& dist_src = matrix_ref.distribution();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType matrix_src(matrix_size, block_size_src, comm_grid);
      copy(matrix_ref, matrix_src);

      matrix_src.waitLocalTiles();
      DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
      dlaf::common::Timer<> timeit;

      if (block_size_src != block_size_dst) {
        // Create destination matrix on device with different block size
        dlaf::matrix::Distribution dist_dst(matrix_size, block_size_dst, dist_src.grid_size(),
                                            dist_src.rank_index(), dist_src.source_rank_index());
        DeviceMatrixType matrix_dst(dist_dst);

        // Copy matrix from source to destination
        dlaf::matrix::copy(matrix_src, matrix_dst, comm_grid);
        matrix_dst.waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        // Copy matrix from destination back to source
        dlaf::matrix::copy(matrix_dst, matrix_src, comm_grid);
      }
      else {
        dlaf::matrix::MatrixMirror<const T, device, Device::CPU> matrix_dst(matrix_src);
        matrix_dst.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
      }

      matrix_src.waitLocalTiles();
      DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
      double elapsed_time = timeit.elapsed();

      // print benchmark results
      if (0 == world.rank() && run_index >= 0) {
        std::cout << "[" << run_index << "]" << " " << elapsed_time << "s" << " "
                  << dlaf::internal::FormatShort{opts.type} << " " << matrix_size << " "
                  << block_size_src << " -> " << block_size_dst << " " << comm_grid.size() << " "
                  << pika::get_os_thread_count() << " " << backend << std::endl;
        if (opts.csv_output) {
          // CSV formatted output with column names that can be read by pandas to simplify
          // post-processing CSVData{-version}, value_0, title_0, value_1, title_1
          std::cout << "CSVData-2, "
                    << "run, " << run_index << ", "
                    << "time, " << elapsed_time << ", "
                    << "type, " << dlaf::internal::FormatShort{opts.type}.value << ", "
                    << "matrixsize, " << matrix_size.rows() << ", "
                    << "blocksize_src, " << block_size_src.rows() << ", "
                    << "blocksize_dst, " << block_size_dst.rows() << ", "
                    << "comm_rows, " << comm_grid.size().rows() << ", "
                    << "comm_cols, " << comm_grid.size().cols() << ", "
                    << "threads, " << pika::get_os_thread_count() << ", "
                    << "backend, " << backend << ", " << opts.info << std::endl;
        }
      }
    }

    comm_grid.wait_all_communicators();
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  pika::scoped_finalize pika_finalizer;
  dlaf::ScopedInitializer init(vm);

  const Options opts(vm);
  dlaf::miniapp::dispatchMiniapp<RedistributionMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_redistribution [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",     value<SizeType>() ->default_value(4096), "Matrix size")
    ("block-size",      value<SizeType>() ->default_value(  64), "Source block cyclic distribution size")
    ("block-size-dst",  value<SizeType>() ->default_value(1024), "Destination block cyclic distribution size")
  ;
  // clang-format on

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  return pika::init(pika_main, argc, argv, p);
}
