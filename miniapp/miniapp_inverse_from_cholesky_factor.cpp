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
#include <string>

#include <blas/util.hh>
#include <mpi.h>

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include <dlaf/common/format_short.h>
#include <dlaf/common/timer.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/init.h>
#include <dlaf/init.h>
#include <dlaf/inverse.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/miniapp/dispatch.h>
#include <dlaf/miniapp/options.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace {

using dlaf::Backend;
using dlaf::DefaultDevice_v;
using dlaf::Device;
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;
using dlaf::matrix::MatrixMirror;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;
  blas::Uplo uplo;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()),
        uplo(dlaf::miniapp::parseUplo(vm["uplo"].as<std::string>())) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);

    if (do_check != dlaf::miniapp::CheckIterFreq::None) {
      std::cerr << "Warning! At the moment result checking it is not implemented." << std::endl;
      do_check = dlaf::miniapp::CheckIterFreq::None;
    }
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};
}

struct InverseFromCholeskyFactorMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    using MatrixMirrorType = MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU>;
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstHostMatrixType = Matrix<const T, Device::CPU>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Allocate memory for the matrix
    GlobalElementSize matrix_size(opts.m, opts.m);
    TileElementSize block_size(opts.mb, opts.mb);

    ConstHostMatrixType matrix_ref = [matrix_size, block_size, &comm_grid]() {
      using dlaf::matrix::util::set_random_non_zero_diagonal;

      HostMatrixType non_zero_diag(matrix_size, block_size, comm_grid);
      set_random_non_zero_diagonal(non_zero_diag);

      return non_zero_diag;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType matrix_host(matrix_size, block_size, comm_grid);
      copy(matrix_ref, matrix_host);

      double elapsed_time;
      {
        MatrixMirrorType matrix(matrix_host);

        // Wait for matrix to be copied to GPU (if necessary)
        matrix.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        dlaf::common::Timer<> timeit;
        if (opts.local)
          dlaf::inverse_from_cholesky_factor<backend, DefaultDevice_v<backend>, T>(opts.uplo,
                                                                                   matrix.get());
        else
          dlaf::inverse_from_cholesky_factor<backend, DefaultDevice_v<backend>, T>(comm_grid, opts.uplo,
                                                                                   matrix.get());

        // wait and barrier for all ranks
        matrix.get().waitLocalTiles();
        comm_grid.wait_all_communicators();

        elapsed_time = timeit.elapsed();
      }

      double gigaflops;
      {
        double n = matrix_host.size().rows();
        auto add_mul = n * n * n * 2 / 3;
        gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0) {
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type}
                  << dlaf::internal::FormatShort{opts.uplo} << " " << matrix_host.size() << " "
                  << matrix_host.blockSize() << " " << comm_grid.size() << " "
                  << pika::get_os_thread_count() << " " << backend << std::endl;
        if (opts.csv_output) {
          // CSV formatted output with column names that can be read by pandas to simplify
          // post-processing CSVData{-version}, value_0, title_0, value_1, title_1
          std::cout << "CSVData-2, "
                    << "run, " << run_index << ", "
                    << "time, " << elapsed_time << ", "
                    << "GFlops, " << gigaflops << ", "
                    << "type, " << dlaf::internal::FormatShort{opts.type}.value << ", "
                    << "uplo, " << dlaf::internal::FormatShort{opts.uplo}.value << ", "
                    << "matrixsize, " << matrix_host.size().rows() << ", "
                    << "blocksize, " << block_size.rows() << ", "
                    << "comm_rows, " << comm_grid.size().rows() << ", "
                    << "comm_cols, " << comm_grid.size().cols() << ", "
                    << "threads, " << pika::get_os_thread_count() << ", "
                    << "backend, " << backend << ", " << opts.info << std::endl;
        }
      }

      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        DLAF_UNIMPLEMENTED("Check");
      }
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  pika::scoped_finalize pika_finalizer;
  dlaf::ScopedInitializer init(vm);

  const Options opts(vm);
  dlaf::miniapp::dispatchMiniapp<InverseFromCholeskyFactorMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_inverse_from_cholesky_factor [options]");

  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size", value<SizeType>()   ->default_value(4096), "Matrix size")
    ("block-size",  value<SizeType>()   ->default_value( 256), "Block cyclic distribution size")
  ;
  // clang-format on
  dlaf::miniapp::addUploOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  return pika::init(pika_main, argc, argv, p);
}
