//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include "dlaf/common/format_short.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/eigensolver/bt_band_to_tridiag.h"
#include "dlaf/init.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/miniapp/dispatch.h"
#include "dlaf/miniapp/options.h"
#include "dlaf/types.h"

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
using dlaf::matrix::MatrixMirror;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType n;
  SizeType mb;
  SizeType nb;
  SizeType b;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["m"].as<SizeType>()), n(vm["n"].as<SizeType>()),
        mb(vm["mb"].as<SizeType>()), nb(vm["nb"].as<SizeType>()), b(vm["b"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);
    DLAF_ASSERT(b > 0 && mb % b == 0, b, mb);

    DLAF_ASSERT(grid_rows * grid_cols == 1,
                "Error! Distributed is not avilable yet. "
                "Please rerun with both --grid-rows and --grid-cols set to 1");

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

struct BacktransformBandToTridiagMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    using MatrixMirrorType = MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU>;
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstHostMatrixType = Matrix<const T, Device::CPU>;

    if (opts.grid_rows * opts.grid_cols != 1)
      DLAF_UNIMPLEMENTED("Distributed implementation not available yet.");

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Allocate memory for the matrix
    GlobalElementSize matrix_size(opts.m, opts.n);
    TileElementSize block_size(opts.mb, opts.nb);

    ConstHostMatrixType matrix_ref = [matrix_size, block_size, comm_grid]() {
      using dlaf::matrix::util::set_random;

      HostMatrixType random(matrix_size, block_size, comm_grid);
      set_random(random);

      return random;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType mat_e_host(matrix_size, block_size, comm_grid);
      copy(matrix_ref, mat_e_host);

      HostMatrixType mat_hh({opts.m, opts.m}, {opts.mb, opts.mb}, comm_grid);

      double elapsed_time;
      {
        MatrixMirrorType mat_e(mat_e_host);

        // Wait for matrices to be copied to GPU (if necessary)
        mat_e.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        dlaf::common::Timer<> timeit;
        dlaf::eigensolver::backTransformationBandToTridiag<backend, DefaultDevice_v<backend>,
                                                           T>(opts.b, mat_e.get(), mat_hh);

        // wait and barrier for all ranks
        mat_e.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        elapsed_time = timeit.elapsed();
      }

      double gigaflops;
      {
        const double m = mat_e_host.size().rows();
        const double n = mat_e_host.size().cols();
        auto add_mul = m * m * n;
        gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type} << " " << mat_e_host.size() << " "
                  << mat_e_host.blockSize() << " " << opts.b << " " << comm_grid.size() << " "
                  << pika::get_os_thread_count() << " " << backend << std::endl;

      // (optional) run test
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

  dlaf::miniapp::dispatchMiniapp<BacktransformBandToTridiagMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_bt_band_to_tridiag [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("m",   value<SizeType>()   ->default_value(2048), "Matrix E rows")
    ("n",   value<SizeType>()   ->default_value(4096), "Matrix E columns")
    ("mb",  value<SizeType>()   ->default_value( 256), "Matrix E block rows")
    ("nb",  value<SizeType>()   ->default_value( 512), "Matrix E block columns")
    ("b",   value<SizeType>()   ->default_value(  64), "Band size")
  ;
  // clang-format on
  dlaf::miniapp::addUploOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}
