//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstdlib>
#include <iostream>
#include <limits>

#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>

#include "dlaf/common/format_short.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/eigensolver/tridiag_solver.h"
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
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::LocalElementSize;
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;
using dlaf::matrix::Distribution;
using dlaf::matrix::MatrixMirror;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::No> {
  SizeType m;
  SizeType mb;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()) {
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

struct TridiagSolverMiniapp {
  template <Backend backend, typename T>
  static void run(const Options& opts) {
    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

    // Allocate the tridiagonal, eigenvalues and eigenvectors matrices
    const Distribution dist_trd(LocalElementSize(opts.m, 2), TileElementSize(opts.mb, 2));
    const Distribution dist_evals(LocalElementSize(opts.m, 1), TileElementSize(opts.mb, 1));
    const Distribution dist_evecs(GlobalElementSize(opts.m, opts.m), TileElementSize(opts.mb, opts.mb),
                                  comm_grid.size(), comm_grid.rank(), {0, 0});

    Matrix<T, Device::CPU> tridiag(dist_trd);
    Matrix<T, Device::CPU> evals(dist_evals);
    Matrix<T, Device::CPU> evecs(dist_evecs);

    Matrix<const T, Device::CPU> tridiag_ref = [dist_trd]() {
      Matrix<T, Device::CPU> tridiag(dist_trd);
      dlaf::matrix::util::set_random(tridiag);
      return tridiag;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      copy(tridiag_ref, tridiag);

      double elapsed_time;
      {
        MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU> tridiag_mirror(tridiag);
        MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU> evals_mirror(evals);
        MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU> evecs_mirror(evecs);

        // Wait for matrix to be copied to GPU (if necessary)
        tridiag_mirror.get().waitLocalTiles();
        evals_mirror.get().waitLocalTiles();
        evecs_mirror.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

        dlaf::common::Timer<> timeit;
        using dlaf::eigensolver::tridiagSolver;
        if (opts.local)
          tridiagSolver<backend>(tridiag_mirror.get(), evals_mirror.get(), evecs_mirror.get());
        else
          tridiagSolver<backend>(comm_grid, tridiag_mirror.get(), evals_mirror.get(),
                                 evecs_mirror.get());

        // wait and barrier for all ranks
        tridiag_mirror.get().waitLocalTiles();
        evals_mirror.get().waitLocalTiles();
        evecs_mirror.get().waitLocalTiles();
        DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));
        elapsed_time = timeit.elapsed();
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << dlaf::internal::FormatShort{opts.type} << " " << tridiag.size() << " "
                  << tridiag.blockSize() << " " << comm_grid.size() << " " << pika::get_os_thread_count()
                  << " " << backend << std::endl;

      // (optional) run test
      if ((opts.do_check == dlaf::miniapp::CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
          opts.do_check == dlaf::miniapp::CheckIterFreq::All) {
        // TODO implement check
        DLAF_UNIMPLEMENTED("Check");
      }
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  pika::scoped_finalize pika_finalizer;
  dlaf::ScopedInitializer init(vm);

  const Options opts(vm);
  dlaf::miniapp::dispatchMiniapp<TridiagSolverMiniapp>(opts);

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_tridiag_solver [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",  value<SizeType>()   ->default_value(4096), "Matrix size")
    ("block-size",   value<SizeType>()   ->default_value( 256), "Block cyclic distribution size")
  ;
  // clang-format on
  dlaf::miniapp::addUploOption(desc_commandline);

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}
