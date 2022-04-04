//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

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
#include "dlaf/eigensolver/reduction_to_band.h"
#include "dlaf/init.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/miniapp/dispatch.h"
#include "dlaf/miniapp/options.h"
#include "dlaf/types.h"

namespace {
using dlaf::Device;
using dlaf::SizeType;

struct Options
    : dlaf::miniapp::MiniappOptions<dlaf::miniapp::SupportReal::Yes, dlaf::miniapp::SupportComplex::Yes> {
  SizeType m;
  SizeType mb;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);

    DLAF_ASSERT(do_check == dlaf::miniapp::CheckIterFreq::None,
                "Error! At the moment result checking is not implemented. Please rerun with --check-result=none.");

    DLAF_ASSERT(backend == dlaf::Backend::MC,
                "Error! At the moment the GPU backend is not supported. Please rerun with --backend=mc.");
  }

  Options(Options&&) = default;
  Options(const Options&) = default;
  Options& operator=(Options&&) = default;
  Options& operator=(const Options&) = default;
};

}

struct reductionToBandMiniapp {
  template <dlaf::Backend backend, typename T>
  static void run(const Options& opts) {
    using namespace dlaf;
    using dlaf::SizeType;
    using dlaf::comm::Communicator;
    using dlaf::comm::CommunicatorGrid;
    using MatrixType = dlaf::Matrix<T, Device::CPU>;
    using ConstMatrixType = dlaf::Matrix<const T, Device::CPU>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, common::Ordering::ColumnMajor);

    // Allocate memory for the matrix
    GlobalElementSize matrix_size(opts.m, opts.m);
    TileElementSize block_size(opts.mb, opts.mb);

    ConstMatrixType matrix_ref = [matrix_size, block_size, comm_grid]() {
      using dlaf::matrix::util::set_random_hermitian;

      MatrixType hermitian(matrix_size, block_size, comm_grid);
      set_random_hermitian(hermitian);

      return hermitian;
    }();

    const auto& distribution = matrix_ref.distribution();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      MatrixType matrix(matrix_size, block_size, comm_grid);
      copy(matrix_ref, matrix);

      // wait all setup tasks before starting benchmark
      matrix.waitLocalTiles();
      DLAF_MPI_CALL(MPI_Barrier(world));

      dlaf::common::Timer<> timeit;
      auto taus = dlaf::eigensolver::reductionToBand<dlaf::Backend::MC>(comm_grid, matrix);

      // wait for last task and barrier for all ranks
      {
        GlobalTileIndex last_tile(matrix.nrTiles().rows() - 1, matrix.nrTiles().cols() - 2);
        if (matrix.rankIndex() == distribution.rankGlobalTile(last_tile))
          pika::this_thread::experimental::sync_wait(matrix.readwrite_sender(last_tile));

        DLAF_MPI_CALL(MPI_Barrier(world));
      }
      auto elapsed_time = timeit.elapsed();

      double gigaflops = std::numeric_limits<double>::quiet_NaN();
      {
        double n = matrix.size().rows();
        double b = matrix.blockSize().rows();
        auto add_mul = 2. / 3. * n * n * n - n * n * b;
        gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type} << " " << matrix.size() << " "
                  << matrix.blockSize() << " " << comm_grid.size() << " " << pika::get_os_thread_count()
                  << " " << backend << std::endl;

      // TODO (optional) run test
    }
  }
};

int pika_main(pika::program_options::variables_map& vm) {
  {
    dlaf::ScopedInitializer init(vm);
    const Options opts(vm);

    dlaf::miniapp::dispatchMiniapp<reductionToBandMiniapp>(opts);
  }

  return pika::finalize();
}

int main(int argc, char** argv) {
  using dlaf::SizeType;

  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::multiple);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_reduction_to_band [options]");
  desc_commandline.add(dlaf::miniapp::getMiniappOptionsDescription());
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",  value<SizeType>()   ->default_value(4),      "Matrix rows")
    ("block-size",   value<SizeType>()   ->default_value(2),      "Block cyclic distribution size")
  ;
  // clang-format on

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}
