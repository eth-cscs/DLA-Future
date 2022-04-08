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
#include "dlaf/matrix/matrix_mirror.h"
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
  SizeType b;

  Options(const pika::program_options::variables_map& vm)
      : MiniappOptions(vm), m(vm["matrix-size"].as<SizeType>()), mb(vm["block-size"].as<SizeType>()),
        b(vm["band-size"].as<SizeType>()) {
    DLAF_ASSERT(m > 0, m);
    DLAF_ASSERT(mb > 0, mb);

    if (b < 0)
      b = mb;
    DLAF_ASSERT(b > 0 && b <= mb, b, mb);

    DLAF_ASSERT(do_check == dlaf::miniapp::CheckIterFreq::None,
                "Error! At the moment result checking is not implemented. "
                "Please rerun with --check-result=none.");

    DLAF_ASSERT(backend == dlaf::Backend::MC ||
                    (vm["grid-rows"].as<int>() * vm["grid-cols"].as<int>()) == 1,
                "Error! At the moment the GPU backend is supported just with local runs. "
                "Please rerun with --backend=mc or with both --grid-rows and --grid-cols set to 1");
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
    using MatrixMirrorType = matrix::MatrixMirror<T, DefaultDevice_v<backend>, Device::CPU>;
    using HostMatrixType = Matrix<T, Device::CPU>;
    using ConstMatrixType = Matrix<const T, Device::CPU>;

    Communicator world(MPI_COMM_WORLD);
    CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, common::Ordering::ColumnMajor);

    // Allocate memory for the matrix
    const GlobalElementSize matrix_size(opts.m, opts.m);
    const TileElementSize block_size(opts.mb, opts.mb);

    ConstMatrixType matrix_ref = [matrix_size, block_size, comm_grid]() {
      using dlaf::matrix::util::set_random_hermitian;

      HostMatrixType hermitian(matrix_size, block_size, comm_grid);
      set_random_hermitian(hermitian);

      return hermitian;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType matrix_host(matrix_size, block_size, comm_grid);
      copy(matrix_ref, matrix_host);

      double elapsed_time;
      {
        MatrixMirrorType matrix_mirror(matrix_host);

        Matrix<T, DefaultDevice_v<backend>>& matrix = matrix_mirror.get();

	// wait all setup tasks before starting benchmark
	matrix_host.waitLocalTiles();
	DLAF_MPI_CALL(MPI_Barrier(world));

        dlaf::common::Timer<> timeit;
        auto taus = [&]() {
          if constexpr (Backend::GPU == backend)
            return dlaf::eigensolver::reductionToBand<backend>(matrix, opts.b);
          else
            return dlaf::eigensolver::reductionToBand<backend>(comm_grid, matrix);
        }();

	// wait and barrier for all ranks
	matrix.waitLocalTiles();
	DLAF_MPI_CALL(MPI_Barrier(world));

        elapsed_time = timeit.elapsed();
      }

      double gigaflops = std::numeric_limits<double>::quiet_NaN();
      {
        double n = matrix_host.size().rows();
        double b = matrix_host.blockSize().rows();
        auto add_mul = 2. / 3. * n * n * n - n * n * b;
        gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type} << " " << matrix_host.size() << " "
                  << matrix_host.blockSize() << " " << opts.b << " " << comm_grid.size() << " "
                  << pika::get_os_thread_count() << " " << backend << std::endl;

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
    ("band-size",    value<SizeType>()   ->default_value(-1),     "Band size")
  ;
  // clang-format on

  pika::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return pika::init(pika_main, argc, argv, p);
}
