//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <blas/util.hh>
#include <iostream>

#include <mpi.h>
#include <pika/future.hpp>
#include <pika/init.hpp>
#include <pika/program_options.hpp>
#include <pika/runtime.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/error.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/mech.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/eigensolver/gen_to_std.h"
#include "dlaf/init.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/miniapp/dispatch.h"
#include "dlaf/miniapp/options.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace {

using dlaf::Device;
using dlaf::Coord;
using dlaf::Backend;
using dlaf::DefaultDevice_v;
using dlaf::SizeType;
using dlaf::comm::Index2D;
using dlaf::GlobalElementSize;
using dlaf::TileElementSize;
using dlaf::Matrix;
using dlaf::matrix::MatrixMirror;
using dlaf::common::Ordering;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;

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

struct GenToStdMiniapp {
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

    ConstHostMatrixType matrix_a_ref = [matrix_size, block_size, comm_grid]() {
      using dlaf::matrix::util::set_random_hermitian;

      HostMatrixType hermitian(matrix_size, block_size, comm_grid);
      set_random_hermitian(hermitian);

      return hermitian;
    }();

    ConstHostMatrixType matrix_b_ref = [matrix_size, block_size, comm_grid]() {
      // As the result of the Cholesky decomposition is a triangular matrix with
      // strictly poitive real elements on the diagonal, and as only the upper/lower
      // part of the tridiagonal is referenced, it is fine to use
      // set_random_hermitian_positive_definite to set the triangular factor.
      using dlaf::matrix::util::set_random_hermitian_positive_definite;

      HostMatrixType triangular(matrix_size, block_size, comm_grid);
      set_random_hermitian_positive_definite(triangular);

      return triangular;
    }();

    for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]" << std::endl;

      HostMatrixType matrix_a_host(matrix_size, block_size, comm_grid);
      HostMatrixType matrix_b_host(matrix_size, block_size, comm_grid);
      copy(matrix_a_ref, matrix_a_host);
      copy(matrix_b_ref, matrix_b_host);

      double elapsed_time;
      {
        MatrixMirrorType matrix_a(matrix_a_host);
        MatrixMirrorType matrix_b(matrix_b_host);

        // Wait all setup tasks and (if necessary) for matrix to be copied to GPU.
        matrix_a.get().waitLocalTiles();
        matrix_b.get().waitLocalTiles();
        DLAF_MPI_CALL(MPI_Barrier(world));

        dlaf::common::Timer<> timeit;
        dlaf::eigensolver::genToStd<Backend::Default, Device::Default, T>(comm_grid, opts.uplo,
                                                                          matrix_a.get(),
                                                                          matrix_b.get());

        // wait and barrier for all ranks
        matrix_a.get().waitLocalTiles();
        DLAF_MPI_CALL(MPI_Barrier(world));
        elapsed_time = timeit.elapsed();
      }

      double gigaflops;
      {
        double n = matrix_a_host.size().rows();
        auto add_mul = n * n * n / 2;
        gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
      }

      // print benchmark results
      if (0 == world.rank() && run_index >= 0)
        std::cout << "[" << run_index << "]"
                  << " " << elapsed_time << "s"
                  << " " << gigaflops << "GFlop/s"
                  << " " << dlaf::internal::FormatShort{opts.type}
                  << dlaf::internal::FormatShort{opts.uplo} << " " << matrix_a_host.size() << " "
                  << matrix_a_host.blockSize() << " " << comm_grid.size() << " "
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
  {
    dlaf::ScopedInitializer init(vm);
    const Options opts(vm);

    dlaf::miniapp::dispatchMiniapp<GenToStdMiniapp>(opts);
  }

  return pika::finalize();
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::multiple);

  // options
  using namespace pika::program_options;
  options_description desc_commandline("Usage: miniapp_gen_to_std.cpp [options]");
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
