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
#include <hpx/init.hpp>
#include <hpx/local/runtime.hpp>
#include <hpx/program_options.hpp>

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
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace {

using dlaf::Device;
using dlaf::Coord;
using dlaf::Backend;
using dlaf::SizeType;
using dlaf::comm::Index2D;
using dlaf::GlobalElementSize;
using dlaf::TileElementSize;
using dlaf::common::Ordering;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;

using T = double;
using MatrixType = dlaf::Matrix<T, Device::Default>;
using HostMatrixType = dlaf::Matrix<T, Device::CPU>;
using ConstHostMatrixType = dlaf::Matrix<const T, Device::CPU>;
using MatrixMirrorType = dlaf::matrix::MatrixMirror<T, Device::Default, Device::CPU>;

enum class CheckIterFreq { None, Last, All };

struct options_t {
  SizeType m;
  SizeType mb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  int64_t nwarmups;
  CheckIterFreq do_check;
};

/// Handle CLI options
options_t parse_options(hpx::program_options::variables_map&);

CheckIterFreq parse_check(const std::string&);

}

int hpx_main(hpx::program_options::variables_map& vm) {
  dlaf::initialize(vm);
  options_t opts = parse_options(vm);

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

      // wait all setup tasks before starting benchmark
      matrix_a.get().waitLocalTiles();
      matrix_b.get().waitLocalTiles();
      DLAF_MPI_CALL(MPI_Barrier(world));

      dlaf::common::Timer<> timeit;
      dlaf::eigensolver::genToStd<Backend::Default, Device::Default, T>(comm_grid, blas::Uplo::Lower,
                                                                        matrix_a.get(), matrix_b.get());

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
                << " " << matrix_a_host.size() << " " << matrix_a_host.blockSize() << " "
                << comm_grid.size() << " " << hpx::get_os_thread_count() << std::endl;

    // (optional) run test
    if ((opts.do_check == CheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
        opts.do_check == CheckIterFreq::All) {
      DLAF_UNIMPLEMENTED("Check");
    }
  }

  dlaf::finalize();

  return hpx::finalize();
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::multiple);

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");
  desc_commandline.add(dlaf::getOptionsDescription());

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",  value<SizeType>()   ->default_value(4096),       "Matrix size")
    ("block-size",   value<SizeType>()   ->default_value( 256),       "Block cyclic distribution size")
    ("grid-rows",    value<int>()        ->default_value(   1),       "Number of row processes in the 2D communicator")
    ("grid-cols",    value<int>()        ->default_value(   1),       "Number of column processes in the 2D communicator")
    ("nruns",        value<int64_t>()    ->default_value(   1),       "Number of runs for the algorithm")
    ("nwarmups",     value<int64_t>()    ->default_value(   1),       "Number of warmup runs")
    ("check-result", value<std::string>()->default_value("none"),     "Enable result checking ('none', 'all', 'last')")
  ;
  // clang-format on

  hpx::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return hpx::init(argc, argv, p);
}

namespace {

options_t parse_options(hpx::program_options::variables_map& vm) {
  // clang-format off
  options_t opts = {
      vm["matrix-size"].as<SizeType>(),
      vm["block-size"].as<SizeType>(),
      vm["grid-rows"].as<int>(),
      vm["grid-cols"].as<int>(),
      vm["nruns"].as<int64_t>(),
      vm["nwarmups"].as<int64_t>(),
      parse_check(vm["check-result"].as<std::string>()),
  };
  // clang-format on

  DLAF_ASSERT(opts.m > 0, opts.m);
  DLAF_ASSERT(opts.mb > 0, opts.mb);
  DLAF_ASSERT(opts.grid_rows > 0, opts.grid_rows);
  DLAF_ASSERT(opts.grid_cols > 0, opts.grid_cols);
  DLAF_ASSERT(opts.nruns > 0, opts.nruns);
  DLAF_ASSERT(opts.nwarmups >= 0, opts.nwarmups);

  if (opts.do_check != CheckIterFreq::None) {
    std::cerr << "Warning! At the moment result checking is not implemented." << std::endl;
    opts.do_check = CheckIterFreq::None;
  }

  return opts;
}

CheckIterFreq parse_check(const std::string& check) {
  if (check == "all")
    return CheckIterFreq::All;
  else if (check == "last")
    return CheckIterFreq::Last;
  else if (check == "none")
    return CheckIterFreq::None;

  std::cout << "Parsing is not implemented for --check-result=" << check << "!" << std::endl;
  std::terminate();
  return CheckIterFreq::None;  // unreachable
}

}
