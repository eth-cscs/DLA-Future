//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iostream>

#include <mpi.h>
#include <hpx/hpx.hpp>

#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/index.h"
#include "dlaf/solver/mc.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/matrix/util_generic_blas.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

#include "dlaf/common/timer.h"

namespace {

using dlaf::Backend;
using dlaf::Device;
using dlaf::GlobalElementIndex;
using dlaf::GlobalElementSize;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;

using dlaf_test::TypeUtilities;

using T = double;
using MatrixType = dlaf::Matrix<T, Device::CPU>;

struct options_t {
  SizeType m;
  SizeType n;
  SizeType mb;
  SizeType nb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  bool do_check;
};

options_t check_options(hpx::program_options::variables_map& vm);

void waitall_tiles(MatrixType& matrix);

}

int hpx_main(hpx::program_options::variables_map& vm) {
  options_t opts = check_options(vm);

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

  // Allocate memory for the matrices
  MatrixType A(GlobalElementSize{opts.m, opts.m}, TileElementSize{opts.mb, opts.mb}, comm_grid);
  MatrixType b(GlobalElementSize{opts.m, opts.n}, TileElementSize{opts.mb, opts.nb}, comm_grid);

  const auto side = blas::Side::Left;
  const auto uplo = blas::Uplo::Lower;
  const auto op = blas::Op::NoTrans;
  const auto diag = blas::Diag::NonUnit;
  const T alpha = 2.0;

  double m = A.size().rows();
  double n = b.size().cols();
  auto add_mul = n * m * m / 2;
  const double total_ops = dlaf::total_ops<T>(add_mul, add_mul);

  using dlaf::matrix::test::getLeftTriangularSystem;
  std::function<T(const GlobalElementIndex&)> setter_A, setter_b, expected_b;
  std::tie(setter_A, setter_b, expected_b) =
      getLeftTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, A.size().rows());

  for (auto run_index = 0; run_index < opts.nruns; ++run_index) {
    if (0 == world.rank())
      std::cout << "[" << run_index << "]" << std::endl;

    // setup matrix A and b
    using dlaf::matrix::util::set;
    set(A, setter_A);
    set(b, setter_b);

    // wait all setup tasks before starting benchmark
    ::waitall_tiles(A);
    ::waitall_tiles(b);
    MPI_Barrier(world);

    dlaf::common::Timer<> timeit;
    dlaf::Solver<Backend::MC>::triangular(comm_grid, side, uplo, op, diag, alpha, A, b);

    // wait for last task and barrier for all ranks
    ::waitall_tiles(A);
    ::waitall_tiles(b);
    MPI_Barrier(world);

    auto elapsed_time = timeit.elapsed();

    // compute gigaflops
    double gigaflops = total_ops / elapsed_time / 1e9;

    // print benchmark results
    if (0 == world.rank())
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << gigaflops << "GFlop/s"
                << " " << A.size() << " " << A.blockSize() << " " << comm_grid.size() << " " << b.size()
                << " " << b.blockSize() << " " << hpx::get_os_thread_count() << std::endl;

    // (optional) run test
    if (opts.do_check) {
      // TODO evaluate to change check
      CHECK_MATRIX_NEAR(expected_b, b, 20 * (b.size().rows() + 1) * TypeUtilities<T>::error,
                        20 * (b.size().rows() + 1) * TypeUtilities<T>::error);
    }
  }

  return hpx::finalize();
}

int main(int argc, char** argv) {
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::serialized);

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Benchmark computation of solution for A . x = 2 . b, "
                                       "where A is a non-unit lower triangular matrix\n\n"
                                       "options");

  // clang-format off
  desc_commandline.add_options()
    ("m",             value<SizeType>()->default_value(4096),  "Matrix b rows")
    ("n",             value<SizeType>()->default_value(512),   "Matrix b columns")
    ("mb",            value<SizeType>()->default_value(256),   "Matrix b block rows")
    ("nb",            value<SizeType>()->default_value(512),   "Matrix b block columns")
    ("grid-rows",     value<int>()     ->default_value(1),     "Number of row processes in the 2D communicator.")
    ("grid-cols",     value<int>()     ->default_value(1),     "Number of column processes in the 2D communicator.")
    ("nruns",         value<int64_t>() ->default_value(1),     "Number of runs to compute the cholesky")
    ("check-result",  bool_switch()    ->default_value(false), "Check the triangular system solution (for each run)")
  ;
  // clang-format on

  // Create the resource partitioner
  hpx::resource::partitioner rp(desc_commandline, argc, argv);

  int ntasks;
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

  // if the user has asked for special thread pools for communication
  // then set them up
  if (ntasks > 1) {
    // Create a thread pool with a single core that we will use for all
    // communication related tasks
    rp.create_thread_pool("mpi", hpx::resource::scheduling_policy::local_priority_fifo);
    rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0], "mpi");
  }

  return hpx::init(hpx_main, desc_commandline, argc, argv);
}

namespace {

options_t check_options(hpx::program_options::variables_map& vm) {
  // clang-format off
  options_t opts = {
    vm["m"].as<SizeType>(),     vm["n"].as<SizeType>(),
    vm["mb"].as<SizeType>(),    vm["nb"].as<SizeType>(),
    vm["grid-rows"].as<int>(),  vm["grid-cols"].as<int>(),

    vm["nruns"].as<int64_t>(),
    vm["check-result"].as<bool>(),
  };
  // clang-format on

  DLAF_ASSERT(opts.m > 0 && opts.n > 0, opts.m, opts.n);
  DLAF_ASSERT(opts.mb > 0 && opts.nb > 0, opts.mb, opts.nb);
  DLAF_ASSERT(opts.grid_rows > 0 && opts.grid_cols > 0, opts.grid_rows, opts.grid_cols);

  return opts;
}

void waitall_tiles(MatrixType& matrix) {
  for (const auto tile_idx : dlaf::common::iterate_range2d(matrix.distribution().localNrTiles()))
    matrix(tile_idx).get();
}

}
