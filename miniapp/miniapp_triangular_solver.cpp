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
using dlaf::Matrix;
using dlaf::SizeType;
using dlaf::TileElementSize;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::common::Ordering;

using dlaf_test::TypeUtilities;

using T = double;

struct options_t {
  SizeType m;
  SizeType mb;
  SizeType n;
  SizeType nb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  bool do_check;
};

options_t check_options(hpx::program_options::variables_map& vm);

}

int hpx_main(hpx::program_options::variables_map& vm) {
  options_t opts = check_options(vm);

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

  using MatrixType = Matrix<T, Device::CPU>;

  // Allocate memory for the matrices
  MatrixType A(GlobalElementSize{opts.m, opts.m}, TileElementSize{opts.mb, opts.mb}, comm_grid);
  MatrixType b(GlobalElementSize{opts.m, opts.n}, TileElementSize{opts.mb, opts.nb}, comm_grid);

  const auto side = blas::Side::Left;
  const auto uplo = blas::Uplo::Lower;
  const auto op = blas::Op::NoTrans;
  const auto diag = blas::Diag::NonUnit;
  const T alpha = 2.0;

  for (auto run_index = 0; run_index < opts.nruns; ++run_index) {
    if (0 == world.rank())
      std::cout << "[" << run_index << "]" << std::endl;

    // setup matrix A and b
    using dlaf::matrix::test::getLeftTriangularSystem;
    std::function<T(const GlobalElementIndex&)> setter_A, setter_b, expected_b;
    std::tie(setter_A, setter_b, expected_b) =
        getLeftTriangularSystem<GlobalElementIndex, T>(uplo, op, diag, alpha, A.size().rows());

    // TODO wait all setup tasks before starting benchmark

    dlaf::common::Timer<> timeit;
    dlaf::Solver<Backend::MC>::triangular(comm_grid, side, uplo, op, diag, alpha, A, b);

    // TODO wait for last task and barrier for all ranks

    auto elapsed_time = timeit.elapsed();

    // TODO compute gigaflops

    // TODO print benchmark results
    if (0 == world.rank())
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
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
  // Initialize MPI
  int threading_required = MPI_THREAD_SERIALIZED;
  int threading_provided;
  MPI_Init_thread(&argc, &argv, threading_required, &threading_provided);

  if (threading_provided != threading_required) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size",       value<SizeType>()->default_value(4096),  "Matrix size.")
    ("block-size",        value<SizeType>()->default_value(256),   "Block cyclic distribution size.")
    ("result-cols",       value<SizeType>()->default_value(512),   "Matrix size.")
    ("result-block-cols", value<SizeType>()->default_value(512),   "Block cyclic distribution size.")
    ("grid-rows",         value<int>()     ->default_value(1),     "Number of row processes in the 2D communicator.")
    ("grid-cols",         value<int>()     ->default_value(1),     "Number of column processes in the 2D communicator.")
    ("nruns",             value<int64_t>() ->default_value(1),     "Number of runs to compute the cholesky")
    ("check-result",      bool_switch()    ->default_value(false), "Check the triangular system solution (for each run)")
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

  auto ret_code = hpx::init(hpx_main, desc_commandline, argc, argv);

  MPI_Finalize();

  return ret_code;
}

namespace {

options_t check_options(hpx::program_options::variables_map& vm) {
  options_t opts = {
      vm["matrix-size"].as<SizeType>(), vm["block-size"].as<SizeType>(),
      vm["result-cols"].as<SizeType>(), vm["result-block-cols"].as<SizeType>(),
      vm["grid-rows"].as<int>(),        vm["grid-cols"].as<int>(),

      vm["nruns"].as<int64_t>(),

      vm["check-result"].as<bool>(),
  };

  if (opts.m <= 0)
    throw std::runtime_error("A size must be a positive number");
  if (opts.mb <= 0)
    throw std::runtime_error("A block size must be a positive number");

  if (opts.n <= 0)
    throw std::runtime_error("b number of cols must be a positive number");
  if (opts.nb <= 0)
    throw std::runtime_error("b block width must be a positive number");

  if (opts.grid_rows <= 0)
    throw std::runtime_error("number of grid rows must be a positive number");
  if (opts.grid_cols <= 0)
    throw std::runtime_error("number of grid columns must be a positive number");

  return opts;
}

}
