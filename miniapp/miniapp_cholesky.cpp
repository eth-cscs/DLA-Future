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
#include <hpx/hpx_init.hpp>

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/factorization/mc.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/matrix/util_matrix.h"
#include "dlaf_test/util_types.h"

#include "dlaf/common/timer.h"

#include "check_cholesky.tpp"

using namespace dlaf;

using T = double;

struct options_t {
  SizeType m;
  SizeType mb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  bool do_check;
};

options_t check_options(hpx::program_options::variables_map& vm);

int hpx_main(hpx::program_options::variables_map& vm) {
  options_t opts = check_options(vm);

  comm::Communicator world(MPI_COMM_WORLD);
  comm::CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, common::Ordering::ColumnMajor);

  // Allocate memory for the matrix
  GlobalElementSize matrix_size(opts.m, opts.m);
  TileElementSize block_size(opts.mb, opts.mb);

  Matrix<T, Device::CPU> matrix(matrix_size, block_size, comm_grid);
  auto distribution = matrix.distribution();

  for (auto run_index = 0; run_index < opts.nruns; ++run_index) {
    if (0 == world.rank())
      std::cout << "[" << run_index << "]" << std::endl;

    // TODO this should be a clone of the original reference one
    matrix::util::set_random_hermitian_positive_definite(matrix);

    // wait all setup tasks before starting benchmark
    {
      for (int local_tile_j = 0; local_tile_j < distribution.localNrTiles().cols(); ++local_tile_j)
        for (int local_tile_i = 0; local_tile_i < distribution.localNrTiles().rows(); ++local_tile_i)
          matrix(LocalTileIndex{local_tile_i, local_tile_j}).get();
      MPI_Barrier(world);
    }

    common::Timer<> timeit;
    Factorization<Backend::MC>::cholesky(comm_grid, blas::Uplo::Lower, matrix);

    // wait for last task and barrier for all ranks
    {
      GlobalTileIndex last_tile(matrix.nrTiles().rows() - 1, matrix.nrTiles().cols() - 1);
      if (matrix.rankIndex() == distribution.rankGlobalTile(last_tile))
        matrix(last_tile).get();

      MPI_Barrier(world);
    }
    auto elapsed_time = timeit.elapsed();

    double gigaflops;
    {
      double n = matrix.size().rows();
      auto add_mul = n * n * n / 6;
      gigaflops = total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
    }

    // print benchmark results
    if (0 == world.rank())
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << gigaflops << "GFlop/s"
                << " " << matrix.size() << " " << matrix.blockSize() << " " << comm_grid.size() << " "
                << hpx::get_os_thread_count() << std::endl;

    // (optional) run test
    if (opts.do_check) {
      // TODO this should be a clone of the original one
      Matrix<T, Device::CPU> original(matrix_size, block_size, comm_grid);
      matrix::util::set_random_hermitian_positive_definite(original);

      check_cholesky(original, matrix, comm_grid);
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
    ("matrix-size",
     value<SizeType>()->default_value(4096),
     "Matrix size.")
    ("block-size",
     value<SizeType>()->default_value(256),
     "Block cyclic distribution size.")
    ("grid-rows",
     value<int>()->default_value(1),
     "Number of row processes in the 2D communicator.")
    ("grid-cols",
     value<int>()->default_value(1),
     "Number of column processes in the 2D communicator.")
    ("nruns",
     value<int64_t>()->default_value(1),
     "Number of runs to compute the cholesky")
    ("check-result",
     bool_switch()->default_value(false),
     "Check the cholesky factorization (for each run)")
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

options_t check_options(hpx::program_options::variables_map& vm) {
  options_t opts = {
      vm["matrix-size"].as<SizeType>(), vm["block-size"].as<SizeType>(),
      vm["grid-rows"].as<int>(),        vm["grid-cols"].as<int>(),

      vm["nruns"].as<int64_t>(),

      vm["check-result"].as<bool>(),
  };

  if (opts.m <= 0)
    throw std::runtime_error("matrix size must be a positive number");
  if (opts.mb <= 0)
    throw std::runtime_error("block size must be a positive number");

  if (opts.grid_rows <= 0)
    throw std::runtime_error("number of grid rows must be a positive number");
  if (opts.grid_cols <= 0)
    throw std::runtime_error("number of grid columns must be a positive number");

  if (opts.do_check && opts.m % opts.mb) {
    std::cerr
        << "Warning! At the moment result checking works just with matrix sizes that are multiple of the block size."
        << std::endl;
    opts.do_check = false;
  }

  return opts;
}
