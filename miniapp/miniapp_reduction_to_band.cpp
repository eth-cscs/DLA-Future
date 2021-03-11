#include <iostream>

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <limits>

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/init.h"
#include "dlaf/eigensolver/reduction_to_band.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/print_numpy.h"
#include "dlaf/types.h"

namespace {
using dlaf::Device;
using dlaf::SizeType;

using T = double;
using MatrixType = dlaf::Matrix<T, Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const T, Device::CPU>;

struct options_t {
  SizeType m;
  SizeType mb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  int64_t nwarmups;
};

/// Handle CLI options
options_t check_options(hpx::program_options::variables_map& vm);
}

int miniapp(hpx::program_options::variables_map& vm) {
  using namespace dlaf;
  using dlaf::SizeType;
  using dlaf::comm::Communicator;
  using dlaf::comm::CommunicatorGrid;

  options_t opts = check_options(vm);

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
    {
      for (const auto tile_idx : dlaf::common::iterate_range2d(distribution.localNrTiles()))
        matrix(tile_idx).get();
      DLAF_MPI_CALL(MPI_Barrier(world));
    }

    dlaf::common::Timer<> timeit;
    auto taus = dlaf::eigensolver::reductionToBand<dlaf::Backend::MC>(comm_grid, matrix);

    // wait for last task and barrier for all ranks
    {
      GlobalTileIndex last_tile(matrix.nrTiles().rows() - 1, matrix.nrTiles().cols() - 2);
      if (matrix.rankIndex() == distribution.rankGlobalTile(last_tile))
        matrix(last_tile).get();

      DLAF_MPI_CALL(MPI_Barrier(world));
    }
    auto elapsed_time = timeit.elapsed();

    double gigaflops = std::numeric_limits<T>::quiet_NaN();
    {
      // double n = matrix.size().rows();
      // auto add_mul = n * n * n / 6;
      // gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
    }

    // print benchmark results
    if (0 == world.rank() && run_index >= 0)
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << gigaflops << "GFlop/s"
                << " " << matrix.size() << " " << matrix.blockSize() << " " << comm_grid.size() << " "
                << hpx::get_os_thread_count() << std::endl;
  }

  return hpx::finalize();
}

int main(int argc, char** argv) {
  using dlaf::SizeType;

  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::serialized);

  // options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // clang-format off
  desc_commandline.add_options()
    ("matrix-size", value<SizeType>() ->default_value(4), "Matrix rows")
    ("block-size",  value<SizeType>() ->default_value(2), "Block cyclic distribution size")
    ("grid-rows",   value<int>()      ->default_value(1), "Number of row processes in the 2D communicator")
    ("grid-cols",   value<int>()      ->default_value(1), "Number of column processes in the 2D communicator")
    ("nruns",       value<int64_t>()  ->default_value(1), "Number of runs to compute the cholesky")
    ("nwarmups",    value<int64_t>()  ->default_value(1), "Number of warmup runs");
  // clang-format on

  hpx::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = [](auto& rp, auto) {
    int ntasks;
    DLAF_MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &ntasks));
    // if the user has asked for special thread pools for communication
    // then set them up
    if (ntasks > 1) {
      // Create a thread pool with a single core that we will use for all
      // communication related tasks
      rp.create_thread_pool("mpi", hpx::resource::scheduling_policy::local_priority_fifo);
      rp.add_resource(rp.numa_domains()[0].cores()[0].pus()[0], "mpi");
    }
  };

  auto ret_code = hpx::init(miniapp, argc, argv, p);

  return ret_code;
}

namespace {

options_t check_options(hpx::program_options::variables_map& vm) {
  options_t opts = {
      vm["matrix-size"].as<SizeType>(), vm["block-size"].as<SizeType>(), vm["grid-rows"].as<int>(),
      vm["grid-cols"].as<int>(),        vm["nruns"].as<int64_t>(),       vm["nwarmups"].as<int64_t>(),
  };

  DLAF_ASSERT(opts.m > 0, opts.m);
  DLAF_ASSERT(opts.mb > 0, opts.mb);
  DLAF_ASSERT(opts.grid_rows > 0, opts.grid_rows);
  DLAF_ASSERT(opts.grid_cols > 0, opts.grid_cols);
  DLAF_ASSERT(opts.nruns > 0, opts.nruns);
  DLAF_ASSERT(opts.nwarmups >= 0, opts.nwarmups);

  return opts;
}

}
