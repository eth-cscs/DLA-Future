//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <chrono>
#include <complex>
#include <cstdio>

#include <mpi.h>
#include <hpx/init.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/runtime.hpp>
#include <hpx/local/unwrap.hpp>
#include <hpx/program_options.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/error.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/executors.h"
#include "dlaf/init.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/util_matrix.h"

// In SIRIUS, `A`, `B` and `C` are usually submatrices of bigger matrices. The
// only difference that entails is that the `lld` for `C` might be larger than
// assumed here. Hence writing to `C` might be slightly faster than in SIRIUS.
//
// Assumptions: Tall and skinny `k` >> `m` and `k` >> `n`.
//
// Matrices: `A` (`m x k`), `B` (`k x n`) and `C` (m x n).
//
// `A` is complex conjugated.
//
// `C` is distributed in 2D block-cyclic manner. The 2D process grid is row
// major (the MPI default) with process 0 in the top left corner.
//
// All matrices are distributed in column-major order.
//
// Local distribution of A and B. Only the `k` dimension is split. In
// SIRIUS, `k_loc` is approximately equally distributed. `k_loc` coincides
// with `lld` for `A` and `B`. If there is a remainder, distributed it
// across ranks starting from the `0`-th.
//

// Forward declarations
namespace {

using dlaf::Device;
using dlaf::Backend;
using dlaf::SizeType;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::matrix::Distribution;
using dlaf::common::Ordering;
using dlaf::TileElementSize;
using dlaf::GlobalElementSize;
using dlaf::LocalElementSize;
using dlaf::TileElementIndex;
using dlaf::SizeType;
using dlaf::GlobalTileIndex;
using dlaf::LocalTileIndex;
using dlaf::LocalTileSize;

using ScalarType = std::complex<double>;
using MatrixType = dlaf::Matrix<ScalarType, Device::Default>;
using ConstMatrixType = dlaf::Matrix<const ScalarType, Device::Default>;
using HostMatrixType = dlaf::Matrix<ScalarType, Device::CPU>;
using ConstHostMatrixType = dlaf::Matrix<const ScalarType, Device::CPU>;
using MatrixMirrorType = dlaf::matrix::MatrixMirror<ScalarType, Device::Default, Device::CPU>;
using TileType = MatrixType::TileType;
using ConstTileType = ConstMatrixType::ConstTileType;

template <Backend backend>
void sirius_gemm(CommunicatorGrid grid, ConstMatrixType& a_mat, ConstMatrixType& b_mat,
                 MatrixType& cini_mat, MatrixType& cfin_mat);

// Initialize matrix
void setMatrix(MatrixType& matrix, ScalarType val);

// Sum the elements of the matrix. Useful for debugging.
ScalarType sumMatrixElements(Communicator const& comm, MatrixType& matrix);

struct options_t {
  SizeType m;
  SizeType n;
  SizeType k;
  SizeType mb;
  SizeType nb;
  int grid_rows;
  int grid_cols;
  // int batch_size;
  int64_t nruns;
  int64_t nwarmups;
  bool do_check;
};

/// Handle CLI options
options_t parseOptions(hpx::program_options::variables_map&);

double calc_tsgemm_ops(const options_t&, double);

}  // end namespace

int hpx_main(hpx::program_options::variables_map& vm) {
  dlaf::initialize(vm);

  // Input
  options_t opts = parseOptions(vm);

  // Communicators
  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

  // Matrices `A` and `B`
  // The matrices are distributed only along the `k` dimension. In SIRIUS, the sections assigned to each
  // process are not exactly equal, they differ by a little in non-trivial ways. SIRIUS's distribution
  // for A and B is NOT a special case of block cyclic distribution. In this miniapp, the distribution is
  // emulated by DLAF local matrices.
  int nprocs = world.size();
  int rank = world.rank();
  SizeType k_loc = opts.k / nprocs + ((rank < opts.k % nprocs) ? 1 : 0);
  MatrixType a_mat(LocalElementSize(k_loc, opts.m), TileElementSize(k_loc, opts.mb));
  MatrixType b_mat(LocalElementSize(k_loc, opts.n), TileElementSize(k_loc, opts.nb));

  ScalarType a_val = 4.2;
  ScalarType b_val = 1.3;

  // Matrices `C`-initial and `C`-final
  using dlaf::matrix::tileLayout;
  MatrixType cini_mat(Distribution(LocalElementSize(opts.m, opts.n), TileElementSize(opts.mb, opts.nb)),
                      tileLayout(LocalElementSize(opts.m, opts.n), TileElementSize(opts.mb, opts.nb)));
  MatrixType cfin_mat(GlobalElementSize(opts.m, opts.n), TileElementSize(opts.mb, opts.nb), comm_grid);

  setMatrix(a_mat, a_val);
  setMatrix(b_mat, b_val);

  // Benchmark calls of `sirius_gemm`
  for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
    if (0 == world.rank() && run_index >= 0) {
      std::cout << "[" << run_index << "]" << std::endl;
    }

    double elapsed_time;
    {
      cfin_mat.waitLocalTiles();
      MPI_Barrier(world);

      dlaf::common::Timer<> timeit;
      sirius_gemm<Backend::Default>(comm_grid, a_mat, b_mat, cini_mat, cfin_mat);

      cfin_mat.waitLocalTiles();
      MPI_Barrier(world);
      elapsed_time = timeit.elapsed();
    }

    if (rank == 0 && run_index >= 0) {
      // clang-format off
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << calc_tsgemm_ops(opts, elapsed_time) << "GFlop/s"
                << " " << opts.m
                << " " << opts.n
                << " " << opts.k
                << " " << opts.mb
                << " " << opts.nb
                << " " << opts.grid_rows
                << " " << opts.grid_cols
                << " " << hpx::get_os_thread_count()
                << std::endl;
      // clang-format on
    }

    // Simple check
    if (opts.do_check) {
      using BaseScalarType = dlaf::BaseType<ScalarType>;
      using numlims_t = std::numeric_limits<BaseScalarType>;
      constexpr auto eps = numlims_t::epsilon();
      ScalarType cfin_sum = sumMatrixElements(world, cfin_mat);
      ScalarType expected_cfin_sum = ScalarType(opts.m * opts.n * opts.k) * a_val * std::conj(b_val);
      if (world.rank() == 0 && std::abs(cfin_sum - expected_cfin_sum) < eps) {
        std::cout.precision(numlims_t::digits10);
        std::cout << "FAILURE !!!" << std::endl;
        std::cout << "ACTUAL : " << cfin_sum << std::endl;
        std::cout << "EXPECTED : " << expected_cfin_sum << std::endl;
      }
    }
  }

  dlaf::finalize();
  return hpx::finalize();
}

int main(int argc, char** argv) {
  // Init MPI
  dlaf::comm::mpi_init mpi_initter(argc, argv, dlaf::comm::mpi_thread_level::multiple);

  // Options
  using namespace hpx::program_options;
  options_description desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

  // clang-format off
  desc_commandline.add_options()
     ("m",            value<SizeType>()->default_value( 100),  "m dimension")
     ("n",            value<SizeType>()->default_value( 100),  "n dimension")
     ("k",            value<SizeType>()->default_value(1000),  "k dimension")
     ("mb",           value<SizeType>()->default_value(  32),  "tile m dimension")
     ("nb",           value<SizeType>()->default_value(  32),  "tile n dimension")
     ("grid-rows",    value<int>()     ->default_value(   1),  "process grid rows")
     ("grid-cols",    value<int>()     ->default_value(   1),  "process grid columns")
     //("batch_size",   value<int>()     ->default_value(  16), "number of tiles batched for computation/communication")
     ("nruns",        value<int64_t>() ->default_value(   1),  "number of iterations")
     ("nwarmups",     value<int64_t>() ->default_value(   1),  "number of iterations")
     ("check-result", bool_switch()    ->default_value(false), "Print the sum of elements of the resulting matrix.")
  ;
  // clang-format on

  desc_commandline.add(dlaf::getOptionsDescription());

  hpx::init_params p;
  p.desc_cmdline = desc_commandline;
  p.rp_callback = dlaf::initResourcePartitionerHandler;
  return hpx::init(argc, argv, p);
}

namespace {

template <Backend backend>
void sirius_gemm(CommunicatorGrid grid, ConstMatrixType& a_mat, ConstMatrixType& b_mat,
                 MatrixType& cini_mat, MatrixType& cfin_mat) {
  using hpx::unwrapping;
  using dlaf::matrix::unwrapExtendTiles;
  using dlaf::common::Pipeline;
  using dlaf::comm::Communicator;
  using dlaf::common::computeLinearIndexColMajor;
  using dlaf::comm::Executor;
  using hpx::util::annotated_function;
  using dlaf::tile::gemm_o;
  using dlaf::matrix::copy_o;

  auto executor_hp = dlaf::getHpExecutor<backend>();
  auto executor_np = dlaf::getNpExecutor<backend>();
  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  Pipeline<Communicator> mpi_chain(grid.fullCommunicator());

  Distribution const& cfin_dist = cfin_mat.distribution();
  int this_rank = grid.rankFullCommunicator(cfin_dist.rankIndex());
  dlaf::LocalTileSize const& tile_grid_size = cini_mat.distribution().localNrTiles();

  for (auto cloc_idx : iterate_range2d(tile_grid_size)) {
    LocalTileIndex a_idx(0, cloc_idx.row());
    LocalTileIndex b_idx(0, cloc_idx.col());
    GlobalTileIndex c_idx(cloc_idx.row(), cloc_idx.col());

    int tile_rank = grid.rankFullCommunicator(cfin_dist.rankGlobalTile(c_idx));

    // GEMM
    hpx::dataflow(executor_np, unwrapExtendTiles(annotated_function(gemm_o, "gemm")), blas::Op::Trans,
                  blas::Op::NoTrans, ScalarType(1), a_mat.read(a_idx), b_mat.read(b_idx), ScalarType(0),
                  cini_mat(c_idx));

    //    int tile_tag = computeLinearIndexColMajor(cloc_idx, tile_grid_size);
    if (this_rank == tile_rank) {
      // RECV
      dlaf::comm::scheduleReduceRecvInPlace(executor_mpi, mpi_chain(), MPI_SUM, cini_mat(c_idx));

      // COPY from c_ini to c_fin
      hpx::dataflow(executor_hp, unwrapExtendTiles(annotated_function(copy_o, "copy")),
                    cini_mat.read(c_idx), cfin_mat(c_idx));
    }
    else {
      dlaf::comm::scheduleReduceSend(executor_mpi, tile_rank, mpi_chain(), MPI_SUM,
                                     cini_mat.read(c_idx));
    }
  }
}

void setMatrix(MatrixType& matrix, ScalarType val) {
  for (auto tile_idx : iterate_range2d(matrix.distribution().localNrTiles())) {
    TileType tile = matrix(tile_idx).get();
    for (auto el_idx : iterate_range2d(tile.size())) {
      tile(el_idx) = val;
    }
  }
}

// Sums the distributed matrix and returns the result to process 0.
ScalarType sumMatrixElements(Communicator const& comm, MatrixType& matrix) {
  ScalarType local_sum = 0;

  for (auto tile_idx : iterate_range2d(matrix.distribution().localNrTiles())) {
    TileType tile = matrix(tile_idx).get();
    for (auto el_idx : iterate_range2d(tile.size())) {
      local_sum += tile(el_idx);
    }
  }

  ScalarType global_sum = 0;
  MPI_Datatype mpi_type = dlaf::comm::mpi_datatype<ScalarType>::type;
  DLAF_MPI_CALL(MPI_Reduce(&local_sum, &global_sum, 1, mpi_type, MPI_SUM, 0, comm));

  return global_sum;
}

options_t parseOptions(hpx::program_options::variables_map& vm) {
  // using dlaf::util::ceilDiv;

  // clang-format off
  options_t opts = {
      vm["m"].as<SizeType>(),
      vm["n"].as<SizeType>(),
      vm["k"].as<SizeType>(),
      vm["mb"].as<SizeType>(),
      vm["nb"].as<SizeType>(),
      vm["grid-rows"].as<int>(),
      vm["grid-cols"].as<int>(),
      //vm["batch_size"].as<int>()
      vm["nruns"].as<int64_t>(),
      vm["nwarmups"].as<int64_t>(),
      vm["check-result"].as<bool>(),
  };
  // clang-format on

  DLAF_ASSERT(opts.m > 0, opts.m);
  DLAF_ASSERT(opts.n > 0, opts.n);
  DLAF_ASSERT(opts.k > 0, opts.k);
  DLAF_ASSERT(opts.grid_rows > 0, opts.grid_rows);
  DLAF_ASSERT(opts.grid_cols > 0, opts.grid_cols);
  DLAF_ASSERT(opts.mb > 0, opts.mb);
  DLAF_ASSERT(opts.nb > 0, opts.nb);
  DLAF_ASSERT(opts.nruns > 0, opts.nruns);

  return opts;
}

double calc_tsgemm_ops(const options_t& opts, double elapsed_time) {
  double mul_ops = opts.m * opts.n * opts.k;
  double add_ops = opts.m * opts.n * (opts.k - 1);
  double total_ops = dlaf::total_ops<ScalarType>(add_ops, mul_ops);
  double gigaflops = total_ops / elapsed_time / 1e9;
  return gigaflops;
}

}
