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
#include "dlaf/factorization/cholesky.h"
#include "dlaf/init.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/util_matrix.h"

// In SIRIUS, `A`, `B` and `C` are usually submatrices of bigger matrices. The
// only difference that entails is that the `lld` for `C` might be larger than
// assumed here. Hence writing to `C` might be slightly faster than in SIRIUS.
//
// Assumptions: Tall and skinny `k` >> `m`.
//
// Matrices: `A` (`m x k`), `B` (`k x m`) and `C` (m x m).
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

enum class CholCheckIterFreq { None, Last, All };

template <Backend backend>
void sirius_tsgemm(CommunicatorGrid grid, ConstMatrixType& a_mat, ConstMatrixType& b_mat,
                   MatrixType& cini_mat, MatrixType& cfin_mat);

template <Backend backend>
void make_diag_dominant(CommunicatorGrid grid, MatrixType& cfin_mat);

// Sum the elements of the matrix. Useful for debugging.
// ScalarType sumMatrixElements(Communicator const& comm, MatrixType& matrix);

struct options_t {
  SizeType m;
  SizeType k;
  SizeType mb;
  int grid_rows;
  int grid_cols;
  bool no_overlap;

  int64_t nruns;
  int64_t nwarmups;
};

/// Handle CLI options
options_t parseOptions(hpx::program_options::variables_map&);

double calc_gemmchol_ops(const options_t&, double);

}  // end namespace

int hpx_main(hpx::program_options::variables_map& vm) {
  using dlaf::factorization::cholesky;

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
  MatrixType b_mat(LocalElementSize(k_loc, opts.m), TileElementSize(k_loc, opts.mb));

  dlaf::matrix::util::set_random(a_mat);
  dlaf::matrix::copy(a_mat, b_mat);

  // Matrices `C`-initial and `C`-final
  using dlaf::matrix::tileLayout;
  MatrixType cini_mat(Distribution(LocalElementSize(opts.m, opts.m), TileElementSize(opts.mb, opts.mb)),
                      tileLayout(LocalElementSize(opts.m, opts.m), TileElementSize(opts.mb, opts.mb)));
  MatrixType cfin_mat(GlobalElementSize(opts.m, opts.m), TileElementSize(opts.mb, opts.mb), comm_grid);

  // Benchmark calls of `sirius_gemm`
  for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
    if (0 == world.rank() && run_index >= 0) {
      std::cout << "[" << run_index << "]" << std::endl;
    }

    double elapsed_time;
    {
      // MatrixMirrorType matrix(matrix_host);

      cfin_mat.waitLocalTiles();
      MPI_Barrier(world);

      dlaf::common::Timer<> timeit;
      sirius_tsgemm<Backend::Default>(comm_grid, a_mat, b_mat, cini_mat, cfin_mat);

      if (opts.no_overlap) {
        cfin_mat.waitLocalTiles();
        MPI_Barrier(world);
      }
      make_diag_dominant<Backend::Default>(comm_grid, cfin_mat);

      cholesky<Backend::Default, Device::Default, ScalarType>(comm_grid, blas::Uplo::Lower, cfin_mat);

      cfin_mat.waitLocalTiles();
      MPI_Barrier(world);
      elapsed_time = timeit.elapsed();
    }

    if (rank == 0 && run_index >= 0) {
      // clang-format off
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << calc_gemmchol_ops(opts, elapsed_time) << "GFlop/s"
                << " " << opts.m
                << " " << opts.k
                << " " << opts.mb
                << " " << opts.grid_rows
                << " " << opts.grid_cols
                << " " << hpx::get_os_thread_count()
                << std::endl;
      // clang-format on
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
     ("m",            value<SizeType>()->default_value( 100),  "short matrix dimension")
     ("k",            value<SizeType>()->default_value(1000),  "long matrix dimension")
     ("mb",           value<SizeType>()->default_value(  32),  "block size")
     ("grid-rows",    value<int>()     ->default_value(   1),  "process grid rows")
     ("grid-cols",    value<int>()     ->default_value(   1),  "process grid columns")
     ("no-overlap",   bool_switch()    ->default_value(false), "Disable overlap of TSGEMM and Cholesky")
     ("nruns",        value<int64_t>() ->default_value(   1),  "Number of runs to compute the cholesky")
     ("nwarmups",     value<int64_t>() ->default_value(   1),  "Number of warmup runs")
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
void sirius_tsgemm(CommunicatorGrid grid, ConstMatrixType& a_mat, ConstMatrixType& b_mat,
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
    hpx::dataflow(executor_np, unwrapExtendTiles(annotated_function(gemm_o, "gemm")),
                  blas::Op::ConjTrans, blas::Op::NoTrans, ScalarType(1), a_mat.read(a_idx),
                  b_mat.read(b_idx), ScalarType(0), cini_mat(c_idx));

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

template <Backend backend>
void make_diag_dominant(CommunicatorGrid grid, MatrixType& cfin_mat) {
  using dlaf::comm::Index2D;
  using dlaf::comm::Executor;
  using hpx::unwrapping;

  DLAF_ASSERT(dlaf::matrix::square_size(cfin_mat), cfin_mat);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(cfin_mat), cfin_mat);

  auto executor_hp = dlaf::getHpExecutor<backend>();

  Distribution const& dist = cfin_mat.distribution();
  const Index2D this_rank = grid.rank();
  const SizeType nrtile = cfin_mat.nrTiles().cols();
  const SizeType m_dim = cfin_mat.size().rows();

  // Iterate over diagonal tiles
  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const Index2D kk_rank = dist.rankGlobalTile(kk_idx);

    if (kk_rank == this_rank) {
      auto update_diag = hpx::unwrapping([m_dim](auto&& tile) {
        SizeType ts = tile.size().rows();
        // Iterate over the diagonal of the tile
        for (SizeType kt = 0; kt < ts; ++kt) {
          tile(TileElementIndex(kt, kt)) += SizeType(m_dim);
        }
      });
      hpx::dataflow(executor_hp, std::move(update_diag), cfin_mat(kk_idx));
    }
  }
}

options_t parseOptions(hpx::program_options::variables_map& vm) {
  // clang-format off
  options_t opts = {
      vm["m"].as<SizeType>(),
      vm["k"].as<SizeType>(),
      vm["mb"].as<SizeType>(),
      vm["grid-rows"].as<int>(),
      vm["grid-cols"].as<int>(),
      vm["no-overlap"].as<bool>(),

      vm["nruns"].as<int64_t>(),
      vm["nwarmups"].as<int64_t>(),
  };
  // clang-format on

  DLAF_ASSERT(opts.m > 0, opts.m);
  DLAF_ASSERT(opts.k > 0, opts.k);
  DLAF_ASSERT(opts.grid_rows > 0, opts.grid_rows);
  DLAF_ASSERT(opts.grid_cols > 0, opts.grid_cols);
  DLAF_ASSERT(opts.mb > 0, opts.mb);
  DLAF_ASSERT(opts.nruns > 0, opts.nruns);

  return opts;
}

double calc_gemmchol_ops(const options_t& opts, double elapsed_time) {
  using dlaf::total_ops;
  double gemm_mul_ops = opts.m * opts.m * opts.k;
  double gemm_add_ops = opts.m * opts.m * (opts.k - 1);
  double chol_mul_ops = opts.m * opts.m * opts.m / 6.0;
  double chol_add_ops = opts.m * opts.m * opts.m / 6.0;

  double tot_ops = total_ops<ScalarType>(gemm_add_ops + chol_add_ops, gemm_mul_ops + chol_mul_ops);
  double gigaflops = tot_ops / elapsed_time / 1e9;
  return gigaflops;
}

}
