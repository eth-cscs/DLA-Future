//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <blas/util.hh>
#include <iostream>

#include <mpi.h>
#include <hpx/init.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/runtime.hpp>
#include <hpx/local/unwrap.hpp>
#include <hpx/program_options.hpp>

#include "dlaf/auxiliary/norm.h"
#include "dlaf/blas/tile.h"
#include "dlaf/common/timer.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/error.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/mech.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/factorization/cholesky.h"
#include "dlaf/init.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/matrix_mirror.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace {

using hpx::unwrapping;

using dlaf::Device;
using dlaf::Coord;
using dlaf::Backend;
using dlaf::SizeType;
using dlaf::comm::Index2D;
using dlaf::GlobalElementSize;
using dlaf::GlobalElementIndex;
using dlaf::GlobalTileIndex;
using dlaf::LocalTileIndex;
using dlaf::TileElementIndex;
using dlaf::TileElementSize;
using dlaf::common::Ordering;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;

using T = double;
using MatrixType = dlaf::Matrix<T, Device::Default>;
using HostMatrixType = dlaf::Matrix<T, Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const T, Device::Default>;
using ConstHostMatrixType = dlaf::Matrix<const T, Device::CPU>;
using MatrixMirrorType = dlaf::matrix::MatrixMirror<T, Device::Default, Device::CPU>;
using TileType = MatrixType::TileType;
using HostTileType = HostMatrixType::TileType;
using ConstTileType = MatrixType::ConstTileType;
using ConstHostTileType = HostMatrixType::ConstTileType;

/// Check Cholesky Factorization results
///
/// Given a matrix A (Hermitian Positive Definite) and its Cholesky factorization in L,
/// this function checks that A == L * L'
void check_cholesky(HostMatrixType& A, HostMatrixType& L, CommunicatorGrid comm_grid);

enum class CholCheckIterFreq { None, Last, All };

struct options_t {
  SizeType m;
  SizeType mb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  int64_t nwarmups;
  CholCheckIterFreq do_check;
};

/// Handle CLI options
options_t parse_options(hpx::program_options::variables_map&);

CholCheckIterFreq parse_chol_check(const std::string&);

}

int hpx_main(hpx::program_options::variables_map& vm) {
  dlaf::initialize(vm);
  options_t opts = parse_options(vm);

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

  // Allocate memory for the matrix
  GlobalElementSize matrix_size(opts.m, opts.m);
  TileElementSize block_size(opts.mb, opts.mb);

  ConstHostMatrixType matrix_ref = [matrix_size, block_size, comm_grid]() {
    using dlaf::matrix::util::set_random_hermitian_positive_definite;

    HostMatrixType hermitian_pos_def(matrix_size, block_size, comm_grid);
    set_random_hermitian_positive_definite(hermitian_pos_def);

    return hermitian_pos_def;
  }();

  const auto& distribution = matrix_ref.distribution();

  for (int64_t run_index = -opts.nwarmups; run_index < opts.nruns; ++run_index) {
    if (0 == world.rank() && run_index >= 0)
      std::cout << "[" << run_index << "]" << std::endl;

    HostMatrixType matrix_host(matrix_size, block_size, comm_grid);
    copy(matrix_ref, matrix_host);

    // wait all setup tasks before starting benchmark
    matrix_host.waitLocalTiles();
    DLAF_MPI_CALL(MPI_Barrier(world));

    double elapsed_time;
    {
      MatrixMirrorType matrix(matrix_host);

      // Wait for matrix to be copied to GPU (if necessary)
      matrix.get().waitLocalTiles();

      dlaf::common::Timer<> timeit;
      dlaf::factorization::cholesky<Backend::Default, Device::Default, T>(comm_grid, blas::Uplo::Lower,
                                                                          matrix.get());

      // wait for last task and barrier for all ranks
      {
        GlobalTileIndex last_tile(matrix.get().nrTiles().rows() - 1, matrix.get().nrTiles().cols() - 1);
        if (matrix.get().rankIndex() == distribution.rankGlobalTile(last_tile))
          matrix.get()(last_tile).get();

        DLAF_MPI_CALL(MPI_Barrier(world));
      }
      elapsed_time = timeit.elapsed();
    }

    double gigaflops;
    {
      double n = matrix_host.size().rows();
      auto add_mul = n * n * n / 6;
      gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
    }

    // print benchmark results
    if (0 == world.rank() && run_index >= 0)
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << gigaflops << "GFlop/s"
                << " " << matrix_host.size() << " " << matrix_host.blockSize() << " " << comm_grid.size()
                << " " << hpx::get_os_thread_count() << std::endl;

    // (optional) run test
    if ((opts.do_check == CholCheckIterFreq::Last && run_index == (opts.nruns - 1)) ||
        opts.do_check == CholCheckIterFreq::All) {
      HostMatrixType original(matrix_size, block_size, comm_grid);
      copy(matrix_ref, original);
      check_cholesky(original, matrix_host, comm_grid);
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
    ("nruns",        value<int64_t>()    ->default_value(   1),       "Number of runs to compute the cholesky")
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

/// Set to zero the upper part of the diagonal tiles
///
/// For the tiles on the diagonal (i.e. row == col), the elements in the upper triangular
/// part of each tile, diagonal excluded, are set to zero.
/// Tiles that are not on the diagonal (i.e. row != col) will not be touched or referenced
void setUpperToZeroForDiagonalTiles(HostMatrixType& matrix) {
  DLAF_ASSERT(dlaf::matrix::square_blocksize(matrix), matrix);

  const auto& distribution = matrix.distribution();

  for (int i_tile_local = 0; i_tile_local < distribution.localNrTiles().rows(); ++i_tile_local) {
    auto k_tile_global = distribution.globalTileFromLocalTile<Coord::Row>(i_tile_local);
    GlobalTileIndex diag_tile{k_tile_global, k_tile_global};

    if (distribution.rankIndex() != distribution.rankGlobalTile(diag_tile))
      continue;

    auto tile_set = unwrapping([](auto&& tile) {
      if (tile.size().rows() > 1)
        lapack::laset(lapack::MatrixType::Upper, tile.size().rows() - 1, tile.size().cols() - 1, 0, 0,
                      tile.ptr({0, 1}), tile.ld());
    });

    matrix(diag_tile).then(tile_set);
  }
}

/// Compute the absolute difference |A - L * L'|
///
/// It computes:
/// 1. L = L * L' (in-place)
/// 2. A = |A - L| (in-place, where L is the updated data from previous step)
///
/// Just the lower triangular part of the matrices will be touched/computed.
///
/// It is used to get the difference matrix between the matrix computed starting from the
/// cholesky factorization and the original one
void cholesky_diff(HostMatrixType& A, HostMatrixType& L, CommunicatorGrid comm_grid) {
  // TODO A and L must be different

  using dlaf::common::make_data;

  // compute tile * tile_to_transpose' with the option to cumulate the result
  auto gemm_f =
      unwrapping([](auto&& tile, auto&& tile_to_transpose, auto&& result, const bool accumulate_result) {
        dlaf::tile::gemm<T>(blas::Op::NoTrans, blas::Op::ConjTrans, 1.0, tile, tile_to_transpose,
                            accumulate_result ? 0.0 : 1.0, result);
      });

  // compute a = abs(a - b)
  auto tile_abs_diff = unwrapping([](auto&& a, auto&& b) {
    for (const auto el_idx : dlaf::common::iterate_range2d(a.size()))
      a(el_idx) = std::abs(a(el_idx) - b(el_idx));
  });

  DLAF_ASSERT(dlaf::matrix::square_size(A), A);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(A), A);

  DLAF_ASSERT(dlaf::matrix::square_size(L), L);
  DLAF_ASSERT(dlaf::matrix::square_blocksize(L), L);

  const auto& distribution = L.distribution();
  const auto current_rank = distribution.rankIndex();

  HostMatrixType mul_result(L.size(), L.blockSize(), comm_grid);

  // k is a global index that keeps track of the diagonal tile
  // it is useful mainly for two reasons:
  // - as limit for when to stop multiplying (because it is triangular and symmetric)
  // - as reference for the row to be used in L, but transposed, as value for L'
  for (SizeType k = 0; k < L.nrTiles().cols(); ++k) {
    const auto k_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Col>(k + 1);

    // workspace for storing the partial results for all the rows in the current rank
    // TODO this size can be reduced to just the part below the current diagonal tile
    HostMatrixType partial_result({distribution.localSize().rows(), L.blockSize().cols()},
                                  L.blockSize());

    // it has to be set to zero, because ranks may not be able to contribute for each row at each step
    // so when the result will be reduced along the rows, they will not alter the final result
    dlaf::matrix::util::set(partial_result, [](auto&&) { return 0; });

    // for each local column, with the limit of the diagonal tile
    for (SizeType j_loc = 0; j_loc < k_loc; ++j_loc) {
      // identify the tile to be used as 2nd operand in the gemm
      const GlobalTileIndex
          transposed_wrt_global{k, distribution.template globalTileFromLocalTile<Coord::Col>(j_loc)};
      const auto owner_transposed = distribution.rankGlobalTile(transposed_wrt_global);

      // collect the 2nd operand, receving it from others if not available locally
      hpx::shared_future<ConstHostTileType> tile_to_transpose;

      if (owner_transposed == current_rank) {  // current rank already has what it needs
        tile_to_transpose = L.read(transposed_wrt_global);

        // if there are more than 1 rank for column, others will need the data from this one
        if (distribution.commGridSize().rows() > 1)
          dlaf::comm::sync::broadcast::send(comm_grid.colCommunicator(),
                                            L.read(transposed_wrt_global).get());
      }
      else {  // current rank has to receive it
        // by construction: this rank has the 1st operand, so if it does not have the 2nd one,
        // for sure another rank in the same column will have it (thanks to the regularity of the
        // distribution given by the 2D grid)
        DLAF_ASSERT_HEAVY(owner_transposed.col() == current_rank.col(), owner_transposed, current_rank);

        HostTileType workspace(L.blockSize(),
                               dlaf::memory::MemoryView<T, Device::CPU>(L.blockSize().linear_size()),
                               L.blockSize().rows());

        dlaf::comm::sync::broadcast::receive_from(owner_transposed.row(), comm_grid.colCommunicator(),
                                                  workspace);

        tile_to_transpose = hpx::make_ready_future<ConstHostTileType>(std::move(workspace));
      }

      // compute the part of results available locally, for each row this rank has in local
      auto i_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(k);
      for (; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
        const LocalTileIndex tile_wrt_local{i_loc, j_loc};

        hpx::dataflow(gemm_f, L.read(tile_wrt_local), tile_to_transpose,
                      partial_result(LocalTileIndex{i_loc, 0}), j_loc == 0);
      }
    }

    // now that each rank has computed its partial result with the local data available
    // aggregate the partial result for each row in the current column k
    for (int i_loc = 0; i_loc < partial_result.nrTiles().rows(); ++i_loc) {
      const auto i = distribution.template globalTileFromLocalTile<Coord::Row>(i_loc);
      const GlobalTileIndex tile_result{i, k};
      const auto owner_result = distribution.rankGlobalTile(tile_result);

      dlaf::common::DataDescriptor<T> output_message;
      if (owner_result == current_rank)
        output_message = make_data(mul_result(tile_result).get());

      dlaf::comm::sync::reduce(owner_result.col(), comm_grid.rowCommunicator(), MPI_SUM,
                               make_data(partial_result.read(LocalTileIndex{i_loc, 0}).get()),
                               output_message);

      // L * L' for the current cell is computed
      // here the owner of the result performs the last step (difference with original)
      if (owner_result == current_rank) {
        hpx::dataflow(tile_abs_diff, A(tile_result), mul_result.read(tile_result));
      }
    }
  }
}

/// Procedure to evaluate the result of the Cholesky factorization
///
/// 1. Compute the max norm of the original matrix
/// 2. Compute the absolute difference between the original and the computed matrix using the factorization
/// 3. Compute the max norm of the difference
/// 4. Evaluate the correctness of the result using the ratio between the two matrix max norms
///
/// Prints a message with the ratio and a note about the error:
/// "":        check ok
/// "ERROR":   error is high, there is an error in the factorization
/// "WARNING": error is slightly high, there can be an error in the factorization
void check_cholesky(HostMatrixType& A, HostMatrixType& L, CommunicatorGrid comm_grid) {
  const Index2D rank_result{0, 0};

  // 1. Compute the max norm of the original matrix in A
  const auto norm_A = dlaf::auxiliary::norm<dlaf::Backend::MC>(comm_grid, rank_result, lapack::Norm::Max,
                                                               blas::Uplo::Lower, A);

  // 2.
  // L is a lower triangular, reset values in the upper part (diagonal excluded)
  // it is needed for the gemm to compute correctly the result when using
  // tiles on the diagonal treating them as all the other ones
  setUpperToZeroForDiagonalTiles(L);

  // compute diff in-place, A = A - L*L'
  cholesky_diff(A, L, comm_grid);

  // 3. Compute the max norm of the difference (it has been compute in-place in A)
  const auto norm_diff =
      dlaf::auxiliary::norm<dlaf::Backend::MC>(comm_grid, rank_result, lapack::Norm::Max,
                                               blas::Uplo::Lower, A);

  // 4.
  // Evaluation of correctness is done just by the master rank
  if (comm_grid.rank() != rank_result)
    return;

  constexpr auto eps = std::numeric_limits<dlaf::BaseType<T>>::epsilon();
  const auto n = A.size().rows();

  const auto diff_ratio = norm_diff / norm_A;

  if (diff_ratio > 100 * eps * n)
    std::cout << "ERROR: ";
  else if (diff_ratio > eps * n)
    std::cout << "Warning: ";

  std::cout << "Max Diff / Max A: " << diff_ratio << std::endl;
}

options_t parse_options(hpx::program_options::variables_map& vm) {
  // clang-format off
  options_t opts = {
      vm["matrix-size"].as<SizeType>(),
      vm["block-size"].as<SizeType>(),
      vm["grid-rows"].as<int>(),
      vm["grid-cols"].as<int>(),
      vm["nruns"].as<int64_t>(),
      vm["nwarmups"].as<int64_t>(),
      parse_chol_check(vm["check-result"].as<std::string>()),
  };
  // clang-format on

  DLAF_ASSERT(opts.m > 0, opts.m);
  DLAF_ASSERT(opts.mb > 0, opts.mb);
  DLAF_ASSERT(opts.grid_rows > 0, opts.grid_rows);
  DLAF_ASSERT(opts.grid_cols > 0, opts.grid_cols);
  DLAF_ASSERT(opts.nruns > 0, opts.nruns);
  DLAF_ASSERT(opts.nwarmups >= 0, opts.nwarmups);

  if (opts.do_check != CholCheckIterFreq::None && opts.m % opts.mb) {
    std::cerr
        << "Warning! At the moment result checking works just with matrix sizes that are multiple of the block size."
        << std::endl;
    opts.do_check = CholCheckIterFreq::None;
  }

  return opts;
}

CholCheckIterFreq parse_chol_check(const std::string& check) {
  if (check == "all")
    return CholCheckIterFreq::All;
  else if (check == "last")
    return CholCheckIterFreq::Last;
  else if (check == "none")
    return CholCheckIterFreq::None;

  std::cout << "Parsing is not implemented for --check-result=" << check << "!" << std::endl;
  std::terminate();
  return CholCheckIterFreq::None;  // unreachable
}

}
