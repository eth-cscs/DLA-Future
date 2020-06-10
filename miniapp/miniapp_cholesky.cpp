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

namespace {

using hpx::util::unwrapping;

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
using MatrixType = dlaf::Matrix<T, Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const T, Device::CPU>;
using TileType = dlaf::Tile<T, Device::CPU>;
using ConstTileType = dlaf::Tile<const T, Device::CPU>;

/// Check Cholesky Factorization results
///
/// Given a matrix A (Hermitian Positive Definite) and its Cholesky factorization in L,
/// this function checks that A == L * L'
void check_cholesky(MatrixType& A, MatrixType& L, CommunicatorGrid comm_grid);

enum class CHECK_RESULT { NONE, LAST, ALL };

struct options_t {
  SizeType m;
  SizeType mb;
  int grid_rows;
  int grid_cols;
  int64_t nruns;
  CHECK_RESULT do_check;
};

/// Handle CLI options
options_t check_options(hpx::program_options::variables_map& vm);

}

int hpx_main(hpx::program_options::variables_map& vm) {
  options_t opts = check_options(vm);

  Communicator world(MPI_COMM_WORLD);
  CommunicatorGrid comm_grid(world, opts.grid_rows, opts.grid_cols, Ordering::ColumnMajor);

  // Allocate memory for the matrix
  GlobalElementSize matrix_size(opts.m, opts.m);
  TileElementSize block_size(opts.mb, opts.mb);

  MatrixType matrix(matrix_size, block_size, comm_grid);
  const auto& distribution = matrix.distribution();

  for (auto run_index = 0; run_index < opts.nruns; ++run_index) {
    if (0 == world.rank())
      std::cout << "[" << run_index << "]" << std::endl;

    // TODO this should be a clone of the original reference one
    dlaf::matrix::util::set_random_hermitian_positive_definite(matrix);

    // wait all setup tasks before starting benchmark
    {
      for (const auto tile_idx : dlaf::common::iterate_range2d(distribution.localNrTiles()))
        matrix(tile_idx).get();
      MPI_Barrier(world);
    }

    dlaf::common::Timer<> timeit;
    dlaf::Factorization<Backend::MC>::cholesky(comm_grid, blas::Uplo::Lower, matrix);

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
      gigaflops = dlaf::total_ops<T>(add_mul, add_mul) / elapsed_time / 1e9;
    }

    // print benchmark results
    if (0 == world.rank())
      std::cout << "[" << run_index << "]"
                << " " << elapsed_time << "s"
                << " " << gigaflops << "GFlop/s"
                << " " << matrix.size() << " " << matrix.blockSize() << " " << comm_grid.size() << " "
                << hpx::get_os_thread_count() << std::endl;

    // (optional) run test
    if (opts.do_check != CHECK_RESULT::NONE) {
      if (opts.do_check == CHECK_RESULT::LAST && run_index != (opts.nruns - 1))
        continue;

      // TODO this should be a clone of the original one
      MatrixType original(matrix_size, block_size, comm_grid);
      dlaf::matrix::util::set_random_hermitian_positive_definite(original);

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
    ("matrix-size",  value<SizeType>()   ->default_value(4096),                        "Matrix size")
    ("block-size",   value<SizeType>()   ->default_value( 256),                        "Block cyclic distribution size")
    ("grid-rows",    value<int>()        ->default_value(   1),                        "Number of row processes in the 2D communicator")
    ("grid-cols",    value<int>()        ->default_value(   1),                        "Number of column processes in the 2D communicator")
    ("nruns",        value<int64_t>()    ->default_value(   1),                        "Number of runs to compute the cholesky")
    ("check-result", value<std::string>()->default_value(  "")->implicit_value("all"), "Enable result check ('all', 'last')")
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

/// Compute max norm of the lower triangular matrix
T matrix_norm(ConstMatrixType& matrix, CommunicatorGrid comm_grid) {
  using dlaf::common::internal::vector;
  using dlaf::common::make_data;

  // TODO unwrapping can be skipped for optimization reasons
  auto max_f = unwrapping([](const auto&& values) {
    // some rank may not have any element
    if (values.size() == 0)
      return std::numeric_limits<T>::min();
    return *std::max_element(values.begin(), values.end());
  });

  const auto& distribution = matrix.distribution();

  vector<hpx::future<T>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each tile in local (lower triangular), create a task that finds the max element in the tile
  for (SizeType j_loc = 0; j_loc < distribution.localNrTiles().cols(); ++j_loc) {
    const SizeType j = distribution.template globalTileFromLocalTile<Coord::Col>(j_loc);
    const SizeType i_diag_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(j);

    for (SizeType i_loc = i_diag_loc; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
      auto current_tile_max =
          hpx::dataflow(unwrapping([is_diag = (i_loc == i_diag_loc)](auto&& tile) -> T {
                          T tile_max_value = std::abs(T{0});
                          for (SizeType j = 0; j < tile.size().cols(); ++j)
                            for (SizeType i = is_diag ? j : 0; i < tile.size().rows(); ++i)
                              tile_max_value = std::max(tile_max_value, tile({i, j}));
                          return tile_max_value;
                        }),
                        matrix.read(LocalTileIndex{i_loc, j_loc}));

      tiles_max.emplace_back(std::move(current_tile_max));
    }
  }

  // than it is necessary to reduce max values from all ranks into a single max value for the matrix

  auto local_max_value = hpx::dataflow(max_f, tiles_max).get();

  T max_value_rows;
  if (comm_grid.size().cols() > 1)
    dlaf::comm::sync::reduce(0, comm_grid.rowCommunicator(), MPI_MAX, make_data(&local_max_value, 1),
                             make_data(&max_value_rows, 1));
  else
    max_value_rows = local_max_value;

  T max_value;
  if (comm_grid.size().rows() > 1)
    dlaf::comm::sync::reduce(0, comm_grid.colCommunicator(), MPI_MAX, make_data(&max_value_rows, 1),
                             make_data(&max_value, 1));
  else
    max_value = max_value_rows;

  return max_value;
}

/// Set to zero the upper part of the diagonal tiles
///
/// For the tiles on the diagonal (i.e. row == col), the elements in the upper triangular
/// part of each tile, diagonal excluded, are set to zero.
/// Tiles that are not on the diagonal (i.e. row != col) will not be touched or referenced
void setUpperToZeroForDiagonalTiles(MatrixType& matrix) {
  DLAF_ASSERT_BLOCKSIZE_SQUARE(matrix);

  const auto& distribution = matrix.distribution();

  for (int i_tile_local = 0; i_tile_local < distribution.localNrTiles().rows(); ++i_tile_local) {
    auto k_tile_global = distribution.globalTileFromLocalTile<Coord::Row>(i_tile_local);
    GlobalTileIndex diag_tile{k_tile_global, k_tile_global};

    if (distribution.rankIndex() != distribution.rankGlobalTile(diag_tile))
      continue;

    auto tile_set = unwrapping([](auto&& tile) {
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
void cholesky_diff(MatrixType& A, MatrixType& L, CommunicatorGrid comm_grid) {
  // TODO A and L must be different

  using dlaf::common::make_data;
  using dlaf::util::size_t::mul;

  // compute tile * tile_to_transpose' with the option to cumulate the result
  auto gemm_f =
      unwrapping([](auto&& tile, auto&& tile_to_transpose, auto&& result, const bool accumulate_result) {
        dlaf::tile::gemm<T, Device::CPU>(blas::Op::NoTrans, blas::Op::ConjTrans, 1.0, tile,
                                         tile_to_transpose, accumulate_result ? 0.0 : 1.0, result);
      });

  // compute a = abs(a - b)
  auto tile_abs_diff = unwrapping([](auto&& a, auto&& b) {
    for (const auto el_idx : dlaf::common::iterate_range2d(a.size()))
      a(el_idx) = std::abs(a(el_idx) - b(el_idx));
  });

  DLAF_ASSERT_SIZE_SQUARE(A);
  DLAF_ASSERT_BLOCKSIZE_SQUARE(A);

  DLAF_ASSERT_SIZE_SQUARE(L);
  DLAF_ASSERT_BLOCKSIZE_SQUARE(L);

  const auto& distribution = L.distribution();
  const auto current_rank = distribution.rankIndex();

  MatrixType mul_result(L.size(), L.blockSize(), comm_grid);

  // k is a global index that keeps track of the diagonal tile
  // it is useful mainly for two reasons:
  // - as limit for when to stop multiplying (because it is triangular and symmetric)
  // - as reference for the row to be used in L, but transposed, as value for L'
  for (SizeType k = 0; k < L.nrTiles().cols(); ++k) {
    const auto k_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Col>(k + 1);

    // workspace for storing the partial results for all the rows in the current rank
    // TODO this size can be reduced to just the part below the current diagonal tile
    MatrixType partial_result({distribution.localSize().rows(), L.blockSize().cols()}, L.blockSize());

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
      hpx::shared_future<dlaf::Tile<const T, Device::CPU>> tile_to_transpose;

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
        DLAF_ASSERT_HEAVY(owner_transposed.col() == current_rank.col());

        TileType workspace(L.blockSize(),
                           dlaf::memory::MemoryView<T, Device::CPU>(
                               mul(L.blockSize().rows(), L.blockSize().cols())),
                           L.blockSize().rows());

        dlaf::comm::sync::broadcast::receive_from(owner_transposed.row(), comm_grid.colCommunicator(),
                                                  workspace);

        tile_to_transpose = hpx::make_ready_future<ConstTileType>(std::move(workspace));
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
void check_cholesky(MatrixType& A, MatrixType& L, CommunicatorGrid comm_grid) {
  // 1. Compute the max norm of the original matrix in A
  float norm_A = matrix_norm(A, comm_grid);

  // 2.
  // L is a lower triangular, reset values in the upper part (diagonal excluded)
  // it is needed for the gemm to compute correctly the result when using
  // tiles on the diagonal treating them as all the other ones
  setUpperToZeroForDiagonalTiles(L);

  // compute diff in-place, A = A - L*L'
  cholesky_diff(A, L, comm_grid);

  // 3. Compute the max norm of the difference (it has been compute in-place in A)
  float norm_diff = matrix_norm(A, comm_grid);

  // 4.
  // Evaluation of correctness is done just by the master rank
  if (comm_grid.rank() != Index2D{0, 0})
    return;

  constexpr auto eps = std::numeric_limits<T>::epsilon();
  const auto n = A.size().rows();

  const auto diff_ratio = norm_diff / norm_A;

  if (diff_ratio > 100 * eps * n)
    std::cout << "ERROR: ";
  else if (diff_ratio > eps * n)
    std::cout << "Warning: ";

  std::cout << "Max Diff / Max A: " << diff_ratio << std::endl;
}

options_t check_options(hpx::program_options::variables_map& vm) {
  options_t opts = {
      vm["matrix-size"].as<SizeType>(), vm["block-size"].as<SizeType>(),
      vm["grid-rows"].as<int>(),        vm["grid-cols"].as<int>(),

      vm["nruns"].as<int64_t>(),        CHECK_RESULT::NONE,
  };

  if (opts.m <= 0)
    throw std::runtime_error("matrix size must be a positive number");
  if (opts.mb <= 0)
    throw std::runtime_error("block size must be a positive number");

  if (opts.grid_rows <= 0)
    throw std::runtime_error("number of grid rows must be a positive number");
  if (opts.grid_cols <= 0)
    throw std::runtime_error("number of grid columns must be a positive number");

  const std::string check_type = vm["check-result"].as<std::string>();

  if (check_type.compare("all") == 0)
    opts.do_check = CHECK_RESULT::ALL;
  else if (check_type.compare("last") == 0)
    opts.do_check = CHECK_RESULT::LAST;
  else if (check_type.compare("") != 0)
    throw std::runtime_error(check_type + " is not a valid value for check-result");

  if (opts.do_check != CHECK_RESULT::NONE && opts.m % opts.mb) {
    std::cerr
        << "Warning! At the moment result checking works just with matrix sizes that are multiple of the block size."
        << std::endl;
    opts.do_check = CHECK_RESULT::NONE;
  }

  return opts;
}

}
