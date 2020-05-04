//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include <dlaf/blas_tile.h>
#include <dlaf/communication/datatypes.h>
#include <dlaf/communication/init.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/pool.h>
#include <dlaf/matrix.h>

#include <hpx/async/dataflow.hpp>
#include <hpx/execution/executors/pool_executor.hpp>  // > HPX v.1.4.1
#include <hpx/hpx_init.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>

#ifdef DLAF_WITH_MPI_FUTURES
#include <hpx/mpi/mpi_executor.hpp>
#include <hpx/mpi/mpi_future.hpp>
#endif

#include <chrono>
#include <complex>
#include <cstdio>

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

using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::matrix::Distribution;
using hpx::program_options::variables_map;

using ScalarType = std::complex<double>;
using MatrixType = dlaf::Matrix<ScalarType, dlaf::Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const ScalarType, dlaf::Device::CPU>;
#ifdef DLAF_WITH_MPI_FUTURES
using ExecutorType = hpx::mpi::executor;
#else
using ExecutorType = hpx::parallel::execution::pool_executor;
#endif

void sirius_gemm(int batch_size, ExecutorType const& mpi_executor, CommunicatorGrid& comm_grid,
                 ConstMatrixType& a_mat, ConstMatrixType& b_mat, MatrixType& cini_mat,
                 MatrixType& cfin_mat);

// Initialize matrix
void init_matrix(MatrixType& matrix, ScalarType val);

// Wait for all tiles of the matrix.
void waitall_tiles(MatrixType& matrix);

// Sum the elements of the matrix. Useful for debugging.
ScalarType sum_matrix(Communicator const& comm, MatrixType& matrix);

struct params {
  bool check;

  int num_iters;
  int len_m;
  int len_n;
  int len_k;
  int tile_m;
  int tile_n;
  int pgrid_rows;
  int pgrid_cols;
  int batch_size;
};

// Initialize parameters and check for consistency
params init_params(variables_map&);

}  // end namespace

int hpx_main(::variables_map& vm) {
  using dlaf::common::Ordering;
  using dlaf::TileElementSize;
  using dlaf::GlobalElementSize;
  using dlaf::LocalElementSize;

  using clock_t = std::chrono::high_resolution_clock;
  using seconds_t = std::chrono::duration<double>;

  // Input
  ::params ps = ::init_params(vm);

  // Communicators
  ::Communicator comm_world(MPI_COMM_WORLD);
  ::CommunicatorGrid comm_grid(comm_world, ps.pgrid_rows, ps.pgrid_cols, Ordering::ColumnMajor);

  // Matrices `A` and `B`
  // The matrices are distributed only along the `k` dimension. In SIRIUS, the sections assigned to each
  // process are not exactly equal, they differ by a little in non-trivial ways. SIRIUS's distribution
  // for A and B is NOT a special case of block cyclic distribution. In this miniapp, the distribution is
  // emulated by DLAF local matrices.
  int num_procs = comm_world.size();
  int rank = comm_world.rank();
  int k_loc = ps.len_k / num_procs + ((rank < ps.len_k % num_procs) ? 1 : 0);
  ::MatrixType a_mat(LocalElementSize(k_loc, ps.len_m), TileElementSize(k_loc, ps.tile_m));
  ::MatrixType b_mat(LocalElementSize(k_loc, ps.len_n), TileElementSize(k_loc, ps.tile_n));

  // Matrices `C`-initial and `C`-final
  using dlaf::matrix::tileLayout;
  MatrixType cini_mat(Distribution(LocalElementSize(ps.len_m, ps.len_n),
                                   TileElementSize(ps.tile_m, ps.tile_n)),
                      tileLayout(LocalElementSize(ps.len_m, ps.len_n),
                                 TileElementSize(ps.tile_m, ps.tile_n)));
  MatrixType cfin_mat(GlobalElementSize(ps.len_m, ps.len_n), TileElementSize(ps.tile_m, ps.tile_n),
                      comm_grid);

  // Initialize matrices
  init_matrix(a_mat, ::ScalarType(1));
  init_matrix(b_mat, ::ScalarType(2));

  // 1. John's branch
  // 2. MPI pool with a single core
  // 3. Default pool with high priority executor for MPI
#if defined(DLAF_WITH_MPI_FUTURES)
  // This needs remain in scope for all uses of hpx::mpi
  std::string pool_name = "default";
  hpx::mpi::enable_user_polling enable_polling(pool_name);
  ExecutorType mpi_executor(comm_grid.fullCommunicator());
#else
  using hpx::threads::thread_priority::thread_priority_high;
  auto pool_name = (hpx::resource::pool_exists("mpi")) ? "mpi" : "default";  // New API v1.5
  ExecutorType mpi_executor(pool_name, thread_priority_high);
#endif

  // Benchmark calls of `sirius_gemm`
  //
  for (int i = 0; i < ps.num_iters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = clock_t::now();

    sirius_gemm(ps.batch_size, mpi_executor, comm_grid, a_mat, b_mat, cini_mat, cfin_mat);
    waitall_tiles(cfin_mat);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = clock_t::now();

    if (rank == 0) {
      std::printf("%d: t_tot  [s] = %.5f\n", i, seconds_t(t_end - t_start).count());
    }

    // Simple check
    if (ps.check) {
      ScalarType cfin_sum = sum_matrix(comm_world, cfin_mat);
      if (rank == 0) {
        std::cout << cfin_sum << '\n';
      }
    }
  }

  // Upon exit the mpi/user polling RAII object will stop polling
  return hpx::finalize();
}

// Example usage:
//
//   mpirun -np 1 tsgemm --len_m      100  --len_n      100  --len_k  10000
//                       --tile_m      32  --tile_n      32
//                       --pgrid_rows   1  --pgrid_cols   1
//
int main(int argc, char** argv) {
  // Initialize MPI
  dlaf::comm::InitMPI mpi_init(argc, argv, MPI_THREAD_MULTIPLE);

  // Declare options
  namespace po = hpx::program_options;
  po::options_description desc("Allowed options.");

  // clang-format off
  desc.add_options()
     ("check",      po::bool_switch() -> default_value(false) , "Print the sum of elements of the resulting matrix.")
     ("mpipool",    po::bool_switch() -> default_value(false)   , "Dedicate a core to MPI if available.")

     ("num_iters",  po::value<int>()  -> default_value(   5)  , "number of iterations")
     ("batch_size", po::value<int>()  -> default_value(  16)  , "number of tiles batched for computation/communication")
     ("len_m",      po::value<int>()  -> default_value( 100)  , "m dimension")
     ("len_n",      po::value<int>()  -> default_value( 100)  , "n dimension")
     ("len_k",      po::value<int>()  -> default_value(1000)  , "k dimension")
     ("tile_m",     po::value<int>()  -> default_value(  32)  , "tile m dimension")
     ("tile_n",     po::value<int>()  -> default_value(  32)  , "tile n dimension")
     ("pgrid_rows", po::value<int>()  -> default_value(   1)  , "process grid rows")
     ("pgrid_cols", po::value<int>()  -> default_value(   1)  , "process grid columns")
  ;
  // clang-format on

  // Init MPI pool is requested
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).allow_unregistered().options(desc).run(), vm);
  bool use_mpi_pool = vm["mpipool"].as<bool>();

  if (use_mpi_pool) {
    dlaf::comm::init_mpi_pool(desc, argc, argv);
  }

  // Start the HPX runtime
  return hpx::init(desc, argc, argv);
}

namespace {

using dlaf::TileElementIndex;
using dlaf::SizeType;
using dlaf::GlobalTileIndex;
using dlaf::LocalTileIndex;
using dlaf::LocalTileSize;
using dlaf::common::iterateRange2D;

using TileType = dlaf::Tile<ScalarType, dlaf::Device::CPU>;
using ConstTileType = dlaf::Tile<const ScalarType, dlaf::Device::CPU>;

// Send Cini tile to the process it belongs (`tile_rank`)
void send_tile(ConstTileType const& c_tile, SizeType tile_rank, SizeType tag, MPI_Comm comm) {
  SizeType num_elements = c_tile.ld() * c_tile.size().cols();
  void const* buf = c_tile.ptr(TileElementIndex(0, 0));
  MPI_Datatype dtype = dlaf::comm::mpi_datatype<ScalarType>::type;

  MPI_Request req;
  MPI_Isend(buf, num_elements, dtype, tile_rank, tag, comm, &req);
  hpx::util::yield_while([&req] {
    int flag;
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    return flag == 0;
  });
}

// Reduce into Cini tile from all other processes
void recv_tile(TileType& c_tile, int this_rank, SizeType tag, MPI_Comm comm) {
  int nprocs = Communicator(MPI_COMM_WORLD).size() - 1;
  SizeType nelems = c_tile.ld() * c_tile.size().cols();
  ScalarType* tile_buf = c_tile.ptr(TileElementIndex(0, 0));
  MPI_Datatype dtype = dlaf::comm::mpi_datatype<ScalarType>::type;

  // Note: these allocations should be better made from a pool
  // auto message = dlaf::comm::make_message(dlaf::common::make_data(c_tile));

  std::vector<MPI_Request> reqs(nprocs);
  std::vector<ScalarType> staging_buf(nprocs * nelems);

  // Issue receives
  for (int r_idx = 0; r_idx < nprocs; ++r_idx) {
    int rank = (r_idx < this_rank) ? r_idx : r_idx + 1;  // skip `this_rank`
    ScalarType* rcv_buf = staging_buf.data() + r_idx * nelems;
    MPI_Irecv(rcv_buf, nelems, dtype, rank, tag, comm, &reqs[r_idx]);
  }

  // Yield until all issued receives completed.
  hpx::util::yield_while([&reqs, nprocs] {
    int flag;
    MPI_Testall(nprocs, reqs.data(), &flag, MPI_STATUSES_IGNORE);
    return flag == 0;
  });

  // Do the reduction in tile_buf
  for (int r_idx = 0; r_idx < nprocs; ++r_idx) {
    ScalarType const* rcv_buf = staging_buf.data() + r_idx * nelems;
    for (int el_idx = 0; el_idx < nelems; ++el_idx) {
      tile_buf[el_idx] += rcv_buf[el_idx];
    }
  }
}

void offload_tile(ConstTileType const& cini_tile, TileType& cfin_tile) {
  for (auto idx : iterateRange2D(cini_tile.size())) {
    cfin_tile(idx) = cini_tile(idx);
  }
}

void sirius_gemm(int batch_size, ExecutorType const& mpi_executor, CommunicatorGrid& comm_grid,
                 ConstMatrixType& a_mat, ConstMatrixType& b_mat, MatrixType& cini_mat,
                 MatrixType& cfin_mat) {
  using hpx::util::unwrapping;
  using hpx::util::annotated_function;
  using dlaf::common::computeLinearIndexColMajor;

  MPI_Comm mpi_comm(comm_grid.fullCommunicator());
  ::Distribution const& cfin_dist = cfin_mat.distribution();
  int this_rank = comm_grid.rankFullCommunicator(cfin_dist.rankIndex());
  dlaf::LocalTileSize const& tile_grid_size = cini_mat.distribution().localNrTiles();
  auto dep_tile_idx = dlaf::common::computeCoordsColMajor(batch_size, tile_grid_size);
  for (auto cloc_idx : iterateRange2D(tile_grid_size)) {
    LocalTileIndex a_idx(0, cloc_idx.row());
    LocalTileIndex b_idx(0, cloc_idx.col());
    GlobalTileIndex c_idx(cloc_idx.row(), cloc_idx.col());

    int tile_rank = comm_grid.rankFullCommunicator(cfin_dist.rankGlobalTile(c_idx));
    int tile_tag = computeLinearIndexColMajor(cloc_idx, tile_grid_size);

    // Order tasks such that tiles are computed/communicated in batches of `batch_size`
    SizeType c_dep_i = cloc_idx.row() - dep_tile_idx.row();
    SizeType c_dep_j = cloc_idx.col() - dep_tile_idx.col();
    bool c_dep_exists = c_dep_i < 0 || c_dep_j < 0;
    hpx::shared_future<void> c_dep_tile_fut =
        (c_dep_exists) ? hpx::make_ready_future()
                       : hpx::shared_future<void>(cini_mat.read(GlobalTileIndex(c_dep_i, c_dep_j)));

    // GEMM
    auto gemm_f = unwrapping(annotated_function(
        [](auto&& a_tile, auto&& b_tile, auto&& c_tile) {
          dlaf::tile::gemm(blas::Op::Trans, blas::Op::NoTrans, ScalarType(1), a_tile, b_tile,
                           ScalarType(0), c_tile);
        },
        "gemm"));
    hpx::dataflow(gemm_f, a_mat.read(a_idx), b_mat.read(b_idx), cini_mat(c_idx), c_dep_tile_fut);

    if (this_rank == tile_rank) {
      // RECV
      auto recv_f = unwrapping(
          annotated_function([=](auto&& c_tile) { recv_tile(c_tile, this_rank, tile_tag, mpi_comm); },
                             "recv"));
      hpx::dataflow(mpi_executor, recv_f, cini_mat(c_idx));

      // OFFLOAD
      auto offload_f =
          unwrapping(annotated_function([](auto&& cini_tile,
                                           auto&& cfin_tile) { offload_tile(cini_tile, cfin_tile); },
                                        "offload"));
      hpx::dataflow(offload_f, cini_mat.read(c_idx), cfin_mat(c_idx));
    }
    else {
      // SEND
      auto send_f = unwrapping(
          annotated_function([=](auto&& c_tile) { send_tile(c_tile, tile_rank, tile_tag, mpi_comm); },
                             "send"));
      hpx::dataflow(mpi_executor, send_f, cini_mat(c_idx));
    }
  }
}

void init_matrix(MatrixType& matrix, ScalarType val) {
  for (auto tile_idx : iterateRange2D(matrix.distribution().localNrTiles())) {
    TileType tile = matrix(tile_idx).get();
    for (auto el_idx : iterateRange2D(tile.size())) {
      tile(el_idx) = val;
    }
  }
}

void waitall_tiles(MatrixType& matrix) {
  for (auto tile_idx : iterateRange2D(matrix.distribution().localNrTiles())) {
    matrix(tile_idx).get();
  }
}

// Sums the distributed matrix and returns the result to process 0.
ScalarType sum_matrix(Communicator const& comm, MatrixType& matrix) {
  ScalarType local_sum = 0;

  for (auto tile_idx : iterateRange2D(matrix.distribution().localNrTiles())) {
    TileType tile = matrix(tile_idx).get();
    for (auto el_idx : iterateRange2D(tile.size())) {
      local_sum += tile(el_idx);
    }
  }

  ScalarType global_sum = 0;
  MPI_Datatype mpi_type = dlaf::comm::mpi_datatype<ScalarType>::type;
  MPI_Reduce(&local_sum, &global_sum, 1, mpi_type, MPI_SUM, 0, comm);

  return global_sum;
}

params init_params(variables_map& vm) {
  using dlaf::util::ceilDiv;

  bool check = vm["check"].as<bool>();

  int num_iters = vm["num_iters"].as<int>();
  int batch_size = vm["batch_size"].as<int>();
  int len_m = vm["len_m"].as<int>();
  int len_n = vm["len_n"].as<int>();
  int len_k = vm["len_k"].as<int>();
  int tile_m = vm["tile_m"].as<int>();
  int tile_n = vm["tile_n"].as<int>();
  int pgrid_rows = vm["pgrid_rows"].as<int>();
  int pgrid_cols = vm["pgrid_cols"].as<int>();

  int rank;
  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (num_procs != pgrid_rows * pgrid_cols) {
    std::printf("[ERROR] Number of processes doesn't match the process grid size\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (tile_m > len_m) {
    std::printf("[ERROR] `tile_m` > `m`.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (tile_n > len_n) {
    std::printf("[ERROR] `tile_n` > `n`.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int ntiles_m = ceilDiv(len_m, tile_m);
  int ntiles_n = ceilDiv(len_n, tile_n);
  if (pgrid_rows > ntiles_m) {
    std::printf("[ERROR] There are more processes along `m` than tiles.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (pgrid_cols > ntiles_n) {
    std::printf("[ERROR] There are more processes along `n` than tiles.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Check if too many non-blocking communications are being issued.
  if (rank == 0 && batch_size > 1000) {
    std::printf("[WARNING] There are too many tiles batched, this may "
                "result in slowdowns as the number of issued non-blocking "
                "communications at each process is proportianal to the batch size.\n");
  }

  // Setup
  if (rank == 0) {
    std::printf("len mnk  = %d %d %d\n", len_m, len_n, len_k);
    std::printf("tile mnk = %d %d\n", tile_m, tile_n);
    std::printf("pgrid    = %d %d\n", pgrid_rows, pgrid_cols);
  }

  return params{check,  num_iters, len_m,      len_n,      len_k,
                tile_m, tile_n,    pgrid_rows, pgrid_cols, batch_size};
}

}
