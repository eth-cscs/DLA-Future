//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/blas_tile.h>
#include <dlaf/communication/datatypes.h>
#include <dlaf/matrix.h>

#include <hpx/dataflow.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
//#include <hpx/execution/executors/pool_executor.hpp> // > HPX v.1.4.1

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

using dlaf::SizeType;
using dlaf::common::Ordering;
using dlaf::comm::Communicator;
using dlaf::comm::CommunicatorGrid;
using dlaf::TileElementSize;
using dlaf::TileElementIndex;
using dlaf::GlobalElementSize;
using dlaf::GlobalTileSize;
using dlaf::GlobalTileIndex;
using dlaf::LocalElementSize;
using dlaf::LocalTileIndex;
using dlaf::LocalTileSize;
using dlaf::matrix::Distribution;
using dlaf::matrix::LayoutInfo;
using CommSize = dlaf::comm::Size2D;
using CommIndex = dlaf::comm::Index2D;
using ScalarType = std::complex<double>;
using MatrixType = dlaf::Matrix<ScalarType, dlaf::Device::CPU>;
using ConstMatrixType = dlaf::Matrix<const ScalarType, dlaf::Device::CPU>;
using TileType = dlaf::Tile<ScalarType, dlaf::Device::CPU>;
using ConstTileType = dlaf::Tile<const ScalarType, dlaf::Device::CPU>;

using hpx::program_options::variables_map;
using hpx::program_options::options_description;

#ifdef DLAF_WITH_MPI_FUTURES
using ExecutorType = hpx::mpi::executor;
#else
using ExecutorType = hpx::threads::executors::pool_executor;
// using ExecutorType = hpx::parallel::execution::pool_executor; // > HPX v.1.4.1
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
  int num_iters;
  int len_m;
  int len_n;
  int len_k;
  int tile_m;
  int tile_n;
  int pgrid_rows;
  int pgrid_cols;
  int batch_size;
  std::string setup;
  bool check;
};

// Initialize parameters and check for consistency
params init_params(variables_map&);

// Handle input options
options_description init_desc();

}  // end namespace

int hpx_main(::variables_map& vm) {
  using clock_t = std::chrono::high_resolution_clock;
  using seconds_t = std::chrono::duration<double>;

  // Input
  ::params ps = ::init_params(vm);

  // Communicators
  ::Communicator comm_world(MPI_COMM_WORLD);
  ::CommunicatorGrid comm_grid(comm_world, ps.pgrid_rows, ps.pgrid_cols, ::Ordering::ColumnMajor);

  // Matrices `A` and `B`
  // The matrices are distributed only along the `k` dimension. In SIRIUS, the sections assigned to each
  // process are not exactly equal, they differ by a little in non-trivial ways. SIRIUS's distribution
  // for A and B is NOT a special case of block cyclic distribution. In this miniapp, the distribution is
  // emulated by DLAF local matrices.
  int num_procs = comm_world.size();
  int rank = comm_world.rank();
  int k_loc = ps.len_k / num_procs + ((rank < ps.len_k % num_procs) ? 1 : 0);
  ::MatrixType a_mat(::LocalElementSize(k_loc, ps.len_m), ::TileElementSize(k_loc, ps.tile_m));
  ::MatrixType b_mat(::LocalElementSize(k_loc, ps.len_n), ::TileElementSize(k_loc, ps.tile_n));

  // Matrices `C`-initial and `C`-final
  using dlaf::matrix::tileLayout;
  ::MatrixType cini_mat(::Distribution(::LocalElementSize(ps.len_m, ps.len_n),
                                       ::TileElementSize(ps.tile_m, ps.tile_n)),
                        tileLayout(::LocalElementSize(ps.len_m, ps.len_n),
                                   ::TileElementSize(ps.tile_m, ps.tile_n)));
  ::MatrixType cfin_mat(::GlobalElementSize(ps.len_m, ps.len_n), ::TileElementSize(ps.tile_m, ps.tile_n),
                        comm_grid);

  // Initialize matrices
  init_matrix(a_mat, ::ScalarType(1));
  init_matrix(b_mat, ::ScalarType(2));

  // 1. John's branch                               (mpi_futures)
  // 2. MPI pool with a single core                 (mpi_pool)
  // 3. Default pool with high priority executor    (priorities)
  // 4. Default pool with default priority executor (default)
#if defined(DLAF_WITH_MPI_FUTURES)
  // This needs remain in scope for all uses of hpx::mpi
  std::string pool_name = "default";
  hpx::mpi::enable_user_polling enable_polling(pool_name);
  ExecutorType mpi_executor(comm_grid.fullCommunicator());
#else
  using hpx::threads::thread_priority;
  std::string pool_name = (ps.setup == "mpi_pool") ? "mpi" : "default";
  auto priority = (ps.setup == "priorities") ? thread_priority::thread_priority_high
                                             : thread_priority::thread_priority_default;
  ExecutorType mpi_executor(pool_name, priority);
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
  // Flush printf
  setbuf(stdout, nullptr);

  // Initialize MPI
  int thd_required = MPI_THREAD_MULTIPLE;
  int thd_provided;
  MPI_Init_thread(&argc, &argv, thd_required, &thd_provided);

  if (thd_required != thd_provided) {
    std::fprintf(stderr, "Provided MPI threading model does not match the required one.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Declare options before creating resource partitioner
  namespace po = hpx::program_options;
  po::options_description desc_cmdline = ::init_desc();
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).allow_unregistered().options(desc_cmdline).run(), vm);
  std::string setup = vm["setup"].as<std::string>();

  if (setup == "mpi_pool") {
    int num_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    if (num_procs > 1) {  // if more than 1 process
      // Create a thread pool for MPI work and add (enabled) PUs on the first core
      //
      // Note: Thread pools must be declared before starting the runtime
      hpx::resource::partitioner rp(desc_cmdline, argc, argv);
      if (rp.numa_domains()[0].cores().size() != 1) {  // if more than 1 core
        rp.create_thread_pool("mpi");
        rp.add_resource(rp.numa_domains()[0].cores()[0].pus(), "mpi");
      }
    }
  }

  // Start the HPX runtime
  hpx::init(desc_cmdline, argc, argv);

  MPI_Finalize();
}

namespace {

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
  TileElementSize tile_size = cini_tile.size();
  for (SizeType j = 0; j < tile_size.cols(); ++j) {
    for (SizeType i = 0; i < tile_size.rows(); ++i) {
      TileElementIndex idx(i, j);
      cfin_tile(idx) = cini_tile(idx);
    }
  }
}

void sirius_gemm(int batch_size, ExecutorType const& mpi_executor, CommunicatorGrid& comm_grid,
                 ConstMatrixType& a_mat, ConstMatrixType& b_mat, MatrixType& cini_mat,
                 MatrixType& cfin_mat) {
  using hpx::util::unwrapping;
  using hpx::util::annotated_function;

  MPI_Comm mpi_comm(comm_grid.fullCommunicator());
  ::Distribution const& cfin_dist = cfin_mat.distribution();
  int this_rank = comm_grid.rankFullCommunicator(cfin_dist.rankIndex());
  dlaf::LocalTileSize const& tile_grid_size = cini_mat.distribution().localNrTiles();

  SizeType dep_tile_j = batch_size / tile_grid_size.rows();
  SizeType dep_tile_i = batch_size - dep_tile_j * tile_grid_size.rows();

  for (SizeType tile_j = 0; tile_j < tile_grid_size.cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < tile_grid_size.rows(); ++tile_i) {
      LocalTileIndex a_idx(0, tile_i);
      LocalTileIndex b_idx(0, tile_j);
      GlobalTileIndex c_idx(tile_i, tile_j);

      int tile_rank = comm_grid.rankFullCommunicator(cfin_dist.rankGlobalTile(c_idx));
      int tile_tag = dlaf::common::computeLinearIndex(Ordering::ColumnMajor, c_idx,
                                                      {tile_grid_size.rows(), tile_grid_size.cols()});

      // Order tasks such that tiles are computed/communicated in batches of `batch_size`
      SizeType c_dep_i = tile_i - dep_tile_i;
      SizeType c_dep_j = tile_j - dep_tile_j;
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
}

void init_matrix(MatrixType& matrix, ScalarType val) {
  Distribution const& dist = matrix.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      TileType tile = matrix(LocalTileIndex(tile_i, tile_j)).get();
      TileElementSize tile_size = tile.size();
      for (SizeType j = 0; j < tile_size.cols(); ++j) {
        for (SizeType i = 0; i < tile_size.rows(); ++i) {
          tile(TileElementIndex(i, j)) = val;
        }
      }
    }
  }
}

void waitall_tiles(MatrixType& matrix) {
  Distribution const& dist = matrix.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      matrix(LocalTileIndex(tile_i, tile_j)).get();
    }
  }
}

// Sums the distributed matrix and returns the result to process 0.
ScalarType sum_matrix(Communicator const& comm, MatrixType& matrix) {
  ScalarType local_sum = 0;
  Distribution const& dist = matrix.distribution();
  for (SizeType tile_j = 0; tile_j < dist.localNrTiles().cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < dist.localNrTiles().rows(); ++tile_i) {
      TileType tile = matrix(LocalTileIndex(tile_i, tile_j)).get();
      TileElementSize tile_size = tile.size();
      for (SizeType j = 0; j < tile_size.cols(); ++j) {
        for (SizeType i = 0; i < tile_size.rows(); ++i) {
          local_sum += tile(TileElementIndex(i, j));
        }
      }
    }
  }

  ScalarType global_sum = 0;
  MPI_Datatype mpi_type = dlaf::comm::mpi_datatype<ScalarType>::type;
  MPI_Reduce(&local_sum, &global_sum, 1, mpi_type, MPI_SUM, 0, comm);

  return global_sum;
}

params init_params(variables_map& vm) {
  std::string setup = vm["setup"].as<std::string>();
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
    std::fprintf(stderr, "[ERROR] Number of processes doesn't match the process grid size");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (tile_m > len_m) {
    std::fprintf(stderr, "[ERROR] tile_m > m");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (tile_n > len_n) {
    std::fprintf(stderr, "[ERROR] tile_n > n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (setup != "default" && setup != "mpi_pool" && setup != "priorities") {
    std::fprintf(stderr, "[ERROR] setup must be one of {default, mpi_pool, priorities}");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Check if too many non-blocking communications are being issued.
  if (rank == 0 && batch_size > 1000) {
    std::printf("[WARNING] There are too many tiles batched, this may "
                "result in slowdowns as the number of issued non-blocking "
                "communications at each process is proportianal to the batch size!");
  }

  // Setup
  if (rank == 0) {
    std::printf("len mnk  = %d %d %d\n", len_m, len_n, len_k);
    std::printf("tile mnk = %d %d\n", tile_m, tile_n);
    std::printf("pgrid    = %d %d\n", pgrid_rows, pgrid_cols);
  }

  return params{num_iters,  len_m,      len_n,      len_k, tile_m, tile_n,
                pgrid_rows, pgrid_cols, batch_size, setup, check};
}

options_description init_desc() {
  using hpx::program_options::value;
  using hpx::program_options::bool_switch;
  using std::string;

  options_description desc("Allowed options.");

  // clang-format off
  desc.add_options()
     ("check",      bool_switch()   -> default_value(false)    , "correctness check")
     ("setup",      value<string>() -> default_value("default"), "TSGEMM Executors setup: [default, mpi_pool, priorities]")
     ("num_iters",  value<int>()    -> default_value(   5)     , "number of iterations")
     ("batch_size", value<int>()    -> default_value(  16)     , "number of tiles batched for computation/communication")
     ("len_m",      value<int>()    -> default_value( 100)     , "m dimension")
     ("len_n",      value<int>()    -> default_value( 100)     , "n dimension")
     ("len_k",      value<int>()    -> default_value(1000)     , "k dimension")
     ("tile_m",     value<int>()    -> default_value(  32)     , "tile m dimension")
     ("tile_n",     value<int>()    -> default_value(  32)     , "tile n dimension")
     ("pgrid_rows", value<int>()    -> default_value(   1)     , "process grid rows")
     ("pgrid_cols", value<int>()    -> default_value(   1)     , "process grid columns")
  ;
  // clang-format on

  return desc;
}

}
