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

// Forward declations
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
#endif

// Local gemm
//
// - `A` is transposed and has similar layout to `B`.
void schedule_gemm(ConstMatrixType& a_mat, ConstMatrixType& b_mat, MatrixType& cini_mat);

void schedule_comm(ExecutorType const& mpi_executor, CommunicatorGrid& comm_grid,
                   Distribution const& cfin_dist, MatrixType& cini_mat);

void schedule_offload(ConstMatrixType& cini_mat, MatrixType& cfin_mat);

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
  ::Distribution const& cfin_dist = cfin_mat.distribution();

  // Initialize matrices
  init_matrix(a_mat, ::ScalarType(1));
  init_matrix(b_mat, ::ScalarType(1));

  // 1. John's branch
  // 2. MPI pool with a single core
  // 3. Default pool with high priority executor
  // 4. Default pool with default priority executor (the default)
#if defined(DLAF_WITH_MPI_FUTURES)
  // This needs remain in scope for all uses of hpx::mpi
  std::string pool_name = "default";
  hpx::mpi::enable_user_polling enable_polling(pool_name);
  ExecutorType mpi_executor(comm_grid.fullCommunicator());
#else
  using hpx::threads::thread_priority;
#if defined(DLAF_WITH_MPI_POOL)
  std::string pool_name = "mpi";
  auto priority = thread_priority::thread_priority_default;
#elif defined(DLAF_WITH_PRIORITIES)
  std::string pool_name = "default";
  auto priority = thread_priority::thread_priority_high;
#else
  std::string pool_name = "default";
  auto priority = thread_priority::thread_priority_default;
#endif
  // an executor that can be used to place work on the MPI pool if it is enabled
  ExecutorType mpi_executor(pool_name, priority);
#endif

  // 0. Reset buffers
  // 1. Schedule multiply
  // 3. Schedule offloads and receives after multiply
  // 2. Schedule sends and loads
  // 4. Wait for all
  //
  for (int i = 0; i < ps.num_iters; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = clock_t::now();

    schedule_gemm(a_mat, b_mat, cini_mat);
    schedule_comm(mpi_executor, comm_grid, cfin_dist, cini_mat);
    schedule_offload(cini_mat, cfin_mat);
    waitall_tiles(cfin_mat);

    MPI_Barrier(MPI_COMM_WORLD);
    auto t_end = clock_t::now();

    if (rank == 0) {
      std::printf("%d: t_tot  [s] = %.5f\n", i, seconds_t(t_end - t_start).count());
    }

    //    // Simple check
    //    ScalarType cfin_sum = sum_matrix(comm_world, cfin_mat);
    //    if (rank == 0) {
    //      std::cout << cfin_sum << '\n';
    //    }
  }

  // Upon exit the mpi/user polling RAII object will stop polling
  return hpx::finalize();
}

// Example usage:
//
//   mpirun -np 1 tsgemm --len_m      100  --len_n      100  --len_k  10000
//                       --tile_m      64  --tile_n
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

  // declare options before creating resource partitioner
  ::options_description desc_cmdline = ::init_desc();

#ifdef DLAF_WITH_MPI_POOL
  // NB.
  // thread pools must be declared before starting the runtime

  // Create resource partitioner
  hpx::resource::partitioner rp(desc_cmdline, argc, argv);

  // create a thread pool that is not "default" that we will use for MPI work
  rp.create_thread_pool("mpi");

  // add (enabled) PUs on the first core to it
  rp.add_resource(rp.numa_domains()[0].cores()[0].pus(), "mpi");
#endif

  // Start the HPX runtime
  hpx::init(desc_cmdline, argc, argv);

  MPI_Finalize();
}

namespace {

void schedule_gemm(ConstMatrixType& a_mat, ConstMatrixType& b_mat, MatrixType& cini_mat) {
  // TODO: Order communication and computation to limit the number of simultaneous non-blocking MPI calls

  dlaf::LocalTileSize const& tile_size = cini_mat.distribution().localNrTiles();
  for (SizeType tile_j = 0; tile_j < tile_size.cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < tile_size.rows(); ++tile_i) {
      LocalTileIndex a_mat_idx(0, tile_i);
      LocalTileIndex b_mat_idx(0, tile_j);
      LocalTileIndex c_mat_idx(tile_i, tile_j);

      auto gemm_f = hpx::util::unwrapping([](auto&& a_tile, auto&& b_tile, auto&& c_tile) {
        dlaf::tile::gemm(blas::Op::Trans, blas::Op::NoTrans, ScalarType(1), a_tile, b_tile,
                         ScalarType(0), c_tile);
      });
      hpx::dataflow(gemm_f, a_mat.read(a_mat_idx), b_mat.read(b_mat_idx), cini_mat(c_mat_idx));
    }
  }
}

void schedule_comm(ExecutorType const& mpi_executor, CommunicatorGrid& comm_grid,
                   Distribution const& cfin_dist, MatrixType& cini_mat) {
  MPI_Datatype mpi_type = dlaf::comm::mpi_datatype<ScalarType>::type;
  MPI_Op mpi_op = MPI_SUM;
  MPI_Comm mpi_comm(comm_grid.fullCommunicator());

  CommIndex this_rank_coords = cfin_dist.rankIndex();
  dlaf::LocalTileSize const& tile_size = cini_mat.distribution().localNrTiles();
  for (SizeType tile_j = 0; tile_j < tile_size.cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < tile_size.rows(); ++tile_i) {
      LocalTileIndex c_mat_idx = LocalTileIndex(tile_i, tile_j);

      CommIndex rank_coords = cfin_dist.rankGlobalTile(GlobalTileIndex(tile_i, tile_j));
      dlaf::comm::IndexT_MPI root_rank = comm_grid.rankFullCommunicator(rank_coords);

      auto mpi_f = hpx::util::unwrapping([=](auto&& c_tile) {
        TileElementSize tile_size = c_tile.size();
        SizeType num_elements = tile_size.rows() * tile_size.cols();

        // Use the same send and recv buffers for `root` process
        void* sendbuf = c_tile.ptr(TileElementIndex(0, 0));
        void* recvbuf = sendbuf;  // only relevant at `root`
        if (rank_coords == this_rank_coords) {
          sendbuf = MPI_IN_PLACE;
        }

        // TODO: collective operations on the same communicator have to be ordered on all processes.

#ifdef DLAF_WITH_MPI_FUTURES
        return hpx::async(mpi_executor, MPI_Ireduce, sendbuf, recvbuf, num_elements, mpi_type, mpi_op,
                          root_rank, mpi_comm);

#else
        MPI_Request req;
        MPI_Ireduce(sendbuf, recvbuf, num_elements, mpi_type, mpi_op, root_rank, mpi_comm, &req);
        hpx::util::yield_while([&req] {
          int flag;
          MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
          return flag == 0;
        });
#endif
      });

      hpx::dataflow(mpi_executor, mpi_f, cini_mat(c_mat_idx));
    }
  }
}

void schedule_offload(ConstMatrixType& cini_mat, MatrixType& cfin_mat) {
  Distribution const& cfin_dist = cfin_mat.distribution();
  CommIndex this_rank_coords = cfin_dist.rankIndex();
  dlaf::LocalTileSize const& tile_size = cini_mat.distribution().localNrTiles();
  for (SizeType tile_j = 0; tile_j < tile_size.cols(); ++tile_j) {
    for (SizeType tile_i = 0; tile_i < tile_size.rows(); ++tile_i) {
      GlobalTileIndex tile_idx(tile_i, tile_j);  // for cfin
      CommIndex tile_rank_coords = cfin_dist.rankGlobalTile(tile_idx);

      // If tile belongs to this process
      if (this_rank_coords == tile_rank_coords) {
        auto offload_f = hpx::util::unwrapping([](auto&& cini_tile, auto&& cfin_tile) {
          TileElementSize tile_size = cini_tile.size();
          for (SizeType j = 0; j < tile_size.cols(); ++j) {
            for (SizeType i = 0; i < tile_size.rows(); ++i) {
              TileElementIndex idx(i, j);
              cfin_tile(idx) = cini_tile(idx);
            }
          }
        });
        hpx::dataflow(offload_f, cini_mat.read(tile_idx), cfin_mat(tile_idx));
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
  using dlaf::util::ceilDiv;

  int num_iters = vm["num_iters"].as<int>();
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

  // TODO: order the communications by sending `num_comm_cols` columns at once
  // TODO: schedule the gemms associated with the columns first
  // Check if too many non-blocking communications are being issued.
  //
  constexpr int max_comms = 1000;
  int num_tiles = ceilDiv(len_m, tile_m) * ceilDiv(len_n, tile_n);
  if (rank == 0 && num_tiles > max_comms) {
    std::printf("[WARNING] There are too many pieces! Increase the tile size!");
  }

  // Setup
  if (rank == 0) {
    std::printf("len mnk  = %d %d %d\n", len_m, len_n, len_k);
    std::printf("tile mnk = %d %d\n", tile_m, tile_n);
    std::printf("pgrid    = %d %d\n", pgrid_rows, pgrid_cols);
  }

  return params{num_iters, len_m, len_n, len_k, tile_m, tile_n, pgrid_rows, pgrid_cols};
}

options_description init_desc() {
  using hpx::program_options::value;

  options_description desc("Allowed options.");

  // clang-format off
  desc.add_options()
     ("num_iters",  value<int>()->default_value(   5), "number of iterations")
     ("len_m",      value<int>()->default_value( 100), "m dimension")
     ("len_n",      value<int>()->default_value( 100), "n dimension")
     ("len_k",      value<int>()->default_value(1000), "k dimension")
     ("tile_m",     value<int>()->default_value(  32), "tile m dimension")
     ("tile_n",     value<int>()->default_value(  32), "tile n dimension")
     ("pgrid_rows", value<int>()->default_value(   1), "process grid rows")
     ("pgrid_cols", value<int>()->default_value(   1), "process grid columns")
  ;
  // clang-format on

  return desc;
}

}
