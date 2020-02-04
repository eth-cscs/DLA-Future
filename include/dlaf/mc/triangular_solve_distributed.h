//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/util_matrix.h"

/// @file

namespace dlaf {
/// @brief Triangular Solve implementation on distributed memory, solving op(A) X = alpha B (when side ==
/// Left) or X op(A) = alpha B (when side == Right). Algorithm 1: matrix A is communicated
///
/// @param side specifies whether op(A) appears on the \a Left or on the \a Right of matrix X
/// @param uplo specifies whether the matrix A is a \a Lower or \a Upper triangular matrix
/// @param op specifies the form of op(A) to be used in the matrix multiplication: \a NoTrans, \a Trans,
/// \a ConjTrans
/// @param diag specifies if the matrix A is assumed to be unit triangular (\a Unit) or not (\a
/// NonUnit)
/// @tparam A refers to a triangular matrix object
/// @tparam B refers to a matrix object
template <class T>
void triangular_solve_distributed(comm::CommunicatorGrid grid, blas::Side side, blas::Uplo uplo,
                                  blas::Op op, blas::Diag diag, T alpha, Matrix<T, Device::CPU>& mat_a,
                                  Matrix<T, Device::CPU>& mat_b) {
  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor executor_hp =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor executor_normal =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  // Check if matrix A is square
  util_matrix::assertSizeSquare(mat_a, "TriangularSolve", "mat_a");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(mat_a, "TriangularSolve", "mat_a");
  // Check if A and B dimensions are compatible
  util_matrix::assertMultipliableMatrices(mat_a, mat_b, side, op, "TriangularSolve", "mat_a", "mat_b");
  // Check compatibility of the communicator grid and the distribution of matrix A
  util_matrix::assertMatrixDistributedOnGrid(grid, mat_a, "TriangularSolve", "mat_a", "grid");
  // Check compatibility of the communicator grid and the distribution of matrix B
  util_matrix::assertMatrixDistributedOnGrid(grid, mat_b, "TriangularSolve", "mat_b", "grid");

  auto col_comm_size = grid.colCommunicator().size();
  auto row_comm_size = grid.rowCommunicator().size();

  const dlaf::matrix::Distribution& distr_a = mat_a.distribution();
  const dlaf::matrix::Distribution& distr_b = mat_b.distribution();

  SizeType mtile = mat_a.nrTiles().rows();
  SizeType ntile = mat_a.nrTiles().cols();

  auto localnrtile_rows = distr_a.localNrTiles().rows();
  auto localnrtile_cols = distr_a.localNrTiles().cols();

  dlaf::common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  if (uplo == blas::Uplo::Upper) {
    if (side == blas::Side::Left) {
      if (op == blas::Op::NoTrans) {
        // Upper Left NoTrans
      }
      else {
        // Upper Left Trans/ConjTrans case
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Upper Right NoTrans case
      }
      else {
        // Upper Right Trans/ConjTrans case
      }
    }
  }
  else {
    if (side == blas::Side::Left) {
      if (op == blas::Op::NoTrans) {
        // Lower Left NoTrans case

        // Loop on rows of A matrix
        for (SizeType k = 0; k < mtile; ++k) {
          // Create a placeholder that will store the shared futures representing the panel
          std::vector<hpx::shared_future<Tile<const T, Device::CPU>>> panel(
              distr_a.localNrTiles().rows());

          auto k_rank_row = distr_a.rankGlobalTile<Coord::Row>(k);
          auto k_rank_col = distr_a.rankGlobalTile<Coord::Col>(k);

          // Broadcast Akk row-wise
          if (mat_a.rankIndex().row() == k_rank_row) {
            auto k_local_row = distr_a.localTileFromGlobalTile<Coord::Row>(k);

            hpx::shared_future<Tile<const T, Device::CPU>> kk_tile;

            if (mat_a.rankIndex().col() == k_rank_col) {
              auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);

              auto kk = LocalTileIndex{k_local_row, k_local_col};

              if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
                // Row-wise broadcast of Akk tile
                hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                                dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(),
                                                                  tile);
                              }),
                              mat_a.read(kk), serial_comm());
              }

              kk_tile = mat_a.read(kk);
            }
            else {
              // Avoid useless communications if one-column communicator and if on the last column
              if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
                // Receive the Akk tile (row-wise broadcast)
                kk_tile =
                    hpx::dataflow(hpx::util::unwrapping(
                                      [](auto index, auto&& tile_size,
                                         auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                                        memory::MemoryView<T, Device::CPU> mem_view(
                                            util::size_t::mul(tile_size.rows(), tile_size.cols()));
                                        Tile<T, Device::CPU> tile(tile_size, std::move(mem_view),
                                                                  tile_size.rows());
                                        dlaf::comm::sync::broadcast::receive_from(index,
                                                                                  comm_wrapper()
                                                                                      .rowCommunicator(),
                                                                                  tile);
                                        return std::move(tile);
                                      }),
                                  k_rank_row, mat_a.tileSize(GlobalTileIndex(k, k)), serial_comm());
              }
            }

            auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);
            // Loop on column of B matrix
            for (SizeType j_local = distr_b.nextLocalTileFromGlobalTile<Coord::Col>(0);
                 j_local < localnrtile_cols; ++j_local) {
              // Triangular solve of the Bkj tile
              hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo,
                            op, diag, alpha, mat_a.read(LocalTileIndex{k_local_row, k_local_col}),
                            std::move(mat_b(LocalTileIndex{k_local_row, j_local})));

              hpx::shared_future<Tile<const T, Device::CPU>> kj_tile;

              if (mat_b.rankIndex().col() == j_local) {
                // Column-wise broadcast of Bkj
                hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                                dlaf::comm::sync::broadcast::send(comm_wrapper().colCommunicator(),
                                                                  tile);
                              }),
                              mat_b.read(LocalTileIndex{k_local_row, j_local}), serial_comm());
              }
              else {
                if (col_comm_size > 1 && k != (mat_b.nrTiles().cols() - 1)) {
                  auto j = distr_b.globalTileFromLocalTile<Coord::Row>(j_local);

                  // Receive Bkj column-wise
                  kj_tile =
                      hpx::dataflow(hpx::util::unwrapping([](auto index, auto&& tile_size,
                                                             auto&& comm_wrapper)
                                                              -> Tile<const T, Device::CPU> {
                                      memory::MemoryView<T, Device::CPU> mem_view(
                                          util::size_t::mul(tile_size.rows(), tile_size.cols()));
                                      Tile<T, Device::CPU> tile(tile_size, std::move(mem_view),
                                                                tile_size.rows());
                                      dlaf::comm::sync::broadcast::receive_from(index,
                                                                                comm_wrapper()
                                                                                    .colCommunicator(),
                                                                                tile);
                                      return std::move(tile);
                                    }),
                                    k_rank_row, mat_b.tileSize(GlobalTileIndex(k, j)), serial_comm());
                }
              }
            }

            auto nextlocaltilek = distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
            for (SizeType i_local = nextlocaltilek; i_local < localnrtile_rows; ++i_local) {
              auto i = distr_a.globalTileFromLocalTile<Coord::Row>(i_local);

              auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;
              auto i_rank_row = distr_a.rankGlobalTile<Coord::Row>(i);

              // Broadcast Aik row-wise
              if (mat_a.rankIndex().row() == i_rank_row) {
                auto i_local_row = distr_a.localTileFromGlobalTile<Coord::Row>(i);

                hpx::shared_future<Tile<const T, Device::CPU>> ik_tile;

                if (mat_a.rankIndex().col() == k_rank_col) {
                  auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);

                  auto ik = LocalTileIndex{i_local_row, k_local_col};

                  if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
                    // Row-wise broadcast of Aik tile
                    hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                                    dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(),
                                                                      tile);
                                  }),
                                  mat_a.read(ik), serial_comm());
                  }

                  ik_tile = mat_a.read(ik);
                }
                else {
                  // Avoid useless communications if one-column communicator and if on the last column
                  if (col_comm_size > 1 && k != (mat_a.nrTiles().cols() - 1)) {
                    // Receive the Aik tile (row-wise broadcast)
                    kk_tile =
                        hpx::dataflow(hpx::util::unwrapping([](auto index, auto&& tile_size,
                                                               auto&& comm_wrapper)
                                                                -> Tile<const T, Device::CPU> {
                                        memory::MemoryView<T, Device::CPU> mem_view(
                                            util::size_t::mul(tile_size.rows(), tile_size.cols()));
                                        Tile<T, Device::CPU> tile(tile_size, std::move(mem_view),
                                                                  tile_size.rows());
                                        dlaf::comm::sync::broadcast::receive_from(index,
                                                                                  comm_wrapper()
                                                                                      .rowCommunicator(),
                                                                                  tile);
                                        return std::move(tile);
                                      }),
                                      k_rank_row, mat_a.tileSize(GlobalTileIndex(i, k)), serial_comm());
                  }
                }
              }

              for (SizeType j_local = distr_b.nextLocalTileFromGlobalTile<Coord::Col>(0);
                   j_local < localnrtile_cols; ++j_local) {
                auto j = distr_a.globalTileFromLocalTile<Coord::Col>(j_local);
                auto k_local_col = distr_a.localTileFromGlobalTile<Coord::Col>(k);

                auto beta = static_cast<T>(-1.0) / alpha;
                // Matrix multiplication to update other eigenvectors
                hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                              blas::Op::NoTrans, beta, mat_a.read(LocalTileIndex{i_local, k_local_col}),
                              mat_b.read(LocalTileIndex{k_local_row, j_local}), 1.0,
                              std::move(mat_b(LocalTileIndex{i_local, j_local})));
              }
            }
          }
        }
      }
      else {
        // Lower Left Trans/ConjTrans case
      }
    }
    else {
      if (op == blas::Op::NoTrans) {
        // Lower Right NoTrans case
      }
      else {
        // Lower Right Trans/ConjTrans case
      }
    }
  }
}
}
