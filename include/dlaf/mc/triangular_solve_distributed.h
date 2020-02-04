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
                                  blas::Op op, blas::Diag diag, T alpha, Matrix<T, Device::CPU>& A,
                                  Matrix<T, Device::CPU>& B) {
  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor executor_hp =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor executor_normal =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  // Check if matrix A is square
  util_matrix::assertSizeSquare(A, "TriangularSolve", "A");
  // Check if block matrix A is square
  util_matrix::assertBlocksizeSquare(A, "TriangularSolve", "A");
  // Check if A and B dimensions are compatible
  util_matrix::assertMultipliableMatrices(A, B, side, op, "TriangularSolve", "A", "B");
  // Check compatibility of the communicator grid and the distribution of matrix A
  util_matrix::assertMatrixDistributedOnGrid(grid, A, "TriangularSolve", "A", "grid");
  // Check compatibility of the communicator grid and the distribution of matrix B
  util_matrix::assertMatrixDistributedOnGrid(grid, B, "TriangularSolve", "B", "grid");

  auto col_comm_size = grid.colCommunicator().size();
  auto row_comm_size = grid.rowCommunicator().size();

  const dlaf::matrix::Distribution& distrA = A.distribution();
  const dlaf::matrix::Distribution& distrB = B.distribution();

  SizeType mtile = A.nrTiles().rows();
  SizeType ntile = A.nrTiles().cols();

  auto localnrtile_rows = distrA.localNrTiles().rows();
  auto localnrtile_cols = distrA.localNrTiles().cols();

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
              distrA.localNrTiles().rows());

          auto k_rank_row = distrA.rankGlobalTile<Coord::Row>(k);
          auto k_rank_col = distrA.rankGlobalTile<Coord::Col>(k);

          // Broadcast Akk row-wise
          if (A.rankIndex().row() == k_rank_row) {
            auto k_local_row = distrA.localTileFromGlobalTile<Coord::Row>(k);

            hpx::shared_future<Tile<const T, Device::CPU>> kk_tile;

            if (A.rankIndex().col() == k_rank_col) {
              auto k_local_col = distrA.localTileFromGlobalTile<Coord::Col>(k);

              auto kk = LocalTileIndex{k_local_row, k_local_col};

              if (col_comm_size > 1 && k != (A.nrTiles().cols() - 1)) {
                // Row-wise broadcast of Akk tile
                hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                                dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(),
                                                                  tile);
                              }),
                              A.read(kk), serial_comm());
              }

              kk_tile = A.read(kk);
            }
            else {
              // Avoid useless communications if one-column communicator and if on the last column
              if (col_comm_size > 1 && k != (A.nrTiles().cols() - 1)) {
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
                                  k_rank_row, A.tileSize(GlobalTileIndex(k, k)), serial_comm());
              }
            }

            auto k_local_col = distrA.localTileFromGlobalTile<Coord::Col>(k);
            // Loop on column of B matrix
            for (SizeType j_local = distrB.nextLocalTileFromGlobalTile<Coord::Col>(0);
                 j_local < localnrtile_cols; ++j_local) {
              // Triangular solve of the Bkj tile
              hpx::dataflow(executor_hp, hpx::util::unwrapping(tile::trsm<T, Device::CPU>), side, uplo,
                            op, diag, alpha, A.read(LocalTileIndex{k_local_row, k_local_col}),
                            std::move(B(LocalTileIndex{k_local_row, j_local})));

              hpx::shared_future<Tile<const T, Device::CPU>> kj_tile;

              if (B.rankIndex().col() == j_local) {
                // Column-wise broadcast of Bkj
                hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                                dlaf::comm::sync::broadcast::send(comm_wrapper().colCommunicator(),
                                                                  tile);
                              }),
                              B.read(LocalTileIndex{k_local_row, j_local}), serial_comm());
              }
              else {
                if (col_comm_size > 1 && k != (B.nrTiles().cols() - 1)) {
                  auto j = distrB.globalTileFromLocalTile<Coord::Row>(j_local);

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
                                    k_rank_row, B.tileSize(GlobalTileIndex(k, j)), serial_comm());
                }
              }
            }

            auto nextlocaltilek = distrA.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
            for (SizeType i_local = nextlocaltilek; i_local < localnrtile_rows; ++i_local) {
              auto i = distrA.globalTileFromLocalTile<Coord::Row>(i_local);

              auto trailing_executor = (i == k + 1) ? executor_hp : executor_normal;
              auto i_rank_row = distrA.rankGlobalTile<Coord::Row>(i);

              // Broadcast Aik row-wise
              if (A.rankIndex().row() == i_rank_row) {
                auto i_local_row = distrA.localTileFromGlobalTile<Coord::Row>(i);

                hpx::shared_future<Tile<const T, Device::CPU>> ik_tile;

                if (A.rankIndex().col() == k_rank_col) {
                  auto k_local_col = distrA.localTileFromGlobalTile<Coord::Col>(k);

                  auto ik = LocalTileIndex{i_local_row, k_local_col};

                  if (col_comm_size > 1 && k != (A.nrTiles().cols() - 1)) {
                    // Row-wise broadcast of Aik tile
                    hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                                    dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(),
                                                                      tile);
                                  }),
                                  A.read(ik), serial_comm());
                  }

                  ik_tile = A.read(ik);
                }
                else {
                  // Avoid useless communications if one-column communicator and if on the last column
                  if (col_comm_size > 1 && k != (A.nrTiles().cols() - 1)) {
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
                                      k_rank_row, A.tileSize(GlobalTileIndex(i, k)), serial_comm());
                  }
                }
              }

              for (SizeType j_local = distrB.nextLocalTileFromGlobalTile<Coord::Col>(0);
                   j_local < localnrtile_cols; ++j_local) {
                auto j = distrA.globalTileFromLocalTile<Coord::Col>(j_local);
                auto k_local_col = distrA.localTileFromGlobalTile<Coord::Col>(k);

                auto beta = static_cast<T>(-1.0) / alpha;
                // Matrix multiplication to update other eigenvectors
                hpx::dataflow(trailing_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>), op,
                              blas::Op::NoTrans, beta, A.read(LocalTileIndex{i_local, k_local_col}),
                              B.read(LocalTileIndex{k_local_row, j_local}), 1.0,
                              std::move(B(LocalTileIndex{i_local, j_local})));
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
