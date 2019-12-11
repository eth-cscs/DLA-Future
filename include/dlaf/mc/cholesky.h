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

static bool use_pools = true;

/// @brief Cholesky implementation on local memory
///
/// @param uplo specifies that the matrix is \a Lower triangular
/// @tparam mat refers to a dlaf::Matrix object
///
/// @throws std::runtime_error if \p uplo = \a Upper decomposition is chosen (not yet implemented)

template <class T>
void cholesky(blas::Uplo uplo, Matrix<T, Device::CPU>& mat) {
  // Check that matrix is on local memory
  if (mat.distribution().commGridSize() == comm::Size2D{1, 1}) {
    // Set up executor on the default queue with high priority.
    hpx::threads::scheduled_executor matrix_HP_executor =
        hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);

    // Set up executor on the default queue with default priority.
    hpx::threads::scheduled_executor matrix_normal_executor =
        hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

    // Check if matrix is square
    util_matrix::assert_size_square(mat, "Cholesky", "mat");
    // Check if block matrix is square
    util_matrix::assert_blocksize_square(mat, "Cholesky", "mat");

    // Number of tile (rows = cols)
    SizeType nrtile = mat.nrTiles().cols();

    if (uplo == blas::Uplo::Lower) {
      for (SizeType k = 0; k < nrtile; ++k) {
        // Cholesky decomposition on mat(k,k) r/w potrf (lapack operation)
        auto kk = LocalTileIndex{k, k};

        hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), uplo,
                      std::move(mat(kk)));

        for (SizeType i = k + 1; i < nrtile; ++i) {
          // Update panel mat(i,k) with trsm (blas operation), using data mat.read(k,k)
          hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::trsm<T, Device::CPU>),
                        blas::Side::Right, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0,
                        mat.read(kk), std::move(mat(LocalTileIndex{i, k})));
        }

        for (SizeType j = k + 1; j < nrtile; ++j) {
          // Choose queue priority
          auto trailing_matrix_executor = (j == k + 1) ? matrix_HP_executor : matrix_normal_executor;

          // Update trailing matrix: diagonal element mat(j,j, reading mat.read(j,k), using herk (blas operation)
          hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>),
                        uplo, blas::Op::NoTrans, -1.0, mat.read(LocalTileIndex{j, k}), 1.0,
                        std::move(mat(LocalTileIndex{j, j})));

          for (SizeType i = j + 1; i < nrtile; ++i) {
            // Update remaining trailing matrix mat(i,j), reading mat.read(i,k) and mat.read(j,k), using
            // gemm (blas operation)
            hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                          blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, mat.read(LocalTileIndex{i, k}),
                          mat.read(LocalTileIndex{j, k}), 1.0, std::move(mat(LocalTileIndex{i, j})));
          }
        }
      }
    }
    else {
      throw std::runtime_error("uplo = Upper not yet implemented");
    }
  }
  else {
    throw std::runtime_error("Matrix is not on local memory");
  }
}

/// @brief Cholesky implementation on distributed memory
///
/// @parame grid refers to a comm::CommunicatorGrid object
/// @param uplo specifies that the matrix is \a Lower triangular
/// @tparam mat refers to a dlaf::Matrix object
///
/// @throws std::runtime_error if \p uplo = \a Upper decomposition is chosen (not yet implemented)

template <class T>
void cholesky(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, Device::CPU>& mat) {
  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor matrix_HP_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);
  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor matrix_normal_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);

  auto col_comm_size = grid.colCommunicator().size();
  auto row_comm_size = grid.rowCommunicator().size();

  // Set up an executor on the mpi pool --> This part need to be added
  //  hpx::threads::scheduled_executor mpi_executor;
  //  if (use_pools && m_size > 1) {
  //    hpx::threads::executors::pool_executor mpi_exec("mpi");
  //    mpi_executor = mpi_exec;
  //   }
  //  else {
  //    mpi_executor = matrix_HP_executor;
  //  }

  // Check if matrix is square
  util_matrix::assert_size_square(mat, "Cholesky", "mat");
  // Check if block matrix is square
  util_matrix::assert_blocksize_square(mat, "Cholesky", "mat");

  const dlaf::matrix::Distribution& distr = mat.distribution();

  SizeType nrtile = mat.nrTiles().cols();

  auto localnrtile_rows = distr.localNrTiles().rows();
  auto localnrtile_cols = distr.localNrTiles().cols();

  dlaf::common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  // Method only for Lower triangular matrix
  if (uplo == blas::Uplo::Lower) {
    for (SizeType k = 0; k < nrtile; ++k) {
      // Create the panel as a vector of shared future
      std::vector<hpx::shared_future<Tile<const T, Device::CPU>>> panel(distr.localNrTiles().rows());

      auto k_rank_row = distr.rankGlobalTile<Coord::Row>(k);
      auto k_rank_col = distr.rankGlobalTile<Coord::Col>(k);

      if (mat.rankIndex().col() == k_rank_col) {
        auto k_local_col = distr.localTileFromGlobalTile<Coord::Col>(k);

        hpx::shared_future<Tile<const T, Device::CPU>> kk_tile;

        if (mat.rankIndex().row() == k_rank_row) {
          // If the diagonal tile is on this node factorize it
          auto k_local_row = distr.localTileFromGlobalTile<Coord::Row>(k);

          auto kk = LocalTileIndex{k_local_row, k_local_col};

          // Cholesky decomposition on mat(k,k) r/w potrf (lapack operation)
          hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), uplo,
                        std::move(mat(kk)));

          // Avoid useless communication if one-column communicator and if on the last column
          if (col_comm_size > 1 && k != (mat.size().cols() - 1)) {
            // Broadcast the panel column-wise
            hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                            dlaf::comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
                          }),
                          mat.read(kk), serial_comm());
          }

          kk_tile = mat.read(kk);
        }
        else {
          // Avoid useless communications if one-column communicator and if on the last column
          if (col_comm_size > 1 && k != (mat.size().cols() - 1)) {
            // Update the panel column-wise
            kk_tile = hpx::dataflow(  //
                hpx::util::unwrapping(
                    [](auto index, auto&& tile_size, auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                      memory::MemoryView<T, Device::CPU> mem_view(
                          util::size_t::mul(tile_size.rows(), tile_size.cols()));
                      Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
                      dlaf::comm::sync::broadcast::receive_from(index, comm_wrapper().colCommunicator(),
                                                                tile);
                      return std::move(tile);
                    }),
                k_rank_row, mat.tileSize(GlobalTileIndex(k, k)), serial_comm());
          }
        }

        for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
             i_local < localnrtile_rows; ++i_local) {
          // Update panel mat(i,k) with trsm (blas operation), using data mat.read(k,k)
          hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::trsm<T, Device::CPU>),
                        blas::Side::Right, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0, kk_tile,
                        std::move(mat(LocalTileIndex{i_local, k_local_col})));

          // Avoid useless communications if one-row communicator grid
          if (row_comm_size > 1) {
            // Broadcast the panel row-wise
            hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                            dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
                          }),
                          mat.read(LocalTileIndex{i_local, k_local_col}), serial_comm());
          }

          panel[i_local] = mat.read(LocalTileIndex{i_local, k_local_col});
        }
      }
      else {
        for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
             i_local < localnrtile_rows; ++i_local) {
          auto i = distr.globalTileFromLocalTile<Coord::Row>(i_local);

          // Avoid useless communications if one-row communicator grid
          if (row_comm_size > 1) {
            // Update the panel row-wise
            panel[i_local] = hpx::dataflow(  //
                hpx::util::unwrapping(
                    [](auto index, auto&& tile_size, auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                      memory::MemoryView<T, Device::CPU> mem_view(
                          util::size_t::mul(tile_size.rows(), tile_size.cols()));
                      Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
                      dlaf::comm::sync::broadcast::receive_from(index, comm_wrapper().rowCommunicator(),
                                                                tile);
                      return std::move(tile);
                    }),
                k_rank_col, mat.tileSize(GlobalTileIndex(i, k)), serial_comm());
          }
        }
      }

      for (SizeType j_local = distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1);
           j_local < localnrtile_cols; ++j_local) {
        // Choose "high priority" for first tile of the trailing matrix
        auto trailing_matrix_executor = (j_local == k + 1) ? matrix_HP_executor : matrix_normal_executor;

        hpx::shared_future<Tile<const T, Device::CPU>> col_panel;

        auto j = distr.globalTileFromLocalTile<Coord::Col>(j_local);

        auto j_rank_row = distr.rankGlobalTile<Coord::Row>(j);

        if (mat.rankIndex().row() == j_rank_row) {
          // Check if the diagonal tile of the trailing matrix is on this node
          auto i_local = distr.localTileFromGlobalTile<Coord::Row>(j);

          // Avoid useless communications if one-row communicator grid and if on the last panel
          if (col_comm_size > 1 && j != (mat.size().cols() - 1)) {
            // Broadcast the (trailing) panel column-wise
            hpx::dataflow(hpx::util::unwrapping([k, j](auto&& tile, auto&& comm_wrapper) {
                            //                                              std::cout << "Send" << k << j
                            //                                              << tile.size() << std::endl;
                            dlaf::comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
                          }),
                          panel[i_local], serial_comm());
          }

          // Compute first tile of the column of the trailing matrix: diagonal element mat(j,j), reading
          // mat.read(j,k), using herk (blas operation)
          hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>),
                        uplo, blas::Op::NoTrans, -1.0, panel[i_local], 1.0,
                        mat(LocalTileIndex{i_local, j_local}));

          col_panel = panel[i_local];
        }
        else {
          // Avoid useless communications if one-row communicator grid and if on the last panel
          if (col_comm_size > 1 && j != (mat.size().cols() - 1)) {
            // Update the (trailing) panel column-wise
            col_panel = hpx::dataflow(  //
                hpx::util::unwrapping([k, j](auto index, auto&& tile_size,
                                             auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                  memory::MemoryView<T, Device::CPU> mem_view(
                      util::size_t::mul(tile_size.rows(), tile_size.cols()));
                  Tile<T, Device::CPU> tile(tile_size, std::move(mem_view), tile_size.rows());
                  //                            std::cout << "Recv" << index << k << j
                  //                            << tile.size() << std::endl;
                  dlaf::comm::sync::broadcast::receive_from(index, comm_wrapper().colCommunicator(),
                                                            tile);
                  return std::move(tile);
                }),
                j_rank_row, mat.tileSize(GlobalTileIndex(j, k)), serial_comm());
          }
        }

        for (SizeType i_local = distr.nextLocalTileFromGlobalTile<Coord::Row>(j + 1);
             i_local < localnrtile_rows; ++i_local) {
          // Update remaining trailing matrix mat(i,j), reading mat.read(i,k) and mat.read(j,k), using
          // gemm (blas operation)
          hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
                        blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, panel[i_local], col_panel, 1.0,
                        std::move(mat(LocalTileIndex{i_local, j_local})));
        }
      }
    }
  }
  else {
    throw std::runtime_error("uplo = Upper not yet implemented");
  }
}
}
