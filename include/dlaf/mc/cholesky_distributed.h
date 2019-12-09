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
//#include "dlaf/matrix/distribution.h"
//#include "dlaf/matrix/util_distribution.h"
#include "dlaf/util_matrix.h"

/// @file

namespace dlaf {

/// @brief Cholesky implementation on distributed memory
///
/// @param uplo specifies that the matrix is \a Lower triangular
/// @tparam mat refers to a dlaf::Matrix object
///
/// @throws std::runtime_error if \p uplo = \a Upper decomposition is chosen (not yet implemented)

static bool use_pools = true;

template <class T>
void cholesky_distributed(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<T, Device::CPU>& mat) {
  // Set up executor on the default queue with high priority.
  hpx::threads::scheduled_executor matrix_HP_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_high);
  // Set up executor on the default queue with default priority.
  hpx::threads::scheduled_executor matrix_normal_executor =
      hpx::threads::executors::pool_executor("default", hpx::threads::thread_priority_default);
  // Set up an executor on the mpi pool --> This part need to be added
  //  hpx::threads::scheduled_executor mpi_executor;
  //  int m_size = grid.size().cols();
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

  // Number of tile (rows = cols)
  SizeType nrtile = mat.nrTiles().cols();

  // Number of local tiles (rows)
  auto localnrtile_rows = mat.distribution().localNrTiles().rows();
  // Number of local tiles (cols)
  auto localnrtile_cols = mat.distribution().localNrTiles().cols();

  // Pipeline for CommunicatorGrid
  dlaf::common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  // Method only for Lower triangular matrix
  if (uplo == blas::Uplo::Lower) {
    for (SizeType k = 0; k < nrtile; ++k) {
      // Create the panel as a vector of shared future
      std::vector<hpx::shared_future<Tile<const T, Device::CPU>>> panel(
          mat.distribution().localNrTiles().rows());

      // Index of rank owning the tile
      auto k_rank_row = mat.distribution().template rankGlobalTile<Coord::Row>(k);
      auto k_rank_col = mat.distribution().template rankGlobalTile<Coord::Col>(k);

      // If the diagonal tile is on this node factorize it
      if (mat.rankIndex().col() == k_rank_col) {
        auto k_local_col = mat.distribution().template localTileFromGlobalTile<Coord::Col>(k);

        hpx::shared_future<Tile<const T, Device::CPU>> kk_tile;

        if (mat.rankIndex().row() == k_rank_row) {
          auto k_local_row = mat.distribution().template localTileFromGlobalTile<Coord::Row>(k);

          // Select tile kk
          auto kk = LocalTileIndex{k_local_row, k_local_col};

          // Cholesky decomposition on mat(k,k) r/w potrf (lapack operation)
          hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::potrf<T, Device::CPU>), uplo,
                        std::move(mat(kk)));

          // Broadcast panel
          hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                          dlaf::comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
                        }),
                        mat.read(kk), serial_comm());

          kk_tile = mat.read(kk);
        }
        else {
          // Update the panel column-wise
          kk_tile =
              hpx::dataflow(hpx::util::unwrapping([](auto index, auto&& tile_size,
                                                     auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                              memory::MemoryView<T, Device::CPU> mem_view(
                                  util::size_t::mul(tile_size.rows(), tile_size.cols()));
                              Tile<T, Device::CPU> tile(tile_size, std::move(mem_view),
                                                        tile_size.rows());
                              dlaf::comm::sync::broadcast::receive_from(index,
                                                                        comm_wrapper().colCommunicator(),
                                                                        tile);
                              return std::move(tile);
                            }),
                            k_rank_row, TileElementSize(mat.blockSize().cols(), mat.blockSize().cols()),
                            serial_comm());
          // will be
          //}), k_rank_row, mat.tileSize(GlobalElementSize(k, k)), serial_comm() );
        }

        for (SizeType i_local =
                 mat.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
             i_local < localnrtile_rows; ++i_local) {
          // Update panel mat(i,k) with trsm (blas operation), using data mat.read(k,k)
          auto k_local_row = mat.distribution().template localTileFromGlobalTile<Coord::Row>(k);

          //	  hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::trsm<T, Device::CPU>),
          //			blas::Side::Right, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0,
          //			panel[k_local_row], std::move(mat(LocalTileIndex{i_local, k_local_col})));

          hpx::dataflow(matrix_HP_executor, hpx::util::unwrapping(tile::trsm<T, Device::CPU>),
                        blas::Side::Right, uplo, blas::Op::ConjTrans, blas::Diag::NonUnit, 1.0, kk_tile,
                        std::move(mat(LocalTileIndex{i_local, k_local_col})));

          // Broadcast the panel row-wise
          hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                          dlaf::comm::sync::broadcast::send(comm_wrapper().rowCommunicator(), tile);
                        }),
                        mat.read(LocalTileIndex{i_local, k_local_col}), serial_comm());

          panel[i_local] = mat.read(LocalTileIndex{i_local, k_local_col});
        }
      }
      else {
        for (SizeType i_local =
                 mat.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
             i_local < localnrtile_rows; ++i_local) {
          auto k_local_row = mat.distribution().template localTileFromGlobalTile<Coord::Row>(k);

          TileElementSize tile_size(mat.blockSize().cols(), mat.blockSize().cols());
          // TileElementSize tile_size = mat.tileSize({i,k});

          panel[i_local] =
              hpx::dataflow(hpx::util::unwrapping([](auto&& index, auto&& tile_size,
                                                     auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                              memory::MemoryView<T, Device::CPU> mem_view(
                                  util::size_t::mul(tile_size.rows(), tile_size.cols()));
                              Tile<T, Device::CPU> tile(tile_size, std::move(mem_view),
                                                        tile_size.rows());
                              dlaf::comm::sync::broadcast::receive_from(index,
                                                                        comm_wrapper().colCommunicator(),
                                                                        tile);
                              return std::move(tile);
                            }),
                            k_rank_col, tile_size, serial_comm());
        }
      }

      for (SizeType j_local = mat.distribution().template nextLocalTileFromGlobalTile<Coord::Col>(k + 1);
           j_local < localnrtile_cols; ++j_local) {
        // Choose queue priority
        auto trailing_matrix_executor = (j_local == k + 1) ? matrix_HP_executor : matrix_normal_executor;

        hpx::shared_future<Tile<const T, Device::CPU>> col_panel;

        auto j = mat.distribution().template globalTileFromLocalTile<Coord::Col>(j_local);

        auto j_rank_row = mat.distribution().template rankGlobalTile<Coord::Row>(j);

        if (mat.rankIndex().row() == j_rank_row) {
          auto i_local = mat.distribution().template localTileFromGlobalTile<Coord::Row>(j);

          hpx::dataflow(hpx::util::unwrapping([](auto&& tile, auto&& comm_wrapper) {
                          dlaf::comm::sync::broadcast::send(comm_wrapper().colCommunicator(), tile);
                        }),
                        panel[i_local], serial_comm());

          // Update trailing matrix: diagonal element mat(j,j), reading mat.read(j,k), using herk (blas operation)
          //  hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::herk<T, Device::CPU>), uplo,
          //		blas::Op::NoTrans, -1.0, mat.read(LocalTileIndex{i_local, j_local}), 1.0,
          //		panel[i_local]);

          col_panel = panel[i_local];
        }
        else {
          col_panel =
              hpx::dataflow(hpx::util::unwrapping([](auto index, auto&& tile_size,
                                                     auto&& comm_wrapper) -> Tile<const T, Device::CPU> {
                              memory::MemoryView<T, Device::CPU> mem_view(
                                  util::size_t::mul(tile_size.rows(), tile_size.cols()));
                              Tile<T, Device::CPU> tile(tile_size, std::move(mem_view),
                                                        tile_size.rows());
                              dlaf::comm::sync::broadcast::receive_from(index,
                                                                        comm_wrapper().colCommunicator(),
                                                                        tile);
                              return std::move(tile);
                            }),
                            k_rank_row, TileElementSize(mat.blockSize().cols(), mat.blockSize().cols()),
                            serial_comm());
          // will be
          //}), k_rank_row, mat.tileSize(GlobalElementSize(k, k)), serial_comm() );
        }
      }

      //      for (SizeType j = k + 1; j < nrtile; ++j) {
      //
      //        for (SizeType i = j + 1; i < nrtile; ++i) {
      //          // Update remaining trailing matrix mat(i,j), reading mat.read(i,k) and mat.read(j,k), using
      //          // gemm (blas operation)
      //          hpx::dataflow(trailing_matrix_executor, hpx::util::unwrapping(tile::gemm<T, Device::CPU>),
      //                        blas::Op::NoTrans, blas::Op::ConjTrans, -1.0, mat.read(LocalTileIndex{i, k}),
      //                        mat.read(LocalTileIndex{j, k}), 1.0, std::move(mat(LocalTileIndex{i, j})));
      //        }
      //      }
    }
  }
  else {
    throw std::runtime_error("uplo = Upper not yet implemented");
  }
}
}
