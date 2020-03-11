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

#include <algorithm>

#include <hpx/hpx.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

using namespace dlaf;

/// Set to zero the upper part of the diagonal tiles
///
/// For the tiles on the diagonal (i.e. row == col), the elements in the upper triangular
/// part of each tile, diagonal excluded, are set to zero.
/// Tiles that are not on the diagonal (i.e. row != col) will not be touched or referenced
template <class T>
void setUpperToZeroForDiagonalTiles(Matrix<T, Device::CPU>& matrix) {
  util_matrix::assertBlocksizeSquare(matrix, "setUpperToZeroForDiagonalTiles", "matrix");

  auto& distribution = matrix.distribution();

  for (int j = 0; j < distribution.localNrTiles().cols(); ++j) {
    for (int i = 0; i < distribution.localNrTiles().rows(); ++i) {
      const auto tile_wrt_global = distribution.globalTileIndex(LocalTileIndex{i, j});
      const auto tl_index = distribution.globalElementIndex(tile_wrt_global, {0, 0});

      if (tile_wrt_global.row() != tile_wrt_global.col())
        continue;

      matrix(tile_wrt_global).then(hpx::util::unwrapping([tl_index](auto&& tile) {
        for (int j = 0; j < tile.size().cols(); ++j)
          for (int i = 0; i < tile.size().rows(); ++i) {
            GlobalElementIndex element_wrt_global{tl_index.row() + i, tl_index.col() + j};
            if (element_wrt_global.row() < element_wrt_global.col())
              tile(TileElementIndex{i, j}) = 0;
          }
      }));
    }
  }
}

template <class T>
T matrix_norm(Matrix<const T, Device::CPU>& matrix, comm::CommunicatorGrid comm_grid) {
  using common::internal::vector;

  const auto& distribution = matrix.distribution();

  vector<hpx::future<T>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each tile in local, create a task that finds the max element in the tile
  for (SizeType j_loc = 0; j_loc < distribution.localNrTiles().cols(); ++j_loc) {
    const SizeType j = distribution.template globalTileFromLocalTile<Coord::Col>(j_loc);

    SizeType i_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(j);
    for (; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
      auto current_tile_max =
          hpx::dataflow(hpx::util::unwrapping([](auto&& tile) -> T {
                          T tile_max_value = std::abs(T{0});
                          for (SizeType j = 0; j < tile.size().cols(); ++j)
                            for (SizeType i = j; i < tile.size().rows(); ++i)
                              // TODO should we consider just elements in the lower triangular?
                              tile_max_value = std::max(tile_max_value, tile({i, j}));
                          return tile_max_value;
                        }),
                        matrix.read(LocalTileIndex{i_loc, j_loc}));

      tiles_max.emplace_back(std::move(current_tile_max));
    }
  }

  // than it is necessary to reduce max values from all ranks into a single max value for the matrix

  // TODO unwrapping can be skipped for optimization reasons
  auto local_max_value = hpx::dataflow(hpx::util::unwrapping([](const auto&& values) {
                                         // some rank may not have any element
                                         if (values.size() == 0)
                                           return std::numeric_limits<T>::min();
                                         return *std::max_element(values.begin(), values.end());
                                       }),
                                       tiles_max)
                             .get();

  T max_value_rows;
  if (comm_grid.size().cols() > 1)
    comm::sync::reduce(0, comm_grid.rowCommunicator(), MPI_MAX, common::make_buffer(&local_max_value, 1),
                       common::make_buffer(&max_value_rows, 1));
  else
    max_value_rows = local_max_value;

  T max_value;
  if (comm_grid.size().rows() > 1)
    comm::sync::reduce(0, comm_grid.colCommunicator(), MPI_MAX, common::make_buffer(&max_value_rows, 1),
                       common::make_buffer(&max_value, 1));
  else
    max_value = max_value_rows;

  return max_value;
}

template <class T>
void cholesky_diff(Matrix<T, Device::CPU>& original, Matrix<T, Device::CPU>& cholesky_lower,
                   comm::CommunicatorGrid comm_grid) {
  // TODO original and cholesky_lower must be different

  util_matrix::assertSizeSquare(original, "check_cholesky", "original");
  util_matrix::assertBlocksizeSquare(original, "check_cholesky", "original");

  util_matrix::assertSizeSquare(cholesky_lower, "check_cholesky", "cholesky_lower");
  util_matrix::assertBlocksizeSquare(cholesky_lower, "check_cholesky", "cholesky_lower");

  const auto& distribution = cholesky_lower.distribution();
  const auto current_rank = distribution.rankIndex();

  Matrix<T, Device::CPU> mul_result(cholesky_lower.size(), cholesky_lower.blockSize(), comm_grid);

  // k is a global index that keeps track of the diagonal tile
  // it is useful mainly for two reasons:
  // - as limit for when to stop multiplying (because it is triangular and symmetric)
  // - as reference for the row to be used in L, but transposed, as value for L'
  for (SizeType k = 0; k < cholesky_lower.nrTiles().cols(); ++k) {
    const auto k_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Col>(k + 1);

    // workspace for storing the partial results for all the rows in the current rank
    // TODO this size can be reduced to just the part below the current diagonal tile
    Matrix<T, Device::CPU> partial_result({distribution.localSize().rows(),
                                           cholesky_lower.blockSize().cols()},
                                          cholesky_lower.blockSize());

    // TODO evaluate setting to zero (is it possible that a rank does not update it)
    matrix::util::set(partial_result, [](auto&&) { return 0; });

    // for each local column, with the limit of the diagonal tile
    for (SizeType j_loc = 0; j_loc < k_loc; ++j_loc) {
      // identify the tile to be used as 2nd operand in the gemm
      const GlobalTileIndex
          transposed_wrt_global{k, distribution.template globalTileFromLocalTile<Coord::Col>(j_loc)};
      const auto owner_transposed = distribution.rankGlobalTile(transposed_wrt_global);

      // collect the 2nd operand, receving it from others if not available locally
      hpx::shared_future<dlaf::Tile<const T, Device::CPU>> tile_transposed;

      if (owner_transposed == current_rank) {
        // current rank already has what it needs
        tile_transposed = cholesky_lower.read(transposed_wrt_global);

        // if there are more than 1 rank for column, others will need the data from this one
        if (distribution.commGridSize().rows() > 1)
          comm::sync::broadcast::send(comm_grid.colCommunicator(),
                                      cholesky_lower.read(transposed_wrt_global).get());
      }
      else {
        // current rank has to receive it

        // by construction: this rank has the 1st operand, so if it does not have the 2nd one,
        // for sure another rank in the same column will have it (thanks to the regularity of the
        // distribution given by the 2D grid)
        assert(owner_transposed.col() == current_rank.col());

        Tile<T, Device::CPU> workspace(cholesky_lower.blockSize(),
                                       dlaf::memory::MemoryView<T, Device::CPU>(
                                           cholesky_lower.blockSize().rows() *
                                           cholesky_lower.blockSize().cols()),
                                       cholesky_lower.blockSize().rows());

        dlaf::comm::sync::broadcast::receive_from(owner_transposed.row(), comm_grid.colCommunicator(),
                                                  workspace);

        tile_transposed = hpx::make_ready_future<Tile<const T, Device::CPU>>(std::move(workspace));
      }

      // compute the part of results available locally, for each row this rank has in local
      auto i_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(k);
      for (; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
        const LocalTileIndex tile_wrt_local{i_loc, j_loc};

        hpx::dataflow(hpx::util::unwrapping(tile::gemm<T, Device::CPU>), blas::Op::NoTrans,
                      blas::Op::ConjTrans, 1.0, cholesky_lower.read(tile_wrt_local), tile_transposed,
                      j_loc == 0 ? 0.0 : 1.0, std::move(partial_result(LocalTileIndex{i_loc, 0})));
      }
    }

    // now that each rank has computed its partial result with the local data available
    // aggregate the partial result for each row in the current column k
    for (int i_loc = 0; i_loc < partial_result.nrTiles().rows(); ++i_loc) {
      const auto i = distribution.template globalTileFromLocalTile<Coord::Row>(i_loc);
      const GlobalTileIndex tile_result{i, k};
      const auto owner_result = distribution.rankGlobalTile(tile_result);

      common::BufferBasic<T> output_message;
      if (owner_result == current_rank)
        output_message = common::make_buffer(mul_result(tile_result).get());

      comm::sync::reduce(owner_result.col(), comm_grid.rowCommunicator(), MPI_SUM,
                         common::make_buffer(partial_result.read(LocalTileIndex{i_loc, 0}).get()),
                         output_message);

      // L * L' for the current cell is computed
      // here the owner of the result performs the last step (difference with original)
      if (owner_result == current_rank) {
        // TODO check for a blas function to replace it
        hpx::dataflow(hpx::util::unwrapping([](auto&& a, auto&& b) {
                        for (SizeType j = 0; j < a.size().cols(); ++j) {
                          for (SizeType i = 0; i < a.size().rows(); ++i) {
                            const TileElementIndex index_element{i, j};
                            a(index_element) -= b(index_element);
                          }
                        }
                      }),
                      original(tile_result), mul_result.read(tile_result));
      }
    }
  }
}

template <class T>
void check_cholesky(Matrix<T, Device::CPU>& A, Matrix<T, Device::CPU>& L,
                    comm::CommunicatorGrid comm_grid) {
  // norm A (original matrix)
  float norm_A = matrix_norm(A, comm_grid);

  // L is a lower triangular, reset values in the upper part (diagonal excluded)
  // it is needed for the gemm to compute correctly the result when using
  // tiles on the diagonal treating them as all the other ones
  setUpperToZeroForDiagonalTiles(L);

  // compute diff in-place, A = A - L*L'
  cholesky_diff(A, L, comm_grid);

  // norm diff
  float norm_diff = matrix_norm(A, comm_grid);

  if (comm_grid.rank() != comm::Index2D{0, 0})
    return;

  // compute error with the two norms
  constexpr auto eps = std::numeric_limits<T>::epsilon();
  const auto n = A.size().rows();

  const auto diff = norm_diff / norm_A;

  if (diff > 100 * eps * n)
    std::cout << "ERROR: ";
  else if (diff > eps * n)
    std::cout << "Warning: ";

  std::cout << "Max Diff / Max A: " << diff << std::endl;
}
