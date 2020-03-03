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
#include "dlaf/communication/sync/broadcast.h"
#include "dlaf/communication/sync/reduce.h"
#include "dlaf/matrix.h"
#include "dlaf/util_matrix.h"

using namespace dlaf;

template <class T>
void setUpperZero(Matrix<T, Device::CPU>& matrix) {
  util_matrix::assertBlocksizeSquare(matrix, "setUpperZero", "matrix");

  auto& distribution = matrix.distribution();

  for (int j = 0; j < distribution.localNrTiles().cols(); ++j) {
    for (int i = 0; i < distribution.localNrTiles().rows(); ++i) {
      auto tile_wrt_global = distribution.globalTileIndex(LocalTileIndex{i, j});
      auto tl_index = distribution.globalElementIndex(tile_wrt_global, {0, 0});

      if (tile_wrt_global.row() <= tile_wrt_global.col()) {
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
}

template <class T>
T matrix_norm(Matrix<const T, Device::CPU>& matrix, comm::CommunicatorGrid comm_grid) {
  const auto& distribution = matrix.distribution();
  const auto current_rank = distribution.rankIndex();

  std::vector<hpx::future<T>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  for (SizeType j_loc = 0; j_loc < distribution.localNrTiles().cols(); ++j_loc) {
    const SizeType j = distribution.template globalTileFromLocalTile<Coord::Col>(j_loc);
    SizeType i_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(j);
    for (; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
      auto current_tile_max =
          hpx::dataflow(hpx::util::unwrapping([](auto&& tile) -> T {
                          T tile_max_value = std::abs(T{0});
                          for (SizeType j = 0; j < tile.size().cols(); ++j)
                            for (SizeType i = j; i < tile.size().rows(); ++i)
                              tile_max_value = std::max(tile_max_value, tile({i, j}));
                          return tile_max_value;
                        }),
                        matrix.read(LocalTileIndex{i_loc, j_loc}));
      tiles_max.emplace_back(std::move(current_tile_max));
    }
  }

  // TODO unwrapping can be skipped for optimization reasons
  auto local_max_value = hpx::dataflow(hpx::util::unwrapping([](const auto&& values) {
                                         // TODO some rank may not have any element
                                         if (values.size() == 0)
                                           return std::numeric_limits<T>::min();
                                         return *std::max_element(values.begin(), values.end());
                                       }),
                                       tiles_max)
                             .get();

  // TODO reduce only if there are multiple rows/cols
  T max_value_rows;
  T max_value;
  comm::sync::reduce(0, comm_grid.rowCommunicator(), MPI_MAX, common::make_buffer(&local_max_value, 1),
                     common::make_buffer(&max_value_rows, 1));
  comm::sync::reduce(0, comm_grid.colCommunicator(), MPI_MAX, common::make_buffer(&max_value_rows, 1),
                     common::make_buffer(&max_value, 1));

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
  for (SizeType k = 0; k < cholesky_lower.nrTiles().cols(); ++k) {
    const auto k_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Col>(k + 1);

    // workspace for storing the partial results for all the rows in the current rank
    // TODO this size can be reduced to just the part below the current diagonal tile
    Matrix<T, Device::CPU> partial_result({distribution.localSize().rows(),
                                           cholesky_lower.blockSize().cols()},
                                          cholesky_lower.blockSize());

    // TODO evaluate setting to zero (is it possible that a rank does not update it)
    matrix::util::set(partial_result, [](auto&&) { return 0; });

    for (SizeType j_loc = 0; j_loc < k_loc; ++j_loc) {
      const GlobalTileIndex
          transposed_wrt_global{k, distribution.template globalTileFromLocalTile<Coord::Col>(j_loc)};
      const auto owner_transposed = distribution.rankGlobalTile(transposed_wrt_global);

      hpx::shared_future<dlaf::Tile<const T, Device::CPU>> tile_transposed;

      if (owner_transposed == current_rank) {
        // current rank already has what it needs
        tile_transposed = cholesky_lower.read(transposed_wrt_global);

        if (distribution.commGridSize().rows() > 1)
          comm::sync::broadcast::send(comm_grid.colCommunicator(),
                                      cholesky_lower.read(transposed_wrt_global).get());
      }
      else {
        // current rank has to receive the "transposed block"
        Tile<T, Device::CPU> workspace(cholesky_lower.blockSize(),
                                       dlaf::memory::MemoryView<T, Device::CPU>(
                                           cholesky_lower.blockSize().rows() *
                                           cholesky_lower.blockSize().cols()),
                                       cholesky_lower.blockSize().rows());

        dlaf::comm::sync::broadcast::receive_from(owner_transposed.row(), comm_grid.colCommunicator(),
                                                  workspace);

        tile_transposed = hpx::make_ready_future<Tile<const T, Device::CPU>>(std::move(workspace));
      }

      auto i_loc = distribution.template nextLocalTileFromGlobalTile<Coord::Row>(k);
      for (; i_loc < distribution.localNrTiles().rows(); ++i_loc) {
        const LocalTileIndex tile_wrt_local{i_loc, j_loc};
        const GlobalTileIndex tile_wrt_global = distribution.globalTileIndex(tile_wrt_local);

        hpx::dataflow(hpx::util::unwrapping(tile::gemm<T, Device::CPU>), blas::Op::NoTrans,
                      blas::Op::ConjTrans, 1.0, cholesky_lower.read(tile_wrt_global), tile_transposed,
                      j_loc == 0 ? 0.0 : 1.0, std::move(partial_result(LocalTileIndex{i_loc, 0})));
      }
    }

    for (int i_loc = 0; i_loc < partial_result.nrTiles().rows(); ++i_loc) {
      const auto i = distribution.template globalTileFromLocalTile<Coord::Row>(i_loc);
      const GlobalTileIndex tile_result{i, k};
      const auto owner_result = distribution.rankGlobalTile(tile_result);

      const auto reduce_operator = MPI_SUM;
      auto row_comm = comm_grid.rowCommunicator();
      auto input_message = common::make_buffer(partial_result.read(LocalTileIndex{i_loc, 0}).get());

      if (owner_result == current_rank)
        comm::sync::internal::reduce::collector(row_comm, reduce_operator, std::move(input_message),
                                                common::make_buffer(mul_result(tile_result).get()));
      else
        comm::sync::internal::reduce::participant(owner_result.col(), row_comm, reduce_operator,
                                                  std::move(input_message));

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
  setUpperZero(L);

  // norm A (original matrix)
  float norm_A = matrix_norm(A, comm_grid);

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
