//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <cstddef>
#include <utility>

#ifdef DLAF_WITH_HDF5
#include <atomic>
#include <sstream>
#endif

#include <pika/execution.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/kernels.h>
#include <dlaf/inverse/triangular/api.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/util_matrix.h>

namespace dlaf::inverse::internal {

namespace triangular_inv_l {
template <Backend backend, class MatrixTileSender>
void inverse_diag_tile(pika::execution::thread_priority priority, blas::Diag diag,
                       MatrixTileSender&& matrix_tile) {
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Lower, diag, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trtri(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsm_col_panel_tile(pika::execution::thread_priority priority, blas::Diag diag,
                         KKTileSender&& kk_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::NoTrans, diag,
                                  ElementType(-1.0), std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsm_row_panel_tile(pika::execution::thread_priority priority, blas::Diag diag,
                         KKTileSender&& kk_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans, diag,
                                  ElementType(1.0), std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class ColPanelTileSender, class RowPanelTileSender, class MatrixTileSender>
void gemm_matrix_tile(pika::execution::thread_priority priority, ColPanelTileSender&& col_tile,
                      RowPanelTileSender&& row_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<MatrixTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, ElementType(1.0),
                                  std::forward<ColPanelTileSender>(col_tile),
                                  std::forward<RowPanelTileSender>(row_tile), ElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}
}
namespace triangular_inv_u {
template <Backend backend, class MatrixTileSender>
void inverse_diag_tile(pika::execution::thread_priority priority, blas::Diag diag,
                       MatrixTileSender&& matrix_tile) {
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Upper, diag, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trtri(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsm_col_panel_tile(pika::execution::thread_priority priority, blas::Diag diag,
                         KKTileSender&& kk_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, diag,
                                  ElementType(1.0), std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsm_row_panel_tile(pika::execution::thread_priority priority, blas::Diag diag,
                         KKTileSender&& kk_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans, diag,
                                  ElementType(-1.0), std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class ColPanelTileSender, class RowPanelTileSender, class MatrixTileSender>
void gemm_matrix_tile(pika::execution::thread_priority priority, ColPanelTileSender&& col_tile,
                      RowPanelTileSender&& row_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<MatrixTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, ElementType(1.0),
                                  std::forward<ColPanelTileSender>(col_tile),
                                  std::forward<RowPanelTileSender>(row_tile), ElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_L(blas::Diag diag, Matrix<T, device>& mat_a) {
  using namespace triangular_inv_l;
  using pika::execution::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtiles = mat_a.nr_tiles().cols();

  for (SizeType k = nrtiles - 1; k >= 0; --k) {
    auto kk = LocalTileIndex{k, k};

    for (SizeType i = k + 1; i < nrtiles; ++i) {
      trsm_col_panel_tile<backend>(thread_priority::high, diag, mat_a.read(kk),
                                   mat_a.readwrite(LocalTileIndex{i, k}));
      for (SizeType j = 0; j < k; ++j) {
        const auto trailing_matrix_priority =
            (j == k - 1) ? thread_priority::high : thread_priority::normal;

        gemm_matrix_tile<backend>(trailing_matrix_priority, mat_a.read(LocalTileIndex{i, k}),
                                  mat_a.read(LocalTileIndex{k, j}),
                                  mat_a.readwrite(LocalTileIndex{i, j}));
      }
    }

    for (SizeType j = 0; j < k; ++j) {
      trsm_row_panel_tile<backend>(thread_priority::high, diag, mat_a.read(kk),
                                   mat_a.readwrite(LocalTileIndex{k, j}));
    }

    inverse_diag_tile<backend>(thread_priority::high, diag, mat_a.readwrite(kk));
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_L(comm::CommunicatorGrid& grid, blas::Diag diag,
                                            Matrix<T, device>& mat_a) {
  using namespace triangular_inv_l;
  using pika::execution::thread_priority;

#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_triangular_inv_calls = 0;
  std::stringstream fname;
  fname << "triangular-inverse-" << matrix::internal::TypeToString_v<T> << "-"
        << std::to_string(num_triangular_inv_calls) << ".h5";
  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_triangular_inverse_data) {
    file = matrix::internal::FileHDF5(grid.fullCommunicator(), fname.str());
    file->write(mat_a, "/input");
  }
#endif

  // Set up MPI executor pipelines
  auto mpi_row_task_chain = grid.row_communicator_pipeline();
  auto mpi_col_task_chain = grid.col_communicator_pipeline();

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& dist = mat_a.distribution();
  const SizeType nrtiles = mat_a.nr_tiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> col_panels(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> row_panels(n_workspaces, dist);

  const SizeType i_local_end = dist.local_nr_tiles().rows();

  for (SizeType k = nrtiles - 1; k >= 0; --k) {
    const GlobalTileIndex kk(k, k);
    const comm::Index2D kk_rank = dist.rank_global_tile(kk);

    const SizeType k_local_row = dist.next_local_tile_from_global_tile<Coord::Row>(k);
    const SizeType k1_local_row = dist.next_local_tile_from_global_tile<Coord::Row>(k + 1);
    const SizeType k_local_col = dist.next_local_tile_from_global_tile<Coord::Col>(k);
    const SizeType k1_local_col = dist.next_local_tile_from_global_tile<Coord::Col>(k + 1);

    auto& col_panel = col_panels.nextResource();
    auto& row_panel = row_panels.nextResource();

    if (k < nrtiles - 1) {
      row_panel.setRange({0, 0}, {k + 1, k + 1});
      if (k == 0)
        row_panel.setHeight(mat_a.tile_size_of(kk).rows());

      if (kk_rank.row() == this_rank.row()) {
        for (SizeType j_local = 0; j_local < k1_local_col; ++j_local) {
          const LocalTileIndex kj_panel(Coord::Col, j_local);
          const LocalTileIndex kj(k_local_row, j_local);

          row_panel.setTile(kj_panel, mat_a.read(kj));
        }
      }
      broadcast(kk_rank.row(), row_panel, mpi_col_task_chain);
    }

    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = k1_local_row; i_local < i_local_end; ++i_local) {
        const LocalTileIndex kk_local_panel{Coord::Col, k_local_col};
        const LocalTileIndex ik_local{i_local, k_local_col};
        trsm_col_panel_tile<backend>(thread_priority::high, diag, row_panel.read(kk_local_panel),
                                     mat_a.readwrite(ik_local));
      }
    }

    if (k > 0) {
      col_panel.setRangeStart(kk);
      if (k == nrtiles - 1)
        col_panel.setWidth(mat_a.tile_size_of(kk).cols());

      if (kk_rank.col() == this_rank.col()) {
        for (SizeType i_local = k_local_row; i_local < i_local_end; ++i_local) {
          const LocalTileIndex ik_panel(Coord::Row, i_local);
          const LocalTileIndex ik(i_local, k_local_col);

          col_panel.setTile(ik_panel, mat_a.read(ik));
        }
      }
      broadcast(kk_rank.col(), col_panel, mpi_row_task_chain);
    }

    for (SizeType i_local = k1_local_row; i_local < i_local_end; ++i_local) {
      for (SizeType j_local = 0; j_local < k_local_col; ++j_local) {
        const LocalTileIndex ik_local_panel{Coord::Row, i_local};
        const LocalTileIndex kj_local_panel{Coord::Col, j_local};
        const LocalTileIndex ij_local{i_local, j_local};

        const auto trailing_matrix_priority =
            (j_local == k_local_col - 1) ? thread_priority::high : thread_priority::normal;

        gemm_matrix_tile<backend>(trailing_matrix_priority, col_panel.read(ik_local_panel),
                                  row_panel.read(kj_local_panel), mat_a.readwrite(ij_local));
      }
    }

    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = 0; j_local < k_local_col; ++j_local) {
        const LocalTileIndex kk_local_panel{Coord::Row, k_local_row};
        const LocalTileIndex kj_local{k_local_row, j_local};
        trsm_row_panel_tile<backend>(thread_priority::high, diag, col_panel.read(kk_local_panel),
                                     mat_a.readwrite(kj_local));
      }
    }

    col_panel.reset();
    row_panel.reset();

    if (kk_rank == this_rank)
      inverse_diag_tile<backend>(thread_priority::high, diag, mat_a.readwrite(kk));
  }

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_triangular_inverse_data) {
    file->write(mat_a, "/triangular_inverse");
  }

  num_triangular_inv_calls++;
#endif
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_U(blas::Diag diag, Matrix<T, device>& mat_a) {
  using namespace triangular_inv_u;
  using pika::execution::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtiles = mat_a.nr_tiles().cols();

  for (SizeType k = nrtiles - 1; k >= 0; --k) {
    auto kk = LocalTileIndex{k, k};

    for (SizeType j = k + 1; j < nrtiles; ++j) {
      trsm_row_panel_tile<backend>(thread_priority::high, diag, mat_a.read(kk),
                                   mat_a.readwrite(LocalTileIndex{k, j}));
      for (SizeType i = 0; i < k; ++i) {
        const auto trailing_matrix_priority =
            (i == k - 1) ? thread_priority::high : thread_priority::normal;

        gemm_matrix_tile<backend>(trailing_matrix_priority, mat_a.read(LocalTileIndex{i, k}),
                                  mat_a.read(LocalTileIndex{k, j}),
                                  mat_a.readwrite(LocalTileIndex{i, j}));
      }
    }

    for (SizeType i = 0; i < k; ++i) {
      trsm_col_panel_tile<backend>(thread_priority::high, diag, mat_a.read(kk),
                                   mat_a.readwrite(LocalTileIndex{i, k}));
    }

    inverse_diag_tile<backend>(thread_priority::high, diag, mat_a.readwrite(kk));
  }
}
}
