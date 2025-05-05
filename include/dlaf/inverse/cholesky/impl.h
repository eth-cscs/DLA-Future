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
#include <dlaf/inverse/cholesky/api.h>
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

namespace assemble_cholesky_inv_l {
template <Backend backend, class MatrixTileSender>
void assemble_diag_tile(pika::execution::thread_priority priority, MatrixTileSender&& matrix_tile) {
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Lower, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::lauum(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trmm_row_panel_tile(pika::execution::thread_priority priority, KKTileSender&& kk_tile,
                         MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, blas::Op::ConjTrans,
                                  blas::Diag::NonUnit, ElementType(1.0),
                                  std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herk_matrix_tile(pika::execution::thread_priority priority, PanelTileSender&& panel_tile,
                      MatrixTileSender&& matrix_tile) {
  using BaseElementType = BaseType<dlaf::internal::SenderElementType<MatrixTileSender>>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Lower, blas::Op::ConjTrans, BaseElementType(1.0),
                                  std::forward<PanelTileSender>(panel_tile), BaseElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class RowPanelTileSender, class ColPanelTileSender, class MatrixTileSender>
void gemm_matrix_tile(pika::execution::thread_priority priority, RowPanelTileSender&& row_tile,
                      ColPanelTileSender&& col_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<MatrixTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, ElementType(1.0),
                                  std::forward<RowPanelTileSender>(row_tile),
                                  std::forward<ColPanelTileSender>(col_tile), ElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}
}

namespace assemble_cholesky_inv_u {
template <Backend backend, class MatrixTileSender>
void assemble_diag_tile(pika::execution::thread_priority priority, MatrixTileSender&& matrix_tile) {
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Upper, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::lauum(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trmm_col_panel_tile(pika::execution::thread_priority priority, KKTileSender&& kk_tile,
                         MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::ConjTrans,
                                  blas::Diag::NonUnit, ElementType(1.0),
                                  std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herk_matrix_tile(pika::execution::thread_priority priority, PanelTileSender&& panel_tile,
                      MatrixTileSender&& matrix_tile) {
  using BaseElementType = BaseType<dlaf::internal::SenderElementType<MatrixTileSender>>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Upper, blas::Op::NoTrans, BaseElementType(1.0),
                                  std::forward<PanelTileSender>(panel_tile), BaseElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class RowPanelTileSender, class ColPanelTileSender, class MatrixTileSender>
void gemm_matrix_tile(pika::execution::thread_priority priority, RowPanelTileSender&& row_tile,
                      ColPanelTileSender&& col_tile, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<MatrixTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, ElementType(1.0),
                                  std::forward<RowPanelTileSender>(row_tile),
                                  std::forward<ColPanelTileSender>(col_tile), ElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}
}

template <Backend backend, Device device, class T>
void AssembleCholeskyInverse<backend, device, T>::call_L(Matrix<T, device>& mat_a) {
  using namespace assemble_cholesky_inv_l;
  using pika::execution::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtiles = mat_a.nr_tiles().cols();

  for (SizeType k = 0; k < nrtiles; ++k) {
    auto kk = LocalTileIndex{k, k};

    for (SizeType i = 0; i < k; ++i) {
      for (SizeType j = 0; j < i; ++j) {
        gemm_matrix_tile<backend>(thread_priority::normal, mat_a.read(LocalTileIndex{k, i}),
                                  mat_a.read(LocalTileIndex{k, j}),
                                  mat_a.readwrite(LocalTileIndex{i, j}));
      }

      herk_matrix_tile<backend>(thread_priority::normal, mat_a.read(LocalTileIndex{k, i}),
                                mat_a.readwrite(LocalTileIndex{i, i}));
    }

    for (SizeType j = 0; j < k; ++j) {
      trmm_row_panel_tile<backend>(thread_priority::high, mat_a.read(kk),
                                   mat_a.readwrite(LocalTileIndex{k, j}));
    }

    assemble_diag_tile<backend>(thread_priority::high, mat_a.readwrite(kk));
  }
}

template <Backend backend, Device device, class T>
void AssembleCholeskyInverse<backend, device, T>::call_L(comm::CommunicatorGrid& grid,
                                                         Matrix<T, device>& mat_a) {
  using namespace assemble_cholesky_inv_l;
  using pika::execution::thread_priority;

#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_cholesky_inv_calls = 0;
  std::stringstream fname;
  fname << "inverse-from-cholesky-factor-L-" << matrix::internal::TypeToString_v<T> << "-"
        << std::to_string(num_cholesky_inv_calls) << ".h5";
  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_inverse_from_cholesky_factor_data) {
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
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panels(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device, matrix::StoreTransposed::Yes>> panelsT(
      n_workspaces, dist);

  for (SizeType k = 0; k < nrtiles; ++k) {
    const GlobalTileIndex kk(k, k);
    const comm::Index2D kk_rank = dist.rank_global_tile(kk);

    const SizeType k_local_row = dist.next_local_tile_from_global_tile<Coord::Row>(k);
    const SizeType k_local_col = dist.next_local_tile_from_global_tile<Coord::Col>(k);

    if (k > 0) {
      auto& panel = panels.nextResource();
      auto& panelT = panelsT.nextResource();

      panel.setRange({0, 0}, kk);
      panelT.setRange({1, 1}, kk);

      if (k == nrtiles - 1) {
        panel.setHeight(mat_a.tile_size_of(kk).rows());
        panelT.setWidth(mat_a.tile_size_of(kk).cols());
      }

      if (kk_rank.row() == this_rank.row()) {
        for (SizeType j_local = 0; j_local < k_local_col; ++j_local) {
          const LocalTileIndex kj_panel_local(Coord::Col, j_local);
          const LocalTileIndex kj_local(k_local_row, j_local);
          panel.setTile(kj_panel_local, mat_a.read(kj_local));
        }
      }
      broadcast(kk_rank.row(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

      for (SizeType i = 0; i < k; ++i) {
        const auto ii_rank = dist.rank_global_tile({i, i});
        if (ii_rank.row() != this_rank.row())
          continue;
        const auto i_local = dist.local_tile_from_global_tile<Coord::Row>(i);

        for (SizeType j = 0; j < i; ++j) {
          const auto j_rank = dist.rank_global_tile<Coord::Col>(j);
          if (j_rank != this_rank.col())
            continue;
          const auto j_local = dist.local_tile_from_global_tile<Coord::Col>(j);

          const LocalTileIndex ik_panel_local{Coord::Row, i_local};
          const LocalTileIndex kj_panel_local{Coord::Col, j_local};
          const LocalTileIndex ij_local{i_local, j_local};

          gemm_matrix_tile<backend>(thread_priority::normal, panelT.read(ik_panel_local),
                                    panel.read(kj_panel_local), mat_a.readwrite(ij_local));
        }

        if (ii_rank == this_rank) {
          const auto i_local_col = dist.local_tile_from_global_tile<Coord::Col>(i);
          const LocalTileIndex ki_panel_local{Coord::Col, i_local_col};
          const LocalTileIndex ii_local{i_local, i_local_col};
          herk_matrix_tile<backend>(thread_priority::normal, panel.read(ki_panel_local),
                                    mat_a.readwrite(ii_local));
        }
      }

      panel.reset();
      panelT.reset();

      if (kk_rank.row() == this_rank.row()) {
        panelT.setRange(kk, {k + 1, k + 1});
        const LocalTileIndex kk_panel_local{Coord::Row, k_local_row};
        if (k == nrtiles - 1)
          panelT.setWidth(mat_a.tile_size_of(kk).cols());

        if (kk_rank.col() == this_rank.col())
          panelT.setTile(kk_panel_local, mat_a.read(kk));

        broadcast(kk_rank.col(), panelT, mpi_row_task_chain);

        for (SizeType j_local = 0; j_local < k_local_col; ++j_local) {
          const LocalTileIndex kj_local{k_local_row, j_local};
          trmm_row_panel_tile<backend>(thread_priority::high, panelT.read(kk_panel_local),
                                       mat_a.readwrite(kj_local));
        }
        panelT.reset();
      }
    }

    if (kk_rank == this_rank)
      assemble_diag_tile<backend>(thread_priority::high, mat_a.readwrite(kk));
  }

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_inverse_from_cholesky_factor_data) {
    file->write(mat_a, "/inverse_from_cholesky_factor");
  }

  num_cholesky_inv_calls++;
#endif
}

template <Backend backend, Device device, class T>
void AssembleCholeskyInverse<backend, device, T>::call_U(Matrix<T, device>& mat_a) {
  using namespace assemble_cholesky_inv_u;
  using pika::execution::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtiles = mat_a.nr_tiles().cols();

  for (SizeType k = 0; k < nrtiles; ++k) {
    auto kk = LocalTileIndex{k, k};

    for (SizeType j = 0; j < k; ++j) {
      for (SizeType i = 0; i < j; ++i) {
        gemm_matrix_tile<backend>(thread_priority::normal, mat_a.read(LocalTileIndex{i, k}),
                                  mat_a.read(LocalTileIndex{j, k}),
                                  mat_a.readwrite(LocalTileIndex{i, j}));
      }

      herk_matrix_tile<backend>(thread_priority::normal, mat_a.read(LocalTileIndex{j, k}),
                                mat_a.readwrite(LocalTileIndex{j, j}));
    }

    for (SizeType i = 0; i < k; ++i) {
      trmm_col_panel_tile<backend>(thread_priority::high, mat_a.read(kk),
                                   mat_a.readwrite(LocalTileIndex{i, k}));
    }

    assemble_diag_tile<backend>(thread_priority::high, mat_a.readwrite(kk));
  }
}

template <Backend backend, Device device, class T>
void AssembleCholeskyInverse<backend, device, T>::call_U(comm::CommunicatorGrid& grid,
                                                         Matrix<T, device>& mat_a) {
  using namespace assemble_cholesky_inv_u;
  using pika::execution::thread_priority;

#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_cholesky_inv_calls = 0;
  std::stringstream fname;
  fname << "inverse-from-cholesky-factor-U-" << matrix::internal::TypeToString_v<T> << "-"
        << std::to_string(num_cholesky_inv_calls) << ".h5";
  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_inverse_from_cholesky_factor_data) {
    file = matrix::internal::FileHDF5(grid.fullCommunicator(), fname.str());
    file->write(mat_a, "/input");
  }
#endif

  // Set up MPI executor pipelines
  auto mpi_row_task_chain = grid.row_communicator_pipeline();
  auto mpi_col_task_chain = grid.col_communicator_pipeline();

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& dist = mat_a.distribution();
  const SizeType nrtiles = mat_a.nr_tiles().rows();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panels(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device, matrix::StoreTransposed::Yes>> panelsT(
      n_workspaces, dist);

  for (SizeType k = 0; k < nrtiles; ++k) {
    const GlobalTileIndex kk(k, k);
    const comm::Index2D kk_rank = dist.rank_global_tile(kk);

    const SizeType k_local_row = dist.next_local_tile_from_global_tile<Coord::Row>(k);
    const SizeType k_local_col = dist.next_local_tile_from_global_tile<Coord::Col>(k);

    if (k > 0) {
      auto& panel = panels.nextResource();
      auto& panelT = panelsT.nextResource();

      panel.setRange({0, 0}, kk);
      panelT.setRange({1, 1}, kk);

      if (k == nrtiles - 1) {
        panel.setWidth(mat_a.tile_size_of(kk).cols());
        panelT.setHeight(mat_a.tile_size_of(kk).rows());
      }

      if (kk_rank.col() == this_rank.col()) {
        for (SizeType i_local = 0; i_local < k_local_row; ++i_local) {
          const LocalTileIndex ik_panel_local(Coord::Row, i_local);
          const LocalTileIndex ik_local(i_local, k_local_col);
          panel.setTile(ik_panel_local, mat_a.read(ik_local));
        }
      }
      broadcast(kk_rank.col(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

      for (SizeType j = 0; j < k; ++j) {
        const auto jj_rank = dist.rank_global_tile({j, j});
        if (jj_rank.col() != this_rank.col())
          continue;
        const auto j_local = dist.local_tile_from_global_tile<Coord::Col>(j);

        for (SizeType i = 0; i < j; ++i) {
          const auto i_rank = dist.rank_global_tile<Coord::Row>(i);
          if (i_rank != this_rank.row())
            continue;
          const auto i_local = dist.local_tile_from_global_tile<Coord::Row>(i);

          const LocalTileIndex ik_panel_local{Coord::Row, i_local};
          const LocalTileIndex kj_panel_local{Coord::Col, j_local};
          const LocalTileIndex ij_local{i_local, j_local};

          gemm_matrix_tile<backend>(thread_priority::normal, panel.read(ik_panel_local),
                                    panelT.read(kj_panel_local), mat_a.readwrite(ij_local));
        }

        if (jj_rank == this_rank) {
          const auto j_local_row = dist.local_tile_from_global_tile<Coord::Row>(j);
          const LocalTileIndex jk_panel_local{Coord::Row, j_local_row};
          const LocalTileIndex jj_local{j_local_row, j_local};
          herk_matrix_tile<backend>(thread_priority::normal, panel.read(jk_panel_local),
                                    mat_a.readwrite(jj_local));
        }
      }

      panel.reset();
      panelT.reset();

      if (kk_rank.col() == this_rank.col()) {
        panelT.setRange(kk, {k + 1, k + 1});
        if (k == nrtiles - 1)
          panelT.setHeight(mat_a.tile_size_of(kk).rows());

        const LocalTileIndex kk_panel_local{Coord::Col, k_local_col};

        if (kk_rank.row() == this_rank.row())
          panelT.setTile(kk_panel_local, mat_a.read(kk));

        broadcast(kk_rank.row(), panelT, mpi_col_task_chain);

        for (SizeType i_local = 0; i_local < k_local_row; ++i_local) {
          const LocalTileIndex ik_local{i_local, k_local_col};
          trmm_col_panel_tile<backend>(thread_priority::high, panelT.read(kk_panel_local),
                                       mat_a.readwrite(ik_local));
        }
        panelT.reset();
      }
    }

    if (kk_rank == this_rank)
      assemble_diag_tile<backend>(thread_priority::high, mat_a.readwrite(kk));
  }

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_inverse_from_cholesky_factor_data) {
    file->write(mat_a, "/inverse_from_cholesky_factor");
  }

  num_cholesky_inv_calls++;
#endif
}
}
