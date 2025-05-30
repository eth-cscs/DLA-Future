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
#include <dlaf/factorization/cholesky/api.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/sender/traits.h>
#include <dlaf/util_matrix.h>

namespace dlaf::factorization::internal {

namespace cholesky_l {
template <Backend backend, class MatrixTileSender>
void potrfDiagTile(pika::execution::thread_priority priority, MatrixTileSender&& matrix_tile) {
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Lower, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::potrf(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsmPanelTile(pika::execution::thread_priority priority, KKTileSender&& kk_tile,
                   MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::ConjTrans,
                                  blas::Diag::NonUnit, ElementType(1.0),
                                  std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herkTrailingDiagTile(pika::execution::thread_priority priority, PanelTileSender&& panel_tile,
                          MatrixTileSender&& matrix_tile) {
  using BaseElementType = BaseType<dlaf::internal::SenderElementType<PanelTileSender>>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Lower, blas::Op::NoTrans, BaseElementType(-1.0),
                                  std::forward<PanelTileSender>(panel_tile), BaseElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrixTile(pika::execution::thread_priority priority, PanelTileSender&& panel_tile,
                            ColPanelSender&& col_panel, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, ElementType(-1.0),
                                  std::forward<PanelTileSender>(panel_tile),
                                  std::forward<ColPanelSender>(col_panel), ElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}
}

namespace cholesky_u {
template <Backend backend, class MatrixTileSender>
void potrfDiagTile(pika::execution::thread_priority priority, MatrixTileSender&& matrix_tile) {
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Upper, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::potrf(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsmPanelTile(pika::execution::thread_priority priority, KKTileSender&& kk_tile,
                   MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<KKTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, blas::Op::ConjTrans,
                                  blas::Diag::NonUnit, ElementType(1.0),
                                  std::forward<KKTileSender>(kk_tile),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herkTrailingDiagTile(pika::execution::thread_priority priority, PanelTileSender&& panel_tile,
                          MatrixTileSender&& matrix_tile) {
  using base_element_type = BaseType<dlaf::internal::SenderElementType<PanelTileSender>>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Upper, blas::Op::ConjTrans, base_element_type(-1.0),
                                  std::forward<PanelTileSender>(panel_tile), base_element_type(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}

template <Backend backend, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrixTile(pika::execution::thread_priority priority, PanelTileSender&& panel_tile,
                            ColPanelSender&& col_panel, MatrixTileSender&& matrix_tile) {
  using ElementType = dlaf::internal::SenderElementType<PanelTileSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, ElementType(-1.0),
                                  std::forward<PanelTileSender>(panel_tile),
                                  std::forward<ColPanelSender>(col_panel), ElementType(1.0),
                                  std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority, thread_stacksize::nostack)));
}
}

// Local implementation of Lower Cholesky factorization.
template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(Matrix<T, device>& mat_a) {
  using namespace cholesky_l;
  using pika::execution::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Cholesky decomposition on mat_a.readwrite(k,k) r/w potrf (lapack operation)
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile<backend>(thread_priority::high, mat_a.readwrite(kk));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a.readwrite(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      trsmPanelTile<backend>(thread_priority::high, mat_a.read(kk),
                             mat_a.readwrite(LocalTileIndex{i, k}));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // first trailing panel gets high priority (look ahead).
      const auto trailing_matrix_priority =
          (j == k + 1) ? thread_priority::high : thread_priority::normal;

      // Update trailing matrix: diagonal element mat_a.readwrite(j,j), reading
      // mat_a.read(j,k), using herk (blas operation)
      herkTrailingDiagTile<backend>(trailing_matrix_priority, mat_a.read(LocalTileIndex{j, k}),
                                    mat_a.readwrite(LocalTileIndex{j, j}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a.readwrite(i,j), reading
        // mat_a.read(i,k) and mat_a.read(j,k), using gemm (blas operation)
        gemmTrailingMatrixTile<backend>(trailing_matrix_priority, mat_a.read(LocalTileIndex{i, k}),
                                        mat_a.read(LocalTileIndex{j, k}),
                                        mat_a.readwrite(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(comm::CommunicatorGrid& grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_l;
  using pika::execution::thread_priority;

#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_cholesky_calls = 0;
  std::stringstream fname;
  fname << "cholesky-factorization-" << matrix::internal::TypeToString_v<T> << "-"
        << std::to_string(num_cholesky_calls) << ".h5";
  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_cholesky_factorization_data) {
    file = matrix::internal::FileHDF5(grid.fullCommunicator(), fname.str());
    file->write(mat_a, "/input");
  }
#endif

  // Set up MPI executor pipelines
  auto mpi_row_task_chain = grid.row_communicator_pipeline();
  auto mpi_col_task_chain = grid.col_communicator_pipeline();

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device, matrix::StoreTransposed::Yes>> panelsT(
      n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Factorization of diagonal tile and broadcast it along the k-th column
    if (kk_rank == this_rank)
      potrfDiagTile<backend>(thread_priority::high, mat_a.readwrite(kk_idx));

    // If there is no trailing matrix
    const SizeType kt = k + 1;
    if (kt == nrtile)
      continue;

    auto& panel = panels.nextResource();
    auto& panelT = panelsT.nextResource();

    panel.setRangeStart(GlobalTileIndex(kt, kt));

    if (kk_rank.col() == this_rank.col()) {
      const LocalTileIndex diag_wp_idx{0, distr.localTileFromGlobalTile<Coord::Col>(k)};

      // Note:
      // panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      panelT.setRange({k, k}, {kt, kt});

      if (kk_rank.row() == this_rank.row())
        panelT.setTile(diag_wp_idx, mat_a.read(kk_idx));
      broadcast(kk_rank.row(), panelT, mpi_col_task_chain);

      // COLUMN UPDATE
      for (SizeType i = distr.nextLocalTileFromGlobalTile<Coord::Row>(kt);
           i < distr.localNrTiles().rows(); ++i) {
        const LocalTileIndex local_idx(Coord::Row, i);
        const LocalTileIndex ik_idx(i, distr.localTileFromGlobalTile<Coord::Col>(k));

        trsmPanelTile<backend>(thread_priority::high, panelT.read(diag_wp_idx), mat_a.readwrite(ik_idx));

        panel.setTile(local_idx, mat_a.read(ik_idx));
      }

      // row panel has been used for temporary storage of diagonal panel for column update
      panelT.reset();
    }

    panelT.setRange({kt, kt}, {nrtile - 1, nrtile - 1});

    broadcast(kk_rank.col(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

    // TRAILING MATRIX
    for (SizeType jt_idx = kt; jt_idx < nrtile; ++jt_idx) {
      const auto owner = distr.rankGlobalTile({jt_idx, jt_idx});

      if (owner.col() != this_rank.col())
        continue;

      const auto j = distr.localTileFromGlobalTile<Coord::Col>(jt_idx);
      const auto trailing_matrix_priority =
          (jt_idx == kt) ? thread_priority::high : thread_priority::normal;
      if (this_rank.row() == owner.row()) {
        const auto i = distr.localTileFromGlobalTile<Coord::Row>(jt_idx);

        herkTrailingDiagTile<backend>(trailing_matrix_priority, panel.read({Coord::Row, i}),
                                      mat_a.readwrite(LocalTileIndex{i, j}));
      }

      for (SizeType i_idx = jt_idx + 1; i_idx < nrtile; ++i_idx) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i_idx);

        if (owner_row != this_rank.row())
          continue;

        const auto i = distr.localTileFromGlobalTile<Coord::Row>(i_idx);
        gemmTrailingMatrixTile<backend>(trailing_matrix_priority, panel.read({Coord::Row, i}),
                                        panelT.read({Coord::Col, j}),
                                        mat_a.readwrite(LocalTileIndex{i, j}));
      }
    }

    panel.reset();
    panelT.reset();
  }

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_cholesky_factorization_data) {
    file->write(mat_a, "/cholesky");
  }

  num_cholesky_calls++;
#endif
}

// Local implementation of Upper Cholesky factorization.
template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_U(Matrix<T, device>& mat_a) {
  using namespace cholesky_u;
  using pika::execution::thread_priority;

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile<backend>(thread_priority::high, mat_a.readwrite(kk));

    for (SizeType j = k + 1; j < nrtile; ++j) {
      trsmPanelTile<backend>(thread_priority::high, mat_a.read(kk),
                             mat_a.readwrite(LocalTileIndex{k, j}));
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const auto trailing_matrix_priority =
          (i == k + 1) ? thread_priority::high : thread_priority::normal;

      herkTrailingDiagTile<backend>(trailing_matrix_priority, mat_a.read(LocalTileIndex{k, i}),
                                    mat_a.readwrite(LocalTileIndex{i, i}));

      for (SizeType j = i + 1; j < nrtile; ++j) {
        gemmTrailingMatrixTile<backend>(trailing_matrix_priority, mat_a.read(LocalTileIndex{k, i}),
                                        mat_a.read(LocalTileIndex{k, j}),
                                        mat_a.readwrite(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_U(comm::CommunicatorGrid& grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_u;
  using pika::execution::thread_priority;

  // Set up MPI executor pipelines
  auto mpi_row_task_chain = grid.row_communicator_pipeline();
  auto mpi_col_task_chain = grid.col_communicator_pipeline();

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device, matrix::StoreTransposed::Yes>> panelsT(
      n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Factorization of diagonal tile and broadcast it along the k-th column
    if (kk_rank == this_rank) {
      potrfDiagTile<backend>(thread_priority::high, mat_a.readwrite(kk_idx));
    }

    // If there is no trailing matrix
    const SizeType kt = k + 1;
    if (kt == nrtile)
      continue;

    auto& panel = panels.nextResource();
    auto& panelT = panelsT.nextResource();

    panel.setRangeStart(GlobalTileIndex(kt, kt));

    if (kk_rank.row() == this_rank.row()) {
      const LocalTileIndex diag_wp_idx{distr.localTileFromGlobalTile<Coord::Row>(k), 0};
      // Note:
      // panel shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the row update
      panelT.setRange({k, k}, {kt, kt});

      if (kk_rank.col() == this_rank.col())
        panelT.setTile(diag_wp_idx, mat_a.read(kk_idx));
      broadcast(kk_rank.col(), panelT, mpi_row_task_chain);

      // ROW UPDATE
      for (SizeType j = distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1);
           j < distr.localNrTiles().cols(); ++j) {
        const LocalTileIndex local_idx(Coord::Col, j);
        const LocalTileIndex kj_idx(distr.localTileFromGlobalTile<Coord::Row>(k), j);

        trsmPanelTile<backend>(thread_priority::high, panelT.read(diag_wp_idx), mat_a.readwrite(kj_idx));

        panel.setTile(local_idx, mat_a.read(kj_idx));
      }

      // col panel has been used for temporary storage of diagonal panel for column update
      panelT.reset();
    }

    panelT.setRange({kt, kt}, {nrtile - 1, nrtile - 1});

    broadcast(kk_rank.row(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

    // TRAILING MATRIX
    for (SizeType it_idx = kt; it_idx < nrtile; ++it_idx) {
      const auto owner = distr.rankGlobalTile({it_idx, it_idx});

      if (owner.row() != this_rank.row())
        continue;

      const auto i = distr.localTileFromGlobalTile<Coord::Row>(it_idx);
      const auto trailing_matrix_priority =
          (i == k + 1) ? thread_priority::high : thread_priority::normal;
      if (this_rank.col() == owner.col()) {
        const auto j = distr.localTileFromGlobalTile<Coord::Col>(it_idx);

        herkTrailingDiagTile<backend>(trailing_matrix_priority, panel.read({Coord::Col, j}),
                                      mat_a.readwrite(LocalTileIndex{i, j}));
      }

      for (SizeType j_idx = it_idx + 1; j_idx < nrtile; ++j_idx) {
        const auto owner_col = distr.rankGlobalTile<Coord::Col>(j_idx);

        if (owner_col != this_rank.col())
          continue;

        const auto j = distr.localTileFromGlobalTile<Coord::Col>(j_idx);

        gemmTrailingMatrixTile<backend>(trailing_matrix_priority, panelT.read({Coord::Row, i}),
                                        panel.read({Coord::Col, j}),
                                        mat_a.readwrite(LocalTileIndex{i, j}));
      }
    }

    panel.reset();
    panelT.reset();
  }
}
}
