//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <hpx/local/future.hpp>
#include <hpx/local/unwrap.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/executors.h"
#include "dlaf/factorization/cholesky/api.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace factorization {
namespace internal {

// TODO: Move.
template <typename ValueTypes>
struct sender_single_value_type_impl {};

template <typename T>
struct sender_single_value_type_impl<hpx::util::pack<hpx::util::pack<T>>> {
  using type = T;
};

// We are only interested in the types wrapped by future and shared_future since
// we will internally unwrap them.
template <typename T>
struct sender_single_value_type_impl<hpx::util::pack<hpx::util::pack<hpx::future<T>>>> {
  using type = T;
};

template <typename T>
struct sender_single_value_type_impl<hpx::util::pack<hpx::util::pack<hpx::shared_future<T>>>> {
  using type = T;
};

// Contains a typedef type if and only if the sender sends a single type. If
// that is the case, type will be defined to the type sent by the sender.
template <typename Sender>
struct sender_single_value_type
    : sender_single_value_type_impl<typename hpx::execution::experimental::sender_traits<
          Sender>::template value_types<hpx::util::pack, hpx::util::pack>> {};

template <typename Sender>
using sender_single_value_type_t = typename sender_single_value_type<Sender>::type;

template <typename T>
struct tile_sender_element_type_impl {
  using type = typename T::ElementType;
};

// Contains a typedef type if and only if the sender sends a single Tile. If
// that is the case, type will be defined to ElementType of the Tile.
template <typename Sender>
struct tile_sender_element_type : tile_sender_element_type_impl<sender_single_value_type_t<Sender>> {};

template <typename Sender>
using tile_sender_element_type_t = typename tile_sender_element_type<Sender>::type;

// Contains a boolean variable value which is true if and only if the given type
// is a Tile.
template <typename T>
struct is_tile : std::false_type {};

template <typename T, Device device>
struct is_tile<matrix::Tile<T, device>> : std::true_type {};

template <typename T>
inline constexpr bool is_tile_v = is_tile<T>::value;

// Contains a boolean variable value which is true if and only if the given type
// is a Tile with const ElementType.
template <typename T>
struct is_const_tile : std::false_type {};

template <typename T, Device device>
struct is_const_tile<matrix::Tile<const T, device>> : std::true_type {};

template <typename T>
inline constexpr bool is_const_tile_v = is_const_tile<T>::value;

template <typename Sender, typename Enable = void>
struct is_sender_of_tile : std::false_type {};

// Contains a boolean variable value which is true if and only if the given sender
// sends a Tile.
template <typename Sender>
struct is_sender_of_tile<Sender, std::enable_if_t<is_tile_v<sender_single_value_type_t<Sender>>>>
    : std::true_type {};

template <typename Sender>
inline constexpr bool is_sender_of_tile_v = is_sender_of_tile<Sender>::value;

// Contains a boolean variable value which is true if and only if the given sender
// sends a Tile with const element type.
template <typename Sender, typename Enable = void>
struct is_sender_of_const_tile : std::false_type {};

template <typename Sender>
struct is_sender_of_const_tile<Sender,
                               std::enable_if_t<is_const_tile_v<sender_single_value_type_t<Sender>>>>
    : std::true_type {};

template <typename Sender>
inline constexpr bool is_sender_of_const_tile_v = is_sender_of_const_tile<Sender>::value;

// Contains a boolean variable value which is true if and only all given senders
// send a tile and the element type of the tiles are the same.
template <typename Sender, typename... Senders>
struct have_same_element_types
    : std::conjunction<
          std::is_same<tile_sender_element_type_t<Sender>, tile_sender_element_type_t<Senders>>...> {};

template <typename... Senders>
inline constexpr bool have_same_element_types_v = have_same_element_types<Senders...>::value;

namespace cholesky_l {
template <Backend backend, class MatrixTileSender>
void potrfDiagTile(MatrixTileSender&& matrix_tile) {
  static_assert(is_sender_of_tile_v<MatrixTileSender>);

  dlaf::internal::whenAllLift(blas::Uplo::Lower, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::potrf(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::normal)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsmPanelTile(KKTileSender&& kk_tile, MatrixTileSender&& matrix_tile) {
  static_assert(is_sender_of_const_tile_v<KKTileSender>);
  static_assert(is_sender_of_tile_v<MatrixTileSender>);
  static_assert(have_same_element_types_v<KKTileSender, MatrixTileSender>);

  using element_type = tile_sender_element_type_t<KKTileSender>;

  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, element_type(1.0),
                              std::forward<KKTileSender>(kk_tile),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herkTrailingDiagTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                          MatrixTileSender&& matrix_tile) {
  static_assert(is_sender_of_const_tile_v<PanelTileSender>);
  static_assert(is_sender_of_tile_v<MatrixTileSender>);
  static_assert(have_same_element_types_v<PanelTileSender, MatrixTileSender>);

  using base_element_type = BaseType<tile_sender_element_type_t<PanelTileSender>>;

  dlaf::internal::whenAllLift(blas::Uplo::Lower, blas::Op::NoTrans, base_element_type(-1.0),
                              std::forward<PanelTileSender>(panel_tile), base_element_type(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                            ColPanelSender&& col_panel, MatrixTileSender&& matrix_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, T(-1.0),
                              std::forward<PanelTileSender>(panel_tile),
                              std::forward<ColPanelSender>(col_panel), T(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace cholesky_u {
template <Backend backend, class MatrixTileSender>
void potrfDiagTile(MatrixTileSender&& matrix_tile) {
  dlaf::internal::whenAllLift(blas::Uplo::Upper, std::forward<MatrixTileSender>(matrix_tile)) |
      tile::potrf(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::normal)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class KKTileSender, class MatrixTileSender>
void trsmPanelTile(KKTileSender&& kk_tile, MatrixTileSender&& matrix_tile) {
  static_assert(is_sender_of_const_tile_v<KKTileSender>);
  static_assert(is_sender_of_tile_v<MatrixTileSender>);
  static_assert(have_same_element_types_v<KKTileSender, MatrixTileSender>);

  using element_type = tile_sender_element_type_t<KKTileSender>;

  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, blas::Op::ConjTrans,
                              blas::Diag::NonUnit, element_type(1.0),
                              std::forward<KKTileSender>(kk_tile),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::trsm(dlaf::internal::Policy<backend>(hpx::threads::thread_priority::high)) |
      hpx::execution::experimental::detach();
}

template <Backend backend, class PanelTileSender, class MatrixTileSender>
void herkTrailingDiagTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                          MatrixTileSender&& matrix_tile) {
  static_assert(is_sender_of_const_tile_v<PanelTileSender>);
  static_assert(is_sender_of_tile_v<MatrixTileSender>);
  static_assert(have_same_element_types_v<PanelTileSender, MatrixTileSender>);

  using base_element_type = BaseType<tile_sender_element_type_t<PanelTileSender>>;

  dlaf::internal::whenAllLift(blas::Uplo::Upper, blas::Op::ConjTrans, base_element_type(-1.0),
                              std::forward<PanelTileSender>(panel_tile), base_element_type(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::herk(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, class PanelTileSender, class ColPanelSender, class MatrixTileSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, PanelTileSender&& panel_tile,
                            ColPanelSender&& col_panel, MatrixTileSender&& matrix_tile) {
  dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, T(-1.0),
                              std::forward<PanelTileSender>(panel_tile),
                              std::forward<ColPanelSender>(col_panel), T(1.0),
                              std::forward<MatrixTileSender>(matrix_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

// Local implementation of Lower Cholesky factorization.
template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(Matrix<T, device>& mat_a) {
  using namespace cholesky_l;

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    // Cholesky decomposition on mat_a(k,k) r/w potrf (lapack operation)
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile<backend>(mat_a.readwrite_sender(kk));

    for (SizeType i = k + 1; i < nrtile; ++i) {
      // Update panel mat_a(i,k) with trsm (blas operation), using data mat_a.read(k,k)
      trsmPanelTile<backend>(mat_a.read_sender(kk), mat_a.readwrite_sender(LocalTileIndex{i, k}));
    }

    for (SizeType j = k + 1; j < nrtile; ++j) {
      // first trailing panel gets high priority (look ahead).
      const auto trailing_matrix_priority =
          (j == k + 1) ? hpx::threads::thread_priority::high : hpx::threads::thread_priority::normal;

      // Update trailing matrix: diagonal element mat_a(j,j), reading mat_a.read(j,k), using herk (blas operation)
      herkTrailingDiagTile<backend>(trailing_matrix_priority, mat_a.read_sender(LocalTileIndex{j, k}),
                                    mat_a.readwrite_sender(LocalTileIndex{j, j}));

      for (SizeType i = j + 1; i < nrtile; ++i) {
        // Update remaining trailing matrix mat_a(i,j), reading mat_a.read(i,k) and mat_a.read(j,k),
        // using gemm (blas operation)
        gemmTrailingMatrixTile<backend, T>(hpx::threads::thread_priority::normal,
                                           mat_a.read_sender(LocalTileIndex{i, k}),
                                           mat_a.read_sender(LocalTileIndex{j, k}),
                                           mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_L(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_l;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panelsT(n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Factorization of diagonal tile and broadcast it along the k-th column
    if (kk_rank == this_rank)
      potrfDiagTile<backend>(mat_a.readwrite_sender(kk_idx));

    // If there is no trailing matrix
    const SizeType kt = k + 1;
    if (kt == nrtile)
      continue;

    auto& panel = panels.nextResource();
    auto& panelT = panelsT.nextResource();

    panel.setRangeStart({kt, kt});

    if (kk_rank.col() == this_rank.col()) {
      const LocalTileIndex diag_wp_idx{0, distr.localTileFromGlobalTile<Coord::Col>(k)};

      // Note:
      // panelT shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the column update
      panelT.setRange({k, k}, {kt, kt});

      if (kk_rank.row() == this_rank.row())
        panelT.setTile(diag_wp_idx, mat_a.read(kk_idx));
      broadcast(executor_mpi, kk_rank.row(), panelT, mpi_col_task_chain);

      // COLUMN UPDATE
      for (SizeType i = distr.nextLocalTileFromGlobalTile<Coord::Row>(kt);
           i < distr.localNrTiles().rows(); ++i) {
        const LocalTileIndex local_idx(Coord::Row, i);
        const LocalTileIndex ik_idx(i, distr.localTileFromGlobalTile<Coord::Col>(k));

        trsmPanelTile<backend>(panelT.read_sender(diag_wp_idx), mat_a.readwrite_sender(ik_idx));

        panel.setTile(local_idx, mat_a.read(ik_idx));
      }

      // row panel has been used for temporary storage of diagonal panel for column update
      panelT.reset();
    }

    panelT.setRange({kt, kt}, indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.col(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

    // TRAILING MATRIX
    for (SizeType jt_idx = kt; jt_idx < nrtile; ++jt_idx) {
      const auto owner = distr.rankGlobalTile({jt_idx, jt_idx});

      if (owner.col() != this_rank.col())
        continue;

      const auto j = distr.localTileFromGlobalTile<Coord::Col>(jt_idx);
      const auto trailing_matrix_priority =
          (jt_idx == kt) ? hpx::threads::thread_priority::high : hpx::threads::thread_priority::normal;
      if (this_rank.row() == owner.row()) {
        const auto i = distr.localTileFromGlobalTile<Coord::Row>(jt_idx);

        herkTrailingDiagTile<backend>(trailing_matrix_priority, panel.read_sender({Coord::Row, i}),
                                      mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }

      for (SizeType i_idx = jt_idx + 1; i_idx < nrtile; ++i_idx) {
        const auto owner_row = distr.rankGlobalTile<Coord::Row>(i_idx);

        if (owner_row != this_rank.row())
          continue;

        const auto i = distr.localTileFromGlobalTile<Coord::Row>(i_idx);
        // TODO: This was using executor_np. Was that intentional, or should it
        // be trailing_matrix_executor/priority?
        gemmTrailingMatrixTile<backend, T>(hpx::threads::thread_priority::normal,
                                           panel.read_sender({Coord::Row, i}),
                                           panelT.read_sender({Coord::Col, j}),
                                           mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }

    panel.reset();
    panelT.reset();
  }
}

// Local implementation of Upper Cholesky factorization.
template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_U(Matrix<T, device>& mat_a) {
  using namespace cholesky_u;

  // Number of tile (rows = cols)
  SizeType nrtile = mat_a.nrTiles().cols();

  for (SizeType k = 0; k < nrtile; ++k) {
    auto kk = LocalTileIndex{k, k};

    potrfDiagTile<backend>(mat_a.readwrite_sender(kk));

    for (SizeType j = k + 1; j < nrtile; ++j) {
      trsmPanelTile<backend>(mat_a.read_sender(kk), mat_a.readwrite_sender(LocalTileIndex{k, j}));
    }

    for (SizeType i = k + 1; i < nrtile; ++i) {
      const auto trailing_matrix_priority =
          (i == k + 1) ? hpx::threads::thread_priority::high : hpx::threads::thread_priority::normal;

      herkTrailingDiagTile<backend>(trailing_matrix_priority, mat_a.read_sender(LocalTileIndex{k, i}),
                                    mat_a.readwrite_sender(LocalTileIndex{i, i}));

      for (SizeType j = i + 1; j < nrtile; ++j) {
        gemmTrailingMatrixTile<backend, T>(hpx::threads::thread_priority::normal,
                                           mat_a.read_sender(LocalTileIndex{k, i}),
                                           mat_a.read_sender(LocalTileIndex{k, j}),
                                           mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }
  }
}

template <Backend backend, Device device, class T>
void Cholesky<backend, device, T>::call_U(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a) {
  using namespace cholesky_u;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr = mat_a.distribution();
  const SizeType nrtile = mat_a.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> panels(n_workspaces, distr);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> panelsT(n_workspaces, distr);

  for (SizeType k = 0; k < nrtile; ++k) {
    const GlobalTileIndex kk_idx(k, k);
    const comm::Index2D kk_rank = distr.rankGlobalTile(kk_idx);

    // Factorization of diagonal tile and broadcast it along the k-th column
    if (kk_rank == this_rank) {
      potrfDiagTile<backend>(mat_a(kk_idx));
    }

    // If there is no trailing matrix
    const SizeType kt = k + 1;
    if (kt == nrtile)
      continue;

    auto& panel = panels.nextResource();
    auto& panelT = panelsT.nextResource();

    panel.setRangeStart({kt, kt});

    if (kk_rank.row() == this_rank.row()) {
      const LocalTileIndex diag_wp_idx{distr.localTileFromGlobalTile<Coord::Row>(k), 0};
      // Note:
      // panel shrinked to a single tile for temporarly storing and communicating the diagonal
      // tile used for the row update
      panelT.setRange({k, k}, {kt, kt});

      if (kk_rank.col() == this_rank.col())
        panelT.setTile(diag_wp_idx, mat_a.read(kk_idx));
      broadcast(executor_mpi, kk_rank.col(), panelT, mpi_row_task_chain);

      // ROW UPDATE
      for (SizeType j = distr.nextLocalTileFromGlobalTile<Coord::Col>(k + 1);
           j < distr.localNrTiles().cols(); ++j) {
        const LocalTileIndex local_idx(Coord::Col, j);
        const LocalTileIndex kj_idx(distr.localTileFromGlobalTile<Coord::Row>(k), j);

        trsmPanelTile<backend>(panelT.read_sender(diag_wp_idx), mat_a.readwrite_sender(kj_idx));

        panel.setTile(local_idx, mat_a.read(kj_idx));
      }

      // col panel has been used for temporary storage of diagonal panel for column update
      panelT.reset();
    }

    panelT.setRange({kt, kt}, indexFromOrigin(distr.nrTiles()));

    broadcast(executor_mpi, kk_rank.row(), panel, panelT, mpi_row_task_chain, mpi_col_task_chain);

    // TRAILING MATRIX
    for (SizeType it_idx = kt; it_idx < nrtile; ++it_idx) {
      const auto owner = distr.rankGlobalTile({it_idx, it_idx});

      if (owner.row() != this_rank.row())
        continue;

      const auto i = distr.localTileFromGlobalTile<Coord::Row>(it_idx);
      const auto trailing_matrix_priority =
          (i == k + 1) ? hpx::threads::thread_priority::high : hpx::threads::thread_priority::normal;
      if (this_rank.col() == owner.col()) {
        const auto j = distr.localTileFromGlobalTile<Coord::Col>(it_idx);

        herkTrailingDiagTile<backend>(trailing_matrix_priority, panel.read_sender({Coord::Col, j}),
                                      mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }

      for (SizeType j_idx = it_idx + 1; j_idx < nrtile; ++j_idx) {
        const auto owner_col = distr.rankGlobalTile<Coord::Col>(j_idx);

        if (owner_col != this_rank.col())
          continue;

        const auto j = distr.localTileFromGlobalTile<Coord::Col>(j_idx);

        gemmTrailingMatrixTile<backend, T>(hpx::threads::thread_priority::normal,
                                           panelT.read_sender({Coord::Row, i}),
                                           panel.read_sender({Coord::Col, j}),
                                           mat_a.readwrite_sender(LocalTileIndex{i, j}));
      }
    }

    panel.reset();
    panelT.reset();
  }
}
}
}
}
