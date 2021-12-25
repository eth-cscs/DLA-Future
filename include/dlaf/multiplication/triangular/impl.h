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

#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/thread.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/executor.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/multiplication/triangular/api.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace multiplication {
namespace internal {

namespace triangular_lln {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Diag diag, T alpha, InSender&& in_tile,
                    OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, T alpha, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace triangular_llt {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, blas::Op op, T alpha,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(op, blas::Op::NoTrans, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace triangular_lun {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Diag diag, T alpha, InSender&& in_tile,
                    OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, T alpha, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace triangular_lut {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Upper, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, blas::Op op, T alpha,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(op, blas::Op::NoTrans, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace triangular_rln {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Diag diag, T alpha, InSender&& in_tile,
                    OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, T alpha, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace triangular_rlt {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Lower, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, blas::Op op, T alpha,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, op, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace triangular_run {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Diag diag, T alpha, InSender&& in_tile,
                    OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, T alpha, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

namespace triangular_rut {
template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(hpx::threads::thread_priority priority, blas::Op op, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  dlaf::internal::whenAllLift(blas::Side::Right, blas::Uplo::Upper, op, diag, alpha,
                              std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(hpx::threads::thread_priority priority, blas::Op op, T alpha,
                            ASender&& a_tile, BSender&& b_tile, CSender&& c_tile) {
  dlaf::internal::whenAllLift(blas::Op::NoTrans, op, alpha, std::forward<ASender>(a_tile),
                              std::forward<BSender>(b_tile), T(1.0), std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lln;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k >= 0; --k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k + 1; i < m; ++i) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha,
                                        mat_a.read_sender(LocalTileIndex{i, k}), mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_llt;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k - 1; i >= 0; --i) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, op, alpha,
                                        mat_a.read_sender(LocalTileIndex{k, i}), mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_lun;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < m; ++k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = 0; i < k; ++i) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha,
                                        mat_a.read_sender(LocalTileIndex{i, k}), mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lut;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = m - 1; k >= 0; --k) {
    for (SizeType j = n - 1; j >= 0; --j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k + 1; i < m; ++i) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, op, alpha,
                                        mat_a.read_sender(LocalTileIndex{k, i}), mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_rln;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k - 1; j >= 0; --j) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{k, j}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rlt;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k >= 0; --k) {
    for (SizeType i = m - 1; i >= 0; --i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k + 1; j < n; ++j) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, op, alpha, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{j, k}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                                              Matrix<T, device>& mat_b) {
  using namespace triangular_run;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = n - 1; k >= 0; --k) {
    for (SizeType i = m - 1; i >= 0; --i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k + 1; j < n; ++j) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{k, j}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUT(blas::Op op, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rut;
  using hpx::threads::thread_priority;

  SizeType m = mat_b.nrTiles().rows();
  SizeType n = mat_b.nrTiles().cols();

  for (SizeType k = 0; k < n; ++k) {
    for (SizeType i = 0; i < m; ++i) {
      auto ik = LocalTileIndex{i, k};

      for (SizeType j = k - 1; j >= 0; --j) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, op, alpha, mat_b.read_sender(ik),
                                        mat_a.read_sender(LocalTileIndex{j, k}),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, op, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(ik));
    }
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lln;
  using hpx::threads::thread_priority;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = mat_a.nrTiles().rows() - 1; k >= 0; --k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1), 0};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    a_panel.setRangeStart(kk);
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setWidth(mat_a.tileSize(kk).rows());
      b_panel.setHeight(mat_a.tileSize(kk).cols());
    }

    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.row(); i_local < distr_a.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.col());

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }
    }
    broadcast(executor_mpi, kk_rank.col(), a_panel, mpi_row_task_chain);

    for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
      if (kk_rank.row() == this_rank.row()) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
        const LocalTileIndex kk_panel(Coord::Row, k_local_row);
        const LocalTileIndex kj(k_local_row, j_local);
        const LocalTileIndex kj_panel(Coord::Col, j_local);

        b_panel.setTile(kj_panel, mat_b.read(kj));
        trmmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(kj));
      }
    }

    broadcast(executor_mpi, kk_rank.row(), b_panel, mpi_col_task_chain);

    for (SizeType i_local = bt_offset.row(); i_local < distr_a.localNrTiles().rows(); ++i_local) {
      const LocalTileIndex ik_panel(Coord::Row, i_local);
      // Update trailing matrix
      for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex ij(i_local, j_local);
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha, a_panel.read_sender(ik_panel),
                                        b_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }

    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_LUN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_lun;
  using hpx::threads::thread_priority;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = 0; k < mat_a.nrTiles().rows(); ++k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k + 1),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k), 0};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setWidth(mat_a.tileSize(kk).rows());
      b_panel.setHeight(mat_a.tileSize(kk).cols());
    }

    if (kk_rank.col() == this_rank.col()) {
      for (SizeType i_local = kk_offset.row() - 1; i_local >= 0; --i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ik(i_local, kk_offset.col());

        a_panel.setTile(ik_panel, mat_a.read(ik));
      }
    }
    broadcast(executor_mpi, kk_rank.col(), a_panel, mpi_row_task_chain);

    for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
      if (kk_rank.row() == this_rank.row()) {
        auto k_local_row = distr_b.localTileFromGlobalTile<Coord::Row>(k);
        const LocalTileIndex kk_panel(Coord::Row, k_local_row);
        const LocalTileIndex kj(k_local_row, j_local);
        const LocalTileIndex kj_panel(Coord::Col, j_local);

        b_panel.setTile(kj_panel, mat_b.read(kj));
        trmmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(kj));
      }
    }
    broadcast(executor_mpi, kk_rank.row(), b_panel, mpi_col_task_chain);

    for (SizeType i_local = bt_offset.row() - 1; i_local >= 0; --i_local) {
      // Choose queue priority
      const LocalTileIndex ik_panel(Coord::Row, i_local);
      // Update trailing matrix
      for (SizeType j_local = 0; j_local < distr_b.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex ij(i_local, j_local);
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha, a_panel.read_sender(ik_panel),
                                        b_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }

    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_rln;
  using hpx::threads::thread_priority;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = 0; k < mat_a.nrTiles().cols(); ++k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k + 1),
    };

    const LocalTileIndex bt_offset{0, distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k)};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setHeight(mat_a.tileSize(kk).cols());
      b_panel.setWidth(mat_a.tileSize(kk).rows());
    }

    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = kk_offset.col() - 1; j_local >= 0; --j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex kj(kk_offset.row(), j_local);

        a_panel.setTile(kj_panel, mat_a.read(kj));
      }
    }
    broadcast(executor_mpi, kk_rank.row(), a_panel, mpi_col_task_chain);

    for (SizeType i_local = 0; i_local < distr_b.localNrTiles().rows(); ++i_local) {
      if (kk_rank.col() == this_rank.col()) {
        auto k_local_col = distr_b.localTileFromGlobalTile<Coord::Col>(k);
        const LocalTileIndex kk_panel(Coord::Col, k_local_col);
        const LocalTileIndex ik(i_local, k_local_col);
        const LocalTileIndex ik_panel(Coord::Row, i_local);

        b_panel.setTile(ik_panel, mat_b.read(ik));
        trmmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(ik));
      }
    }

    broadcast(executor_mpi, kk_rank.col(), b_panel, mpi_row_task_chain);

    for (SizeType j_local = bt_offset.col() - 1; j_local >= 0; --j_local) {
      // Choose queue priority
      const LocalTileIndex kj_panel(Coord::Col, j_local);
      // Update trailing matrix
      for (SizeType i_local = 0; i_local < distr_b.localNrTiles().rows(); ++i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ij(i_local, j_local);
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha, b_panel.read_sender(ik_panel),
                                        a_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }

    a_panel.reset();
    b_panel.reset();
  }
}

template <Backend backend, Device device, class T>
void Triangular<backend, device, T>::call_RUN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                                              Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b) {
  using namespace triangular_run;
  using hpx::threads::thread_priority;

  auto executor_mpi = dlaf::getMPIExecutor<backend>();

  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const comm::Index2D this_rank = grid.rank();

  const matrix::Distribution& distr_a = mat_a.distribution();
  const matrix::Distribution& distr_b = mat_b.distribution();

  if (mat_b.size().isEmpty())
    return;

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Row, T, device>> a_panels(n_workspaces, distr_a);
  common::RoundRobin<matrix::Panel<Coord::Col, T, device>> b_panels(n_workspaces, distr_b);

  for (SizeType k = mat_a.nrTiles().cols() - 1; k >= 0; --k) {
    const GlobalTileIndex kk(k, k);
    auto kk_rank = distr_a.rankGlobalTile(kk);

    const LocalTileIndex kk_offset{
        distr_a.nextLocalTileFromGlobalTile<Coord::Row>(k),
        distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k),
    };

    const LocalTileIndex bt_offset{0, distr_a.nextLocalTileFromGlobalTile<Coord::Col>(k + 1)};

    auto& a_panel = a_panels.nextResource();
    auto& b_panel = b_panels.nextResource();
    a_panel.setRangeStart(kk);
    if (k == mat_a.nrTiles().cols() - 1) {
      a_panel.setHeight(mat_a.tileSize(kk).cols());
      b_panel.setWidth(mat_a.tileSize(kk).rows());
    }

    if (kk_rank.row() == this_rank.row()) {
      for (SizeType j_local = kk_offset.col(); j_local < distr_a.localNrTiles().cols(); ++j_local) {
        const LocalTileIndex kj_panel(Coord::Col, j_local);
        const LocalTileIndex kj(kk_offset.row(), j_local);

        a_panel.setTile(kj_panel, mat_a.read(kj));
      }
    }
    broadcast(executor_mpi, kk_rank.row(), a_panel, mpi_col_task_chain);

    for (SizeType i_local = distr_b.localNrTiles().rows() - 1; i_local >= 0; --i_local) {
      if (kk_rank.col() == this_rank.col()) {
        auto k_local_col = distr_b.localTileFromGlobalTile<Coord::Col>(k);
        const LocalTileIndex kk_panel(Coord::Col, k_local_col);
        const LocalTileIndex ik(i_local, k_local_col);
        const LocalTileIndex ik_panel(Coord::Row, i_local);

        b_panel.setTile(ik_panel, mat_b.read(ik));
        trmmBPanelTile<backend>(thread_priority::high, diag, alpha, a_panel.read_sender(kk_panel),
                                mat_b.readwrite_sender(ik));
      }
    }
    broadcast(executor_mpi, kk_rank.col(), b_panel, mpi_row_task_chain);

    for (SizeType j_local = bt_offset.col(); j_local < distr_a.localNrTiles().cols(); ++j_local) {
      // Choose queue priority
      const LocalTileIndex kj_panel(Coord::Col, j_local);
      // Update trailing matrix
      for (SizeType i_local = distr_b.localNrTiles().rows() - 1; i_local >= 0; --i_local) {
        const LocalTileIndex ik_panel(Coord::Row, i_local);
        const LocalTileIndex ij(i_local, j_local);
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha, b_panel.read_sender(ik_panel),
                                        a_panel.read_sender(kj_panel), mat_b.readwrite_sender(ij));
      }
    }

    a_panel.reset();
    b_panel.reset();
  }
}

}
}
}
