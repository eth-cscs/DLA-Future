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

#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/util.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/init.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/backtransformation/api.h"
#include "dlaf/executors.h"
#include "dlaf/factorization/qr.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/layout_info.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
void set0(Matrix<T, Device::CPU>& mat) {
  dlaf::matrix::util::set(mat, [](auto&&) { return static_cast<T>(0.0); });
}

template <Device device, class T>
void copySingleTile(hpx::shared_future<matrix::Tile<const T, device>> in,
                    hpx::future<matrix::Tile<T, device>> out) {
  hpx::dataflow(dlaf::getCopyExecutor<Device::CPU, Device::CPU>(),
                matrix::unwrapExtendTiles(matrix::copy_o), in, out);
}

template <class Executor, Device device, class T>
void trmmPanel(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> t,
               hpx::future<matrix::Tile<T, device>> w) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::trmm_o), blas::Side::Right, blas::Uplo::Upper,
                blas::Op::ConjTrans, blas::Diag::NonUnit, T(1.0), t, w);
}

template <class Executor, Device device, class T>
void gemmUpdateW2(Executor&& ex, hpx::future<matrix::Tile<T, device>> w,
                  hpx::shared_future<matrix::Tile<const T, device>> c,
                  hpx::future<matrix::Tile<T, device>> w2) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::ConjTrans, blas::Op::NoTrans,
                T(1.0), w, c, T(1.0), std::move(w2));
}

template <class Executor, Device device, class T>
void gemmTrailingMatrix(Executor&& ex, hpx::shared_future<matrix::Tile<const T, device>> v,
                        hpx::shared_future<matrix::Tile<const T, device>> w2,
                        hpx::future<matrix::Tile<T, device>> c) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::gemm_o), blas::Op::NoTrans, blas::Op::NoTrans,
                T(-1.0), v, w2, T(1.0), std::move(c));
}

// Implementation based on:
// 1. Part of algorithm 6 "LAPACK Algorithm for the eigenvector back-transformation", page 15, PhD thesis
// "GPU Accelerated Implementations of a Generalized Eigenvalue Solver for Hermitian Matrices with
// Systematic Energy and Time to Solution Analysis" presented by Raffaele Solcà (2016)
// 2. Report "Gep + back-transformation", Alberto Invernizzi (2020)
// 3. Report "Reduction to band + back-transformation", Raffaele Solcà (2020)
// 4. G. H. Golub and C. F. Van Loan, Matrix Computations, chapter 5, The Johns Hopkins University Press
template <class T>
struct BackTransformation<Backend::MC, Device::CPU, T> {
  static void call_FC(Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
                      common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus);
  static void call_FC(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c,
                      Matrix<const T, Device::CPU>& mat_v,
                      common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus);
};

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(
    Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  using hpx::util::unwrapping;

  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType mv = mat_v.nrTiles().rows();
  const SizeType nv = mat_v.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();
  const SizeType nb = mat_v.blockSize().cols();
  const SizeType ms = mat_v.size().rows();
  const SizeType ns = mat_v.size().cols();

  // Matrix T
  int tottaus;
  if (ms < mb || ms == 0 || nv == 0)
    tottaus = 0;
  else
    tottaus = (ms / mb - 1) * mb + ms % mb;

  if (tottaus == 0)
    return;

  LocalElementSize sizeT(tottaus, tottaus);
  TileElementSize blockSizeT(mb, mb);
  Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);

  Matrix<T, Device::CPU> mat_vv({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w({mat_v.size().rows(), mb}, mat_v.blockSize());
  Matrix<T, Device::CPU> mat_w2({mb, mat_c.size().cols()}, mat_c.blockSize());

  SizeType last_mb;
  if (mat_v.blockSize().cols() == 1) {
    last_mb = 1;
  }
  else if (mat_v.size().cols()) {
    if (mat_v.size().cols() % mat_v.blockSize().cols() == 0)
      last_mb = mat_v.blockSize().cols();
    else
      last_mb = mat_v.size().cols() % mat_v.blockSize().cols();
  }

  // Specific for V matrix layout where last column of tiles is empty
  const SizeType last_reflector_idx = mat_v.nrTiles().cols() - 2;

  for (SizeType k = last_reflector_idx; k >= 0; --k) {
    bool is_last = (k == last_reflector_idx) ? true : false;

    for (SizeType i = k + 1; i < mat_v.nrTiles().rows(); ++i) {
      // Copy V panel into VV
      copySingleTile(mat_v.read(LocalTileIndex(i, k)), mat_vv(LocalTileIndex(i, 0)));

      // Setting VV
      auto setting_vv = unwrapping([=](auto&& tile) {
        if (i <= k) {
          tile::set0<T>(tile);
        }
        else if (i == k + 1) {
          tile::laset<T>(lapack::MatrixType::Upper, 0.f, 1.f, tile);
        }
      });
      hpx::dataflow(executor_hp, setting_vv, mat_vv(LocalTileIndex(i, 0)));

      // Copy VV into W
      copySingleTile(mat_vv.read(LocalTileIndex(i, 0)), mat_w(LocalTileIndex(i, 0)));
    }

    // Reset W2 to zero
    set0(mat_w2);

    // TODO: instead of using a full matrix, choose a "column" matrix. The problem is that last tile
    // should be square but may have different size.
    const GlobalTileIndex v_start{k + 1, k};
    auto taus_panel = taus[k];
    const SizeType taupan = (is_last) ? last_mb : mat_v.blockSize().cols();
    dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel,
                                                               mat_t(LocalTileIndex{k, k}));

    for (SizeType i = k + 1; i < m; ++i) {
      auto kk = LocalTileIndex{k, k};
      // WH = V T
      auto ik = LocalTileIndex{i, 0};
      hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t = mat_t.read(kk);
      auto tile_w = mat_w(ik);

      if (mat_t.tileSize(GlobalTileIndex{k, k}).rows() != mat_w.tileSize(GlobalTileIndex{i, 0}).cols()) {
        TileElementIndex origin(0, 0);
        TileElementSize size(mat_t.tileSize(GlobalTileIndex{k, k}).rows(),
                             mat_t.tileSize(GlobalTileIndex{k, k}).cols());
        const matrix::SubTileSpec spec({origin, size});
        auto subtile_w = splitTile(tile_w, spec);
        trmmPanel(executor_np, tile_t, std::move(subtile_w));
      }
      else
        trmmPanel(executor_np, tile_t, std::move(tile_w));
    }

    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{0, j};
      for (SizeType i = k + 1; i < m; ++i) {
        auto ik = LocalTileIndex{i, 0};
        auto ij = LocalTileIndex{i, j};
        // W2 = W C
        gemmUpdateW2(executor_np, mat_w(ik), mat_c.read(ij), mat_w2(kj));
      }
    }

    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{i, 0};
      for (SizeType j = 0; j < n; ++j) {
        auto kj = LocalTileIndex{0, j};
        auto ij = LocalTileIndex{i, j};
        // C = C - V W2
        gemmTrailingMatrix(executor_np, mat_vv.read(ik), mat_w2.read(kj), mat_c(ij));
      }
    }
  }
}

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(
    comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  using hpx::util::unwrapping;

  using hpx::execution::parallel_executor;
  using hpx::resource::get_thread_pool;
  using hpx::threads::thread_priority;

  auto executor_hp = dlaf::getHpExecutor<Backend::MC>();
  auto executor_np = dlaf::getNpExecutor<Backend::MC>();

  // Set up MPI
  auto executor_mpi = dlaf::getMPIExecutor<Backend::MC>();
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator());
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());

  const SizeType m = mat_c.nrTiles().rows();
  const SizeType n = mat_c.nrTiles().cols();
  const SizeType c_local_rows = mat_c.distribution().localNrTiles().rows();
  const SizeType c_local_cols = mat_c.distribution().localNrTiles().cols();
  const SizeType mv = mat_v.nrTiles().rows();
  const SizeType nv = mat_v.nrTiles().cols();
  const SizeType mb = mat_v.blockSize().rows();
  const SizeType nb = mat_v.blockSize().cols();
  const SizeType ms = mat_v.size().rows();
  const SizeType ns = mat_v.size().cols();

  auto dist_c = mat_c.distribution();
  auto dist_v = mat_v.distribution();

  const comm::Index2D this_rank = grid.rank();
  // Compute number of taus
  common::Pipeline<comm::CommunicatorGrid> serial_comm(grid);
  int tottaus;
  if (ms < mb || ms == 0 || ns == 0)
    tottaus = 0;
  else
    tottaus = (ms / mb - 1) * mb + ms % mb;

  if (tottaus == 0)
    return;

  LocalElementSize sizeT(tottaus, tottaus);
  TileElementSize blockSizeT(mb, mb);
  Matrix<T, Device::CPU> mat_t(sizeT, blockSizeT);

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsVV(n_workspaces,
                                                                         mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsW(n_workspaces,
                                                                        mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsW2(n_workspaces,
                                                                         mat_c.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsT(n_workspaces,
                                                                        mat_t.distribution());

  SizeType last_mb;
  if (mat_v.blockSize().cols() == 1) {
    last_mb = 1;
  }
  else {
    if (mat_v.size().cols() % mat_v.blockSize().cols() == 0)
      last_mb = mat_v.blockSize().cols();
    else
      last_mb = mat_v.size().cols() % mat_v.blockSize().cols();
  }

  // Specific for V matrix layout where last column of tiles is empty
  const SizeType last_reflector_idx = mat_v.nrTiles().cols() - 2;

  for (SizeType k = last_reflector_idx; k >= 0; --k) {
    bool is_last = (k == last_reflector_idx) ? true : false;

    auto& panelVV = panelsVV.nextResource();
    auto& panelW = panelsW.nextResource();
    auto& panelW2 = panelsW2.nextResource();
    auto& panelT = panelsT.nextResource();

    panelVV.setRangeStart({k + 1, k + 1});
    panelW.setRangeStart({k + 1, k + 1});
    panelW2.setRangeStart({0, 0});

    for (SizeType i_local = mat_c.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < c_local_rows; ++i_local) {
      auto i = mat_v.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);

      // Copy V panel into VV
      auto k_rank_col = dist_v.template rankGlobalTile<Coord::Col>(k);
      auto k_local_col = dist_v.template localTileFromGlobalTile<Coord::Col>(k);
      auto ik = LocalTileIndex{i_local, k_local_col};
      auto i0 = LocalTileIndex(i_local, 0);
      if (this_rank.col() == k_rank_col) {
        copySingleTile(mat_v.read(ik), panelVV(i0));
      }

      // Setting VV
      // Fixing elements of VV and copying them into WH
      auto setting_vv = unwrapping([=](auto&& tile) {
        if (i <= k) {
          tile::set0<T>(tile);
        }
        else if (i == k + 1) {
          tile::laset<T>(lapack::MatrixType::Upper, 0.f, 1.f, tile);
        }
      });
      if (this_rank.col() == k_rank_col) {
        hpx::dataflow(executor_hp, setting_vv, panelVV(i0));
      }

      // Copy VV into W
      if (this_rank.col() == k_rank_col) {
        copySingleTile(panelVV.read(i0), panelW(i0));
      }
    }

    // Reset W2 to zero (rw-access)
    for (const auto& idx : panelW2.iteratorLocal()) {
      panelW2(idx).then(unwrapping([](auto&& tile) {
        for (SizeType j = 0; j < tile.size().cols(); ++j) {
          for (SizeType i = 0; i < tile.size().rows(); ++i) {
            tile(TileElementIndex{i, j}) = static_cast<T>(0.0);
          }
        }
      }));
    }

    int taupan = (is_last) ? last_mb : mat_v.blockSize().cols();

    // Matrix T
    const GlobalTileIndex v_start{k + 1, k};
    auto taus_panel = taus[k];

    auto k_t_local_row = mat_t.distribution().template localTileFromGlobalTile<Coord::Row>(k);
    auto k_t_local_col = mat_t.distribution().template localTileFromGlobalTile<Coord::Col>(k);
    auto k_v_rank_row = mat_v.distribution().template rankGlobalTile<Coord::Row>(k);
    auto k_v_rank_col = mat_v.distribution().template rankGlobalTile<Coord::Col>(k);
    auto kk = LocalTileIndex(k_t_local_row, k_t_local_col);
    // Compute matrix T
    dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel,
                                                               mat_t(kk), serial_comm);

    panelT.setRange({k, k}, {k + 1, k + 1});

    const LocalTileIndex diag_wp_idx{Coord::Row, 0};
    if (this_rank.col() == k_v_rank_col) {
      panelT.setTile(diag_wp_idx, mat_t.read(kk));
    }
    // Broadcast T(k,k) row-wise
    broadcast(executor_mpi, k_v_rank_col, panelT, mpi_row_task_chain);

    for (SizeType i_local = mat_c.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < mat_c.distribution().localNrTiles().rows(); ++i_local) {
      auto i = mat_c.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);
      auto i_rank_row = mat_v.distribution().template rankGlobalTile<Coord::Row>(i);
      auto k_rank_col = mat_v.distribution().template rankGlobalTile<Coord::Col>(k);
      auto i_v = mat_v.distribution().template localTileFromGlobalTile<Coord::Row>(i);
      auto ik = LocalTileIndex{i_v, 0};

      // WH = V T
      if (this_rank.row() == i_rank_row && this_rank.col() == k_rank_col) {
        hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t = panelT.read(diag_wp_idx);

        if (mat_t.tileSize(GlobalTileIndex{k, k}).rows() !=
            mat_v.tileSize(GlobalTileIndex{i, k}).cols()) {
          panelW.setWidth(mat_t.tileSize(GlobalTileIndex{k, k}).cols());
          trmmPanel(executor_np, tile_t, std::move(panelW(ik)));
        }
        else {
          trmmPanel(executor_np, tile_t, std::move(panelW(ik)));
        }
      }

    }  // end loop on i_local row

    for (SizeType i_local = mat_c.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < mat_c.distribution().localNrTiles().rows(); ++i_local) {
      auto i = mat_c.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);
      auto i_rank_row = mat_v.distribution().template rankGlobalTile<Coord::Row>(i);
      auto k_rank_col = mat_v.distribution().template rankGlobalTile<Coord::Col>(k);
      auto i_v = mat_v.distribution().template localTileFromGlobalTile<Coord::Row>(i);
      auto iglo = mat_v.distribution().template globalTileFromLocalTile<Coord::Row>(i_v);
      auto ik = LocalTileIndex(i_v, 0);

      // Broadcast W(i,0) row-wise
      broadcast(executor_mpi, k_rank_col, panelW, mpi_row_task_chain);

      for (SizeType j_local = 0; j_local < c_local_cols; ++j_local) {
        // W2 = W C
        auto i_c = mat_c.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);
        auto j_c = mat_c.distribution().template globalTileFromLocalTile<Coord::Col>(j_local);
        auto i_c_rank_row = mat_c.distribution().template rankGlobalTile<Coord::Row>(i_c);
        auto j_c_rank_col = mat_c.distribution().template rankGlobalTile<Coord::Col>(j_c);

        auto kj = LocalTileIndex(0, j_local);
        auto ij = LocalTileIndex(i_local, j_local);

        if (this_rank.row() == i_c_rank_row && this_rank.col() == j_c_rank_col) {
          panelW.setWidth(mat_v.tileSize(GlobalTileIndex{k, k}).cols());
          gemmUpdateW2(executor_np, panelW(ik), mat_c.read(ij), std::move(panelW2(kj)));
        }

      }  // end loop on j_local (cols)
    }    // end loop on i_local (rows)

    for (SizeType j_local = 0; j_local < c_local_cols; ++j_local) {
      auto j_c = mat_c.distribution().template globalTileFromLocalTile<Coord::Col>(j_local);
      auto kj = LocalTileIndex(0, j_local);

      auto all_reduce_w2_func = unwrapping([=](auto&& tile, auto&& comm_wrapper) {
        dlaf::comm::sync::allReduceInPlace(comm_wrapper.ref().colCommunicator(), MPI_SUM,
                                           common::make_data(tile));
        return std::move(tile);
      });

      panelW2(kj) = hpx::dataflow(executor_hp, all_reduce_w2_func, panelW2(kj), serial_comm());
    }

    for (SizeType i_local = mat_c.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1);
         i_local < c_local_rows; ++i_local) {
      auto i = mat_c.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);
      auto i_v = mat_v.distribution().template localTileFromGlobalTile<Coord::Row>(i);
      auto i_rank_row = mat_v.distribution().template rankGlobalTile<Coord::Row>(i);
      auto k_rank_col = mat_v.distribution().template rankGlobalTile<Coord::Col>(k);
      auto ik = LocalTileIndex(i_v, 0);

      // Broadcast VV(i,0) row-wise
      broadcast(executor_mpi, k_rank_col, panelVV, mpi_row_task_chain);

      for (SizeType j_local = 0; j_local < c_local_cols; ++j_local) {
        auto i_c = mat_c.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);
        auto j_c = mat_c.distribution().template globalTileFromLocalTile<Coord::Col>(j_local);
        auto i_c_rank_row = mat_c.distribution().template rankGlobalTile<Coord::Row>(i_c);
        auto j_c_rank_col = mat_c.distribution().template rankGlobalTile<Coord::Col>(j_c);
        auto kj = LocalTileIndex{0, j_local};
        auto ij = LocalTileIndex(i_local, j_local);

        // C = C - V W2
        if (this_rank.row() == i_c_rank_row && this_rank.col() == j_c_rank_col) {
          gemmTrailingMatrix(executor_np, panelVV.read(ik), panelW2.read(kj), mat_c(ij));
        }

      }  // end of j_local loop on cols

    }  // end of i_local loop on rows

    panelVV.reset();
    panelW.reset();
    panelW2.reset();
    panelT.reset();
  }
}

/// ---- ETI
#define DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct BackTransformation<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, float)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, double)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_MC_ETI(extern, std::complex<double>)

}
}
}
