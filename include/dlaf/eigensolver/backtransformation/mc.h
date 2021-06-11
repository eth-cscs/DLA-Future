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
void set_zero(Matrix<T, Device::CPU>& mat) {
  dlaf::matrix::util::set(mat, [](auto&&) { return static_cast<T>(0.0); });
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
  constexpr auto Left = blas::Side::Left;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NonUnit = blas::Diag::NonUnit;

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
  comm::CommunicatorGrid comm_grid(MPI_COMM_WORLD, 1, 1, common::Ordering::ColumnMajor);
  common::Pipeline<comm::CommunicatorGrid> serial_comm(comm_grid);

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

    for (SizeType i = 0; i < mat_v.nrTiles().rows(); ++i) {
      // Copy V panel into VV
      hpx::dataflow(dlaf::getCopyExecutor<Device::CPU, Device::CPU>(),
                    matrix::unwrapExtendTiles(dlaf::matrix::copy_o), mat_v.read(LocalTileIndex(i, k)),
                    mat_vv(LocalTileIndex(i, 0)));

      // Setting VV
      auto setting_vv = unwrapping([=](auto&& tile) {
        if (i <= k) {
          lapack::laset(lapack::MatrixType::General, tile.size().rows(), tile.size().cols(), 0, 0,
                        tile.ptr(), tile.ld());
        }
        else if (i == k + 1) {
          lapack::laset(lapack::MatrixType::Upper, tile.size().rows(), tile.size().cols(), 0, 1,
                        tile.ptr(), tile.ld());
        }
      });
      hpx::dataflow(executor_hp, setting_vv, mat_vv(LocalTileIndex(i, 0)));

      // Copy VV into W
      hpx::dataflow(dlaf::getCopyExecutor<Device::CPU, Device::CPU>(),
                    matrix::unwrapExtendTiles(matrix::copy_o), mat_vv.read(LocalTileIndex(i, 0)),
                    mat_w(LocalTileIndex(i, 0)));
    }

    // Reset W2 to zero
    set_zero(mat_w2);

    const GlobalTileIndex v_start{k + 1, k};
    auto taus_panel = taus[k];
    int taupan = (is_last) ? last_mb : mat_v.blockSize().cols();
    dlaf::factorization::internal::computeTFactor<Backend::MC>(taupan, mat_v, v_start, taus_panel,
                                                               mat_t(LocalTileIndex{k, k}), serial_comm);

    for (SizeType i = k + 1; i < m; ++i) {
      auto kk = LocalTileIndex{k, k};
      // WH = V T
      auto ik = LocalTileIndex{i, 0};
      hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::trmm_o), Right, Upper, ConjTrans,
                    NonUnit, T(1.0), mat_t.read(kk), std::move(mat_w(ik)));
    }

    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{0, j};
      for (SizeType i = k + 1; i < m; ++i) {
        auto ik = LocalTileIndex{i, 0};
        auto ij = LocalTileIndex{i, j};
        // W2 = W C
        hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::gemm_o), ConjTrans, NoTrans, T(1.0),
                      mat_w.read(ik), mat_c.read(ij), T(1.0), std::move(mat_w2(kj)));
      }
    }

    for (SizeType i = k + 1; i < m; ++i) {
      auto ik = LocalTileIndex{i, 0};
      for (SizeType j = 0; j < n; ++j) {
        auto kj = LocalTileIndex{0, j};
        auto ij = LocalTileIndex{i, j};
        // C = C - V W2
        hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, T(-1.0),
                      mat_vv.read(ik), mat_w2.read(kj), T(1.0), std::move(mat_c(ij)));
      }
    }
  }
}

template <class T>
void BackTransformation<Backend::MC, Device::CPU, T>::call_FC(
    comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_c, Matrix<const T, Device::CPU>& mat_v,
    common::internal::vector<hpx::shared_future<common::internal::vector<T>>> taus) {
  constexpr auto Left = blas::Side::Left;
  constexpr auto Right = blas::Side::Right;
  constexpr auto Upper = blas::Uplo::Upper;
  constexpr auto Lower = blas::Uplo::Lower;
  constexpr auto NoTrans = blas::Op::NoTrans;
  constexpr auto ConjTrans = blas::Op::ConjTrans;
  constexpr auto NonUnit = blas::Diag::NonUnit;

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
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsVV(n_workspaces, mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsW(n_workspaces, mat_v.distribution());
  common::RoundRobin<matrix::Panel<Coord::Row, T, Device::CPU>> panelsW2(n_workspaces, mat_c.distribution());
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panelsT(n_workspaces, mat_t.distribution());
  
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
  //  Compute number of last_reflector_idx
  //  const SizeType last_reflector_idx = (mat_v.size().cols() < mat_v.size().rows()) ?
  //  mat_v.nrTiles().rows() - 2 : mat_v.nrTiles().cols() - 2;

  for (SizeType k = last_reflector_idx; k >= 0; --k) {
    bool is_last = (k == last_reflector_idx) ? true : false;

    auto& panelVV = panelsVV.nextResource();
    auto& panelW = panelsW.nextResource();
    auto& panelW2 = panelsW2.nextResource();
    auto& panelT = panelsT.nextResource();

    const LocalTileSize kkv_offset{
        mat_v.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k+1),
        mat_v.distribution().template nextLocalTileFromGlobalTile<Coord::Col>(k+1),
    };    

    const LocalTileSize kkc_offset{
        mat_c.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k+1),
        mat_c.distribution().template nextLocalTileFromGlobalTile<Coord::Col>(k+1),
    }; 

//    panelVV.setRangeStart(kkv_offset);
//    panelW.setRangeStart(kkv_offset);
//    panelW2.setRangeStart(kkc_offset);
    panelVV.setRange(kkv_offset, mat_v.distribution().template localNrTiles());
    panelW.setRange(kkv_offset, mat_v.distribution().template localNrTiles());
    panelW2.setRange(kkc_offset, mat_c.distribution().template localNrTiles());
    
    for (SizeType i_local = mat_c.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k+1); i_local < c_local_rows; ++i_local) {
      auto i = mat_v.distribution().template globalTileFromLocalTile<Coord::Row>(i_local);

      // Copy V panel into VV
      auto k_rank_col = dist_v.template rankGlobalTile<Coord::Col>(k);
      auto k_local_col = dist_v.template localTileFromGlobalTile<Coord::Col>(k);
      auto ik = LocalTileIndex{i_local, k_local_col};
      auto i0 = LocalTileIndex(i_local, 0);
      if (this_rank.col() == k_rank_col) {
        hpx::dataflow(dlaf::getCopyExecutor<Device::CPU, Device::CPU>(),
                      matrix::unwrapExtendTiles(dlaf::matrix::copy_o), mat_v.read(ik), panelVV(i0));
      }

      // Setting VV
      // Fixing elements of VV and copying them into WH
      auto setting_vv = unwrapping([=](auto&& tile) {
        if (i <= k) {
          lapack::laset(lapack::MatrixType::General, tile.size().rows(), tile.size().cols(), 0, 0,
                        tile.ptr(), tile.ld());
        }
        else if (i == k + 1) {
          lapack::laset(lapack::MatrixType::Upper, tile.size().rows(), tile.size().cols(), 0, 1,
                        tile.ptr(), tile.ld());
        }
      });
      if (this_rank.col() == k_rank_col) {
	hpx::dataflow(executor_hp, setting_vv, panelVV(i0));
      }

      // Copy VV into W
      if (this_rank.col() == k_rank_col) {
	hpx::dataflow(dlaf::getCopyExecutor<Device::CPU, Device::CPU>(), matrix::unwrapExtendTiles(matrix::copy_o), panelVV.read(i0), panelW(i0));
      }
    }
    
    // Reset W2 to zero
     // rw-access
    for (const auto& idx : panelW2.iterator()) {
      panelW2(idx).then(unwrapping([idx](auto&& tile) {
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
    std::cout << "mat T " << mat_t(kk).get()({0,0}) << " rank: " << this_rank << " k: " << k << std::endl;

    const LocalTileSize kkt_offset{
        mat_t.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k),
        mat_t.distribution().template nextLocalTileFromGlobalTile<Coord::Col>(k),
    };    
    const LocalTileSize att_offset{
        mat_t.distribution().template nextLocalTileFromGlobalTile<Coord::Row>(k + 1),
        mat_t.distribution().template nextLocalTileFromGlobalTile<Coord::Col>(k + 1),
    };
    panelT.setRange(kkt_offset, att_offset);    
    const LocalTileIndex diag_wp_idx{0, kkt_offset.cols()};
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
	hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::trmm_o), Right, Upper, ConjTrans, NonUnit, T(1.0), panelT.read(diag_wp_idx), std::move(panelW(ik)));

	std::cout << "TRMM: i " << i << " k " << k << " mat t " << panelT.read(diag_wp_idx).get()({0,0}) << " mat w " << panelW.read(ik).get()({0,0}) << " rank " << this_rank << std::endl;
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
	  hpx::dataflow(executor_hp, matrix::unwrapExtendTiles(tile::gemm_o), ConjTrans, NoTrans, T(1.0), panelW.read(ik), mat_c.read(ij), T(0.0), std::move(panelW2(kj)));

	  std::cout << "GEMM #1: k " << k << " i " << i_c << " j " << j_c << " W " << panelW.read(ik).get()({0,0}) << " mat_c " << mat_c.read(ij).get()({0,0})  << " W2 " << panelW2.read(kj).get()({0,0}) << std::endl; 
        }

      }  // end loop on j_local (cols)
    }    // end loop on i_local (rows)

    for (SizeType j_local = 0; j_local < c_local_cols; ++j_local) {
      auto j_c = mat_c.distribution().template globalTileFromLocalTile<Coord::Col>(j_local);
      auto kj = LocalTileIndex(0, j_local);
      
      auto all_reduce_w2_func = unwrapping([=](auto&& tile, auto&& comm_wrapper) {
	  dlaf::comm::sync::allReduceInPlace(comm_wrapper.ref().colCommunicator(), MPI_SUM, common::make_data(tile));
	  return std::move(tile);
	});
      
      std::cout << "REDUCE BEFORE: k " << k << " j " << j_c << " W2 " << panelW2.read(kj).get()({0,0}) << std::endl; 
      panelW2(kj) = hpx::dataflow(executor_hp, all_reduce_w2_func, panelW2(kj), serial_comm());
      std::cout << "REDUCE AFTER: k " << k << " j " << j_c << " W2 " << panelW2.read(kj).get()({0,0}) << std::endl; 
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
          hpx::dataflow(executor_np, matrix::unwrapExtendTiles(tile::gemm_o), NoTrans, NoTrans, T(-1.0), panelVV.read(ik), panelW2.read(kj), T(1.0), std::move(mat_c(ij)));

	  std::cout << "GEMM #2: k " << k << " i " << i_c << " j " << j_c << " W " << panelVV.read(ik).get()({0,0}) << " W2 " << panelW2.read(kj).get()({0,0}) << " C " << mat_c.read(ij).get()({0,0}) << std::endl; 
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
