//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <sstream>
#include <utility>
#include <vector>

#include <pika/barrier.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/data.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/common/vector.h>
#include <dlaf/communication/broadcast_panel.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/functions_sync.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/communication/kernels/reduce.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/eigensolver/reduction_to_band/api.h>
#include <dlaf/eigensolver/reduction_to_band/common.h>
#include <dlaf/factorization/qr.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/matrix/views.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/traits.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

namespace red2band {

namespace local {
template <Backend B, Device D, class T>
void hemmComputeX(matrix::Panel<Coord::Col, T, D>& x, const matrix::SubMatrixView& view,
                  matrix::Matrix<const T, D>& a, matrix::Panel<Coord::Col, const T, D>& w) {
  namespace ex = pika::execution::experimental;

  using pika::execution::thread_priority;

  const auto dist = a.distribution();

  // Note:
  // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
  // "initialized" during computation, so they should not contribute with any spurious value to the final
  // result.
  matrix::util::set0<B>(thread_priority::high, x);

  const LocalTileIndex at_offset = view.begin();

  for (SizeType i = at_offset.row(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = i + 1;
    for (SizeType j = limit - 1; j >= at_offset.col(); --j) {
      const LocalTileIndex ij{i, j};

      const bool is_diagonal_tile = (ij.row() == ij.col());

      const auto& tile_a = splitTile(a.read(ij), view(ij));

      if (is_diagonal_tile) {
        hemmDiag<B>(thread_priority::high, tile_a, w.read(ij), x.readwrite(ij));
      }
      else {
        // Note:
        // Because A is hermitian and just the lower part contains the data, for each a(ij) not
        // on the diagonal, two computations are done:
        // - using a(ij) in its position;
        // - using a(ij) in its "transposed" position (applying the ConjTrans to its data)

        {
          const LocalTileIndex index_x(Coord::Row, ij.row());
          const LocalTileIndex index_w(Coord::Row, ij.col());
          hemmOffDiag<B>(thread_priority::high, blas::Op::NoTrans, tile_a, w.read(index_w),
                         x.readwrite(index_x));
        }

        {
          const LocalTileIndex index_pretended = transposed(ij);
          const LocalTileIndex index_x(Coord::Row, index_pretended.row());
          const LocalTileIndex index_w(Coord::Row, index_pretended.col());
          hemmOffDiag<B>(thread_priority::high, blas::Op::ConjTrans, tile_a, w.read(index_w),
                         x.readwrite(index_x));
        }
      }
    }
  }
}

template <Backend B, Device D, class T>
void her2kUpdateTrailingMatrix(const matrix::SubMatrixView& view, matrix::Matrix<T, D>& a,
                               matrix::Panel<Coord::Col, const T, D>& x,
                               matrix::Panel<Coord::Col, const T, D>& v) {
  static_assert(std::is_signed_v<BaseType<T>>, "alpha in computations requires to be -1");

  using pika::execution::thread_priority;

  const auto dist = a.distribution();

  const LocalTileIndex at_start = view.begin();

  for (SizeType i = at_start.row(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = at_start.col(); j < limit; ++j) {
      const LocalTileIndex ij_local{i, j};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      auto getSubA = [&a, &view, ij_local]() {
        return splitTile(a.readwrite(ij_local), view(ij_local));
      };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority = (j == at_start.col()) ? thread_priority::high : thread_priority::normal;

      if (is_diagonal_tile) {
        her2kDiag<B>(priority, v.read(ij_local), x.read(ij_local), getSubA());
      }
      else {
        // A -= X . V*
        her2kOffDiag<B>(priority, x.read(ij_local), v.read(transposed(ij_local)), getSubA());

        // A -= V . X*
        her2kOffDiag<B>(priority, v.read(ij_local), x.read(transposed(ij_local)), getSubA());
      }
    }
  }
}

}

namespace distributed {
template <Device D, class T>
T computeReflector(const bool has_head, comm::Communicator& communicator,
                   const std::vector<matrix::Tile<T, D>>& panel, SizeType j) {
  std::array<T, 2> x0_and_squares = computeX0AndSquares(has_head, panel, j);

  // Note:
  // This is an optimization for grouping two separate low bandwidth communications, respectively
  // bcast(x0) and reduce(norm), where the latency was degrading performances.
  //
  // In particular this allReduce allows to:
  // - bcast x0, since for all ranks is 0 and just the root rank has the real value;
  // - allReduce squares for the norm computation.
  //
  // Moreover, by all-reducing squares and broadcasting x0, all ranks have all the information to
  // update locally the reflectors (section they have). This is more efficient than computing params
  // (e.g. norm, y, tau) just on the root rank and then having to broadcast them (i.e. additional
  // communication).
  comm::sync::allReduceInPlace(communicator, MPI_SUM,
                               common::make_data(x0_and_squares.data(),
                                                 to_SizeType(x0_and_squares.size())));

  auto tau = computeReflectorAndTau(has_head, panel, j, std::move(x0_and_squares));

  return tau;
}

template <class MatrixLikeA, class MatrixLikeTaus, class TriggerSender, class CommSender>
void computePanelReflectors(TriggerSender&& trigger, comm::IndexT_MPI rank_v0,
                            CommSender&& mpi_col_chain_panel, MatrixLikeA& mat_a,
                            MatrixLikeTaus& mat_taus, SizeType j_sub,
                            const matrix::SubPanelView& panel_view) {
  static Device constexpr D = MatrixLikeA::device;
  using T = typename MatrixLikeA::ElementType;
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<matrix::ReadWriteTileSender<T, D>> panel_tiles;
  panel_tiles.reserve(to_sizet(std::distance(panel_view.iteratorLocal().begin(),
                                             panel_view.iteratorLocal().end())));
  for (const auto& i : panel_view.iteratorLocal()) {
    const matrix::SubTileSpec& spec = panel_view(i);
    panel_tiles.emplace_back(matrix::splitTile(mat_a.readwrite(i), spec));
  }

  const std::size_t nthreads = getReductionToBandPanelNWorkers();
  auto s =
      ex::when_all(ex::just(std::make_unique<pika::barrier<>>(nthreads),
                            std::vector<common::internal::vector<T>>{}),  // w (internally required)
                   mat_taus.readwrite(GlobalTileIndex(j_sub, 0)),
                   ex::when_all_vector(std::move(panel_tiles)),
                   std::forward<CommSender>(mpi_col_chain_panel), std::forward<TriggerSender>(trigger)) |
      ex::transfer(di::getBackendScheduler<Backend::MC>(pika::execution::thread_priority::high)) |
      ex::bulk(nthreads, [nthreads, rank_v0,
                          cols = panel_view.cols()](const std::size_t index, auto& barrier_ptr, auto& w,
                                                    auto& taus, auto& tiles, auto&& pcomm) {
        const bool rankHasHead = rank_v0 == pcomm.get().rank();

        const auto barrier_busy_wait = getReductionToBandBarrierBusyWait();
        const std::size_t batch_size = util::ceilDiv(tiles.size(), nthreads);
        const std::size_t begin = index * batch_size;
        const std::size_t end = std::min(index * batch_size + batch_size, tiles.size());
        const SizeType nrefls = taus.size().rows();

        if (index == 0) {
          w.resize(nthreads);
        }

        for (SizeType j = 0; j < nrefls; ++j) {
          // STEP1: compute tau and reflector (single-thread)
          if (index == 0) {
            const bool has_head = rankHasHead;
            taus({j, 0}) = computeReflector(has_head, pcomm.get(), tiles, j);
          }
          barrier_ptr->arrive_and_wait(barrier_busy_wait);

          // STEP2a: compute w (multi-threaded)
          const SizeType pt_cols = cols - (j + 1);
          if (pt_cols == 0)
            break;

          const bool has_head = rankHasHead && (index == 0);

          w[index] = common::internal::vector<T>(pt_cols, 0);
          computeWTrailingPanel(has_head, tiles, w[index], j, pt_cols, begin, end);
          barrier_ptr->arrive_and_wait(barrier_busy_wait);

          // STEP2b: reduce w results (single-threaded)
          if (index == 0) {
            dlaf::eigensolver::internal::reduceColumnVectors(w);
            comm::sync::allReduceInPlace(pcomm.get(), MPI_SUM, common::make_data(w[0].data(), pt_cols));
          }
          barrier_ptr->arrive_and_wait(barrier_busy_wait);

          // STEP3: update trailing panel (multi-threaded)
          updateTrailingPanel(has_head, tiles, j, w[0], taus({j, 0}), begin, end);
          barrier_ptr->arrive_and_wait(barrier_busy_wait);
        }
      });
  ex::start_detached(std::move(s));
}

template <Backend B, Device D, class T>
void hemmComputeX(comm::IndexT_MPI reducer_col, matrix::Panel<Coord::Col, T, D>& x,
                  matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& xt,
                  const matrix::SubMatrixView& view, matrix::Matrix<const T, D>& a,
                  matrix::Panel<Coord::Col, const T, D>& w,
                  matrix::Panel<Coord::Row, const T, D, matrix::StoreTransposed::Yes>& wt,
                  comm::CommunicatorPipeline<comm::CommunicatorType::Row>& mpi_row_chain,
                  comm::CommunicatorPipeline<comm::CommunicatorType::Col>& mpi_col_chain) {
  namespace ex = pika::execution::experimental;

  using pika::execution::thread_priority;

  const auto dist = a.distribution();
  const auto rank = dist.rankIndex();

  // Note:
  // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
  // "initialized" during computation, so they should not contribute with any spurious value to the final
  // result.
  matrix::util::set0<B>(thread_priority::high, x);
  matrix::util::set0<B>(thread_priority::high, xt);

  const LocalTileIndex at_offset = view.begin();

  for (SizeType i = at_offset.row(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = limit - 1; j >= at_offset.col(); --j) {
      const LocalTileIndex ij_local{i, j};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      auto tile_a = splitTile(a.read(ij), view(ij_local));

      if (is_diagonal_tile) {
        hemmDiag<B>(thread_priority::high, std::move(tile_a), w.read(ij_local), x.readwrite(ij_local));
      }
      else {
        // Note:
        // Since it is not a diagonal tile, otherwise it would have been managed in the previous
        // branch, the second operand is not available in W but it is accessible through the
        // support panel Wt.
        // However, since we are still computing the "straight" part, the result can be stored
        // in the "local" panel X.
        hemmOffDiag<B>(thread_priority::high, blas::Op::NoTrans, tile_a, wt.read(ij_local),
                       x.readwrite(ij_local));

        // Note:
        // Here we are considering the hermitian part of A, so coordinates have to be "mirrored".
        // So, first step is identifying the mirrored cell coordinate, i.e. swap row/col, together
        // with realizing if the new coord lays on an owned row or not.
        // If yes, the result can be stored in the X, otherwise Xt support panel will be used.
        // For what concerns the second operand, it can be found for sure in W. In fact, the
        // multiplication requires matching col(A) == row(W), but since coordinates are mirrored,
        // we are matching row(A) == row(W), so it is local by construction.
        const auto owner = dist.template rankGlobalTile<Coord::Row>(ij.col());

        const LocalTileIndex index_x{dist.template localTileFromGlobalTile<Coord::Row>(ij.col()), 0};
        const LocalTileIndex index_xt{0, ij_local.col()};

        auto tile_x = (dist.rankIndex().row() == owner) ? x.readwrite(index_x) : xt.readwrite(index_xt);

        hemmOffDiag<B>(thread_priority::high, blas::Op::ConjTrans, std::move(tile_a), w.read(ij_local),
                       std::move(tile_x));
      }
    }
  }

  // Note:
  // At this point, partial results of X and Xt are available in the panels, and they have to be reduced,
  // both row-wise and col-wise.
  // The final X result will be available just on Ai panel column.

  // Note:
  // The first step in reducing partial results distributed over X and Xt, it is to reduce the row
  // panel Xt col-wise, by collecting all Xt results on the rank which can "mirror" the result on its
  // rows (i.e. diagonal). So, for each tile of the row panel, select who is the "diagonal" rank that can
  // mirror and reduce on it.
  if (mpi_col_chain.size() > 1) {
    for (const auto& index_xt : xt.iteratorLocal()) {
      const auto index_k = dist.template globalTileFromLocalTile<Coord::Col>(index_xt.col());
      const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(index_k);

      if (rank_owner_row == rank.row()) {
        // Note:
        // Since it is the owner, it has to perform the "mirroring" of the results from columns to
        // rows.
        //
        // Moreover, it reduces in place because the owner of the diagonal stores the partial result
        // directly in x (without using xt)
        const auto i = dist.template localTileFromGlobalTile<Coord::Row>(index_k);
        ex::start_detached(comm::schedule_reduce_recv_in_place(mpi_col_chain.exclusive(), MPI_SUM,
                                                               x.readwrite({i, 0})));
      }
      else {
        ex::start_detached(comm::schedule_reduce_send(mpi_col_chain.exclusive(), rank_owner_row, MPI_SUM,
                                                      xt.read(index_xt)));
      }
    }
  }

  // Note:
  // At this point partial results are all collected in X (Xt has been embedded in previous step),
  // so the last step needed is to reduce these last partial results in the final results.
  // The result is needed just on the column with reflectors.
  if (mpi_row_chain.size() > 1) {
    for (const auto& index_x : x.iteratorLocal()) {
      if (reducer_col == rank.col())
        ex::start_detached(comm::schedule_reduce_recv_in_place(mpi_row_chain.exclusive(), MPI_SUM,
                                                               x.readwrite(index_x)));
      else
        ex::start_detached(comm::schedule_reduce_send(mpi_row_chain.exclusive(), reducer_col, MPI_SUM,
                                                      x.read(index_x)));
    }
  }
}

template <Backend B, Device D, class T>
void her2kUpdateTrailingMatrix(const matrix::SubMatrixView& view, Matrix<T, D>& a,
                               matrix::Panel<Coord::Col, const T, D>& x,
                               matrix::Panel<Coord::Row, const T, D, matrix::StoreTransposed::Yes>& vt,
                               matrix::Panel<Coord::Col, const T, D>& v,
                               matrix::Panel<Coord::Row, const T, D, matrix::StoreTransposed::Yes>& xt) {
  static_assert(std::is_signed_v<BaseType<T>>, "alpha in computations requires to be -1");

  using pika::execution::thread_priority;

  const auto dist = a.distribution();

  const LocalTileIndex at_start = view.begin();

  for (SizeType i = at_start.row(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = at_start.col(); j < limit; ++j) {
      const LocalTileIndex ij_local{i, j};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      auto getSubA = [&a, &view, ij_local]() {
        return splitTile(a.readwrite(ij_local), view(ij_local));
      };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority = (j == at_start.col()) ? thread_priority::high : thread_priority::normal;

      if (is_diagonal_tile) {
        her2kDiag<B>(priority, v.read(ij_local), x.read(ij_local), getSubA());
      }
      else {
        // A -= X . V*
        her2kOffDiag<B>(priority, x.read(ij_local), vt.read(ij_local), getSubA());

        // A -= V . X*
        her2kOffDiag<B>(priority, v.read(ij_local), xt.read(ij_local), getSubA());
      }
    }
  }
}
}

template <Backend B, Device D, class T>
struct ComputePanelHelper;

template <class T>
struct ComputePanelHelper<Backend::MC, Device::CPU, T> {
  ComputePanelHelper(const std::size_t, matrix::Distribution) {}

  void call(Matrix<T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_taus, const SizeType j_sub,
            const matrix::SubPanelView& panel_view) {
    using red2band::local::computePanelReflectors;
    computePanelReflectors(mat_a, mat_taus, j_sub, panel_view);
  }

  template <Device D, class CommSender, class TriggerSender>
  void call(TriggerSender&& trigger, comm::IndexT_MPI rank_v0, CommSender&& mpi_col_chain_panel,
            Matrix<T, D>& mat_a, Matrix<T, Device::CPU>& mat_taus, const SizeType j_sub,
            const matrix::SubPanelView& panel_view) {
    using red2band::distributed::computePanelReflectors;
    computePanelReflectors(std::forward<TriggerSender>(trigger), rank_v0,
                           std::forward<CommSender>(mpi_col_chain_panel), mat_a, mat_taus, j_sub,
                           panel_view);
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct ComputePanelHelper<Backend::GPU, Device::GPU, T> {
  ComputePanelHelper(const std::size_t n_workspaces, matrix::Distribution dist_a)
      : panels_v(n_workspaces, dist_a) {}

  void call(Matrix<T, Device::GPU>& mat_a, Matrix<T, Device::CPU>& mat_taus, const SizeType j_sub,
            const matrix::SubPanelView& panel_view) {
    using red2band::local::computePanelReflectors;

    namespace ex = pika::execution::experimental;

    // Note:
    // - copy panel_view from GPU to CPU
    // - computePanelReflectors on CPU (on a matrix like, with just a panel)
    // - copy back matrix "panel" from CPU to GPU

    auto& v = panels_v.nextResource();

    copyToCPU(panel_view, mat_a, v);
    computePanelReflectors(v, mat_taus, j_sub, panel_view);
    copyFromCPU(panel_view, v, mat_a);
  }

  template <Device D, class CommSender, class TriggerSender>
  void call(TriggerSender&& trigger, comm::IndexT_MPI rank_v0, CommSender&& mpi_col_chain_panel,
            Matrix<T, D>& mat_a, Matrix<T, Device::CPU>& mat_taus, SizeType j_sub,
            const matrix::SubPanelView& panel_view) {
    auto& v = panels_v.nextResource();

    // copy to CPU
    copyToCPU(panel_view, mat_a, v);

    // compute on CPU
    using dlaf::eigensolver::internal::red2band::distributed::computePanelReflectors;
    computePanelReflectors(std::forward<TriggerSender>(trigger), rank_v0,
                           std::forward<CommSender>(mpi_col_chain_panel), v, mat_taus, j_sub,
                           panel_view);

    // copy back to GPU
    copyFromCPU(panel_view, v, mat_a);
  }

protected:
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panels_v;

  void copyToCPU(const matrix::SubPanelView panel_view, matrix::Matrix<T, Device::GPU>& mat_a,
                 matrix::Panel<Coord::Col, T, Device::CPU>& v) {
    namespace ex = pika::execution::experimental;

    using dlaf::internal::Policy;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;
    using pika::execution::thread_stacksize;

    for (const auto& i : panel_view.iteratorLocal()) {
      auto spec = panel_view(i);
      auto tmp_tile = v.readwrite(i);
      ex::start_detached(
          ex::when_all(splitTile(mat_a.read(i), spec), splitTile(std::move(tmp_tile), spec)) |
          matrix::copy(Policy<CopyBackend_v<Device::GPU, Device::CPU>>(thread_priority::high,
                                                                       thread_stacksize::nostack)));
    }
  }

  void copyFromCPU(const matrix::SubPanelView panel_view, matrix::Panel<Coord::Col, T, Device::CPU>& v,
                   matrix::Matrix<T, Device::GPU>& mat_a) {
    namespace ex = pika::execution::experimental;

    using dlaf::internal::Policy;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;
    using pika::execution::thread_stacksize;

    for (const auto& i : panel_view.iteratorLocal()) {
      auto spec = panel_view(i);
      auto tile_a = mat_a.readwrite(i);
      ex::start_detached(ex::when_all(splitTile(v.read(i), spec), splitTile(std::move(tile_a), spec)) |
                         matrix::copy(Policy<CopyBackend_v<Device::CPU, Device::GPU>>(
                             thread_priority::high, thread_stacksize::nostack)));
    }
  }
};
#endif

}

// Local implementation of reduction to band
template <Backend B, Device D, class T>
Matrix<T, Device::CPU> ReductionToBand<B, D, T>::call(Matrix<T, D>& mat_a, const SizeType band_size) {
  using dlaf::matrix::Matrix;
  using dlaf::matrix::Panel;

  using namespace red2band::local;

  using common::iterate_range2d;
  using factorization::internal::computeTFactor;

  using pika::execution::experimental::any_sender;

  const auto dist_a = mat_a.distribution();
  const matrix::Distribution dist({mat_a.size().rows(), band_size},
                                  {dist_a.blockSize().rows(), band_size});

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real nor complex)
  const SizeType nrefls = std::max<SizeType>(0, dist_a.size().rows() - band_size - 1);

  // Row-vector that is distributed over columns, but exists locally on all rows of the grid
  DLAF_ASSERT(mat_a.blockSize().cols() % band_size == 0, mat_a.blockSize().cols(), band_size);
  Matrix<T, Device::CPU> mat_taus(matrix::Distribution(GlobalElementSize(nrefls, 1),
                                                       TileElementSize(mat_a.blockSize().cols(), 1),
                                                       comm::Size2D(mat_a.commGridSize().cols(), 1),
                                                       comm::Index2D(mat_a.rankIndex().col(), 0),
                                                       comm::Index2D(mat_a.sourceRankIndex().col(), 0)));

  if (nrefls == 0)
    return mat_taus;

  Matrix<T, Device::CPU> mat_taus_retiled =
      mat_taus.retiledSubPipeline(LocalTileSize(mat_a.blockSize().cols() / band_size, 1));

  const SizeType ntiles = (nrefls - 1) / band_size + 1;
  DLAF_ASSERT(ntiles == mat_taus_retiled.nrTiles().rows(), ntiles, mat_taus_retiled.nrTiles().rows());

  const bool is_full_band = (band_size == dist_a.blockSize().cols());

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<Panel<Coord::Col, T, D>> panels_v(n_workspaces, dist);
  common::RoundRobin<Panel<Coord::Col, T, D>> panels_w(n_workspaces, dist);
  common::RoundRobin<Panel<Coord::Col, T, D>> panels_x(n_workspaces, dist);

  // Note:
  // Here dist_a is given with full panel size instead of dist with just the part actually needeed,
  // because the GPU Helper internally exploits Panel data-structure. Indeed, the full size panel is
  // needed in order to mimick Matrix with Panel, so it is possible to apply a SubPanelView to it.
  //
  // It is a bit hacky usage, because SubPanelView is not meant to be used with Panel, but just with
  // Matrix. This results in a variable waste of memory, depending no the ratio band_size/nb.
  red2band::ComputePanelHelper<B, D, T> compute_panel_helper(n_workspaces, dist_a);

  for (SizeType j_sub = 0; j_sub < ntiles; ++j_sub) {
    const auto i_sub = j_sub + 1;

    const GlobalElementIndex ij_offset(i_sub * band_size, j_sub * band_size);

    const SizeType nrefls_tile = mat_taus_retiled.tileSize(GlobalTileIndex(j_sub, 0)).rows();

    const bool isPanelIncomplete = (nrefls_tile != band_size);

    // Note: if this is running, it must have at least one valid reflector (i.e. with size > 1)
    DLAF_ASSERT_HEAVY(nrefls_tile != 0, nrefls_tile);

    // Note:  SubPanelView is (at most) band_size wide, but it may contain a smaller number of
    //        reflectors (i.e. at the end when last reflector size is 1)
    const matrix::SubPanelView panel_view(dist_a, ij_offset, band_size);

    Panel<Coord::Col, T, D>& v = panels_v.nextResource();
    v.setRangeStart(ij_offset);
    if (isPanelIncomplete)
      v.setWidth(nrefls_tile);

    // PANEL
    compute_panel_helper.call(mat_a, mat_taus_retiled, j_sub, panel_view);

    // Note:
    // - has_reflector_head tells if this rank owns the first tile of the panel (being local, always true)
    // - if !is_full_band it has to force copy as a workaround, otherwise in update matrix it would deadlock
    // due to tile shared between panel and trailing matrix
    constexpr bool has_reflector_head = true;
    setupReflectorPanelV<B, D, T>(has_reflector_head, panel_view, nrefls_tile, v, mat_a, !is_full_band);

    const LocalTileIndex t_idx(0, 0);
    // TODO used just by the column, maybe we can re-use a panel tile?
    // TODO probably the first one in any panel is ok?
    Matrix<T, D> t({nrefls_tile, nrefls_tile}, dist.blockSize());

    computeTFactor<B>(v, mat_taus_retiled.read(GlobalTileIndex(j_sub, 0)), t.readwrite(t_idx));

    // PREPARATION FOR TRAILING MATRIX UPDATE
    const GlobalElementIndex at_offset(ij_offset + GlobalElementSize(0, band_size));

    // Note: if there is no trailing matrix, algorithm has finised
    if (!at_offset.isIn(mat_a.size()))
      break;

    const matrix::SubMatrixView trailing_matrix_view(dist_a, at_offset);

    // W = V . T
    Panel<Coord::Col, T, D>& w = panels_w.nextResource();
    w.setRangeStart(at_offset);
    if (isPanelIncomplete)
      w.setWidth(nrefls_tile);

    trmmComputeW<B>(w, v, t.read(t_idx));

    // X = At . W
    Panel<Coord::Col, T, D>& x = panels_x.nextResource();
    x.setRangeStart(at_offset);
    if (isPanelIncomplete)
      x.setWidth(nrefls_tile);

    // Note:
    // Since At is hermitian, just the lower part is referenced.
    // When the tile is not part of the main diagonal, the same tile has to be used for two computations
    // that will contribute to two different rows of X: the ones indexed with row and col.
    hemmComputeX<B>(x, trailing_matrix_view, mat_a, w);

    // In the next section the next two operations are performed
    // A) W2 = W* . X
    // B) X -= 1/2 . V . W2

    // Note:
    // T can be re-used because it is not needed anymore in this step and it has the same shape
    Matrix<T, D> w2 = std::move(t);

    gemmComputeW2<B>(w2, w, x);
    gemmUpdateX<B>(x, w2, v);

    // TRAILING MATRIX UPDATE

    // At -= X . V* + V . X*
    her2kUpdateTrailingMatrix<B>(trailing_matrix_view, mat_a, x, v);

    x.reset();
    w.reset();
    v.reset();
  }

  return mat_taus;
}

// Distributed implementation of reduction to band
template <Backend B, Device D, class T>
Matrix<T, Device::CPU> ReductionToBand<B, D, T>::call(comm::CommunicatorGrid& grid, Matrix<T, D>& mat_a,
                                                      const SizeType band_size) {
  using namespace red2band::distributed;

  using common::iterate_range2d;
  using factorization::internal::computeTFactor;

  namespace ex = pika::execution::experimental;

  // Note:
  // This is a temporary workaround.
  // See issue https://github.com/eth-cscs/DLA-Future/issues/729
  pika::wait();

  // This algorithm requires the grid to have at least 2 independent column communicators in the round
  // robin array. If there is only one communicator mpi_col_chain and mpi_col_chain_panel will be
  // separate pipelines to the same communicator, but since communication is interleaved between the
  // pipelines this algorithm will deadlock (separate subpipelines means that all work on the previous
  // subpipeline has to complete before the next subpipeline can even start scheduling work).
  DLAF_ASSERT(grid.num_pipelines() >= 2, grid.num_pipelines());
  auto mpi_row_chain = grid.row_communicator_pipeline();
  auto mpi_col_chain = grid.col_communicator_pipeline();
  auto mpi_col_chain_panel = grid.col_communicator_pipeline();

#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_reduction_to_band_calls = 0;
  std::stringstream fname;
  fname << "reduction_to_band-" << matrix::internal::TypeToString_v<T> << "-"
        << std::to_string(num_reduction_to_band_calls) << ".h5";
  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_reduction_to_band_data) {
    file = matrix::internal::FileHDF5(grid.fullCommunicator(), fname.str());
    file->write(mat_a, "/input");
  }
#endif

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real nor complex)
  const SizeType nrefls = std::max<SizeType>(0, dist.size().rows() - band_size - 1);

  // Row-vector that is distributed over columns, but exists locally on all rows of the grid
  DLAF_ASSERT(mat_a.blockSize().cols() % band_size == 0, mat_a.blockSize().cols(), band_size);
  Matrix<T, Device::CPU> mat_taus(matrix::Distribution(GlobalElementSize(nrefls, 1),
                                                       TileElementSize(mat_a.blockSize().cols(), 1),
                                                       comm::Size2D(mat_a.commGridSize().cols(), 1),
                                                       comm::Index2D(mat_a.rankIndex().col(), 0),
                                                       comm::Index2D(mat_a.sourceRankIndex().col(), 0)));

  if (nrefls == 0) {
#ifdef DLAF_WITH_HDF5
    if (getTuneParameters().debug_dump_reduction_to_band_data) {
      file->write(mat_a, "/band");
    }

    num_reduction_to_band_calls++;
#endif

    return mat_taus;
  }

  Matrix<T, Device::CPU> mat_taus_retiled =
      mat_taus.retiledSubPipeline(LocalTileSize(mat_a.blockSize().cols() / band_size, 1));

  const SizeType ntiles = (nrefls - 1) / band_size + 1;
  DLAF_ASSERT(ntiles == mat_taus_retiled.nrTiles().rows(), ntiles, mat_taus_retiled.nrTiles().rows());

  const bool is_full_band = (band_size == dist.blockSize().cols());

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_v(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>> panels_vt(
      n_workspaces, dist);

  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_w(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>> panels_wt(
      n_workspaces, dist);

  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_x(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>> panels_xt(
      n_workspaces, dist);

  red2band::ComputePanelHelper<B, D, T> compute_panel_helper(n_workspaces, dist);

  ex::unique_any_sender<> trigger_panel{ex::just()};
  for (SizeType j_sub = 0; j_sub < ntiles; ++j_sub) {
    const SizeType i_sub = j_sub + 1;

    const GlobalElementIndex ij_offset(i_sub * band_size, j_sub * band_size);
    const GlobalElementIndex at_offset(i_sub * band_size, (j_sub + 1) * band_size);

    const comm::Index2D rank_v0{
        dist.template rankGlobalElement<Coord::Row>(ij_offset.row()),
        dist.template rankGlobalElement<Coord::Col>(ij_offset.col()),
    };

    const bool is_panel_rank_col = rank_v0.col() == rank.col();

    const SizeType nrefls_tile = mat_taus_retiled.tileSize(GlobalTileIndex(j_sub, 0)).rows();

    if (nrefls_tile == 0)
      break;

    auto& v = panels_v.nextResource();
    auto& vt = panels_vt.nextResource();

    v.setRangeStart(at_offset);
    vt.setRangeStart(at_offset);

    v.setWidth(nrefls_tile);
    vt.setHeight(nrefls_tile);

    const LocalTileIndex t_idx(0, 0);
    // TODO used just by the column, maybe we can re-use a panel tile?
    // TODO or we can keep just the sh_future and allocate just inside if (is_panel_rank_col)
    matrix::Matrix<T, D> t({nrefls_tile, nrefls_tile}, dist.blockSize());

    // PANEL
    const matrix::SubPanelView panel_view(dist, ij_offset, band_size);

    if (is_panel_rank_col) {
      compute_panel_helper.call(std::move(trigger_panel), rank_v0.row(), mpi_col_chain_panel.exclusive(),
                                mat_a, mat_taus_retiled, j_sub, panel_view);

      // Note:
      // - has_reflector_head tells if this rank owns the first tile of the panel
      // - if !is_full_band it has to force copy as a workaround, otherwise in update matrix it would
      // deadlock due to tile shared between panel and trailing matrix
      red2band::local::setupReflectorPanelV<B, D, T>(rank.row() == rank_v0.row(), panel_view,
                                                     nrefls_tile, v, mat_a, !is_full_band);
      computeTFactor<B>(v, mat_taus_retiled.read(GlobalTileIndex(j_sub, 0)), t.readwrite(t_idx),
                        mpi_col_chain);
    }

    // PREPARATION FOR TRAILING MATRIX UPDATE

    // Note: if there is no trailing matrix, algorithm has finised
    if (!at_offset.isIn(mat_a.size()))
      break;

    const matrix::SubMatrixView trailing_matrix_view(dist, at_offset);

    comm::broadcast(rank_v0.col(), v, vt, mpi_row_chain, mpi_col_chain);

    // W = V . T
    auto& w = panels_w.nextResource();
    auto& wt = panels_wt.nextResource();

    w.setRangeStart(at_offset);
    wt.setRangeStart(at_offset);

    w.setWidth(nrefls_tile);
    wt.setHeight(nrefls_tile);

    if (is_panel_rank_col)
      red2band::local::trmmComputeW<B, D>(w, v, t.read(t_idx));

    comm::broadcast(rank_v0.col(), w, wt, mpi_row_chain, mpi_col_chain);

    // X = At . W
    auto& x = panels_x.nextResource();
    auto& xt = panels_xt.nextResource();

    x.setRangeStart(at_offset);
    xt.setRangeStart(at_offset);

    x.setWidth(nrefls_tile);
    xt.setHeight(nrefls_tile);

    // Note:
    // Since At is hermitian, just the lower part is referenced.
    // When the tile is not part of the main diagonal, the same tile has to be used for two computations
    // that will contribute to two different rows of X: the ones indexed with row and col.
    // This is achieved by storing the two results in two different workspaces: X and X_conj respectively.
    //
    // On exit, x will contain a valid result just on ranks belonging to the column panel.
    // For what concerns xt, it is just used as support and it contains junk data on all ranks.
    hemmComputeX<B, D>(rank_v0.col(), x, xt, trailing_matrix_view, mat_a, w, wt, mpi_row_chain,
                       mpi_col_chain);

    // In the next section the next two operations are performed
    // A) W2 = W* . X
    // B) X -= 1/2 . V . W2

    // Note:
    // Now the intermediate result for X is available on the panel column ranks,
    // which have locally all the needed stuff for updating X and finalize the result
    if (is_panel_rank_col) {
      // Note:
      // T can be re-used because it is not needed anymore in this step and it has the same shape
      matrix::Matrix<T, D> w2 = std::move(t);

      red2band::local::gemmComputeW2<B, D>(w2, w, x);
      if (mpi_col_chain.size() > 1) {
        ex::start_detached(comm::schedule_all_reduce_in_place(mpi_col_chain.exclusive(), MPI_SUM,
                                                              w2.readwrite(LocalTileIndex(0, 0))));
      }

      red2band::local::gemmUpdateX<B, D>(x, w2, v);
    }

    // Note:
    // xt has been used previously as workspace for hemmComputeX, so it has to be reset, because now it
    // will be used for accessing the broadcasted version of x
    xt.reset();
    xt.setRangeStart(at_offset);
    xt.setHeight(nrefls_tile);

    comm::broadcast(rank_v0.col(), x, xt, mpi_row_chain, mpi_col_chain);

    // TRAILING MATRIX UPDATE

    // Note:
    // This trigger mechanism allows to control when the next iteration of compute panel will start.
    //
    // * What?
    // Compute panel uses MPI blocking communication that might block the only computing thread
    // available (since blocking communication are scheduled on normal queues and not on the MPI
    // dedicated one).
    //
    // * How?
    // If pika runtime has only 2 threads, one is dedicated to MPI and there is just one for
    // computation, that might get blocked by blocking MPI communication, without the chance to do
    // anything else. (TODO this might happen even with more reductions happening in parallel)
    //
    // * Why?
    // Panel computation at step i is done on the first column of the trailing matrix computed
    // at step i-1.
    // The rank owning the top-left tile of the trailing matrix, can update it as soon as it
    // receives X[0], which due to the pivot position is also the Xt[0]. Once it can go to the next
    // iteration, it ends up stucked in an MPI blocking communication, waiting for the others joining
    // before being able to advance.
    //
    // But at the same time, other ranks in the same column (needed for the next panel update), cannot
    // complete the trailing matrix update. Indeed, they are waiting for the pivot rank to communicate
    // column-wise Xt[0] (during x -> xt panel transpose broadcast), but he is not going to schedule
    // anything because the only normal thread which can do that is stuck in an MPI blocking
    // communication that is not going to advance... and so it's a DEADLOCK!
    //
    // * Solution:
    // The idea is to make the next panel depending not only on tiles stored locally, but also to
    // ensure that others have received Xt[0], which is needed to advance the computation and let
    // others arrive at the next iteration where the pivot will wait for them to complete the MPI
    // blocking communication.
    //
    // * Why is it different between MC and GPU?
    // As said above, the problem is related to the communication. But the communication is not said
    // to be an atomic operation happening in a single task. It might have to create a copy to
    // a buffer more suitable for the communication (e.g. GPU -> CPU if GPU-aware MPI is not
    // available).
    //
    // And in order to not be blocked, it must be ensured that the actual communication task has
    // been scheduled.
    const SizeType j_tile_current = ij_offset.col() / dist.blockSize().cols();
    const SizeType j_tile_next = at_offset.col() / dist.blockSize().cols();
    const bool isNextColumnOnSameRank = (j_tile_current == j_tile_next);
    const comm::IndexT_MPI rank_next_col =
        isNextColumnOnSameRank ? rank_v0.col() : (rank_v0.col() + 1) % dist.commGridSize().cols();

    if (rank.col() == rank_next_col) {
      const LocalTileIndex at{
          dist.template nextLocalTileFromGlobalElement<Coord::Row>(at_offset.row()),
          dist.template nextLocalTileFromGlobalElement<Coord::Col>(at_offset.col()),
      };

      // Note:
      // This additional communication of the last tile is a workaround for supporting following trigger
      // when b < mb.
      // Indeed, if b < mb the last column have (at least) a panel to compute, but differently from
      // other columns, broadcast transposed doesn't communicate the last tile, which is an assumption
      // needed to make the following trigger work correctly.
      const SizeType at_tile_col =
          dist.template globalTileFromGlobalElement<Coord::Col>(at_offset.col());

      if (at_tile_col == dist.nrTiles().cols() - 1) {
        const comm::IndexT_MPI owner = rank_v0.row();
        if (rank.row() == owner) {
          xt.setTile(at, x.read(at));

          if (dist.commGridSize().rows() > 1)
            ex::start_detached(comm::schedule_bcast_send(mpi_col_chain.exclusive(), xt.read(at)));
        }
        else {
          if (dist.commGridSize().rows() > 1)
            ex::start_detached(comm::schedule_bcast_recv(mpi_col_chain.exclusive(), owner,
                                                         xt.readwrite(at)));
        }
      }

      if constexpr (dlaf::comm::CommunicationDevice_v<D> == D) {
        // Note:
        // if there is no need for additional buffers, we can just wait that xt[0] is ready for
        // reading.
        if (rank.row() == rank_v0.row()) {
          trigger_panel = xt.read(at) | ex::drop_value() | ex::ensure_started();
        }
        else {
          // Note:
          // Conservatively ensure that xt[0] needed for updating the first column has been
          // received. Just wait for xt because communication of x happens over rows, while the
          // pivot rank can just block rank in the same column.
          trigger_panel = xt.read(at) | ex::drop_value() | ex::ensure_started();
        }
      }
      else {
        if (rank.row() == rank_v0.row()) {
          // Note:
          // on the pivot rank, i.e. the one that would quickly go to the next panel and block, from
          // implementation we know that xt[0] is set as an external tile pointing to x[0].
          // We cannot wait on xt readwrite (because it is an external tile in a panel, that constraints
          // it to be just readable), but we can wait on its source x[0]. This has a subtle implication,
          // since we will wait not just for the communication to be complete (which is already more
          // than what needed), but we will also wait till xt[0] will be released, so after all local
          // communication and computation on the first column of the trailing matrix will be completed.
          trigger_panel = x.readwrite(at) | ex::drop_value() | ex::ensure_started();
        }
        else {
          // Note:
          // Conservatively ensure that xt[0] needed for updating the first column has been
          // received. Just wait for xt because communication of x happens over rows, while the
          // pivot rank can just block rank in the same column.
          trigger_panel = xt.read(at) | ex::drop_value() | ex::ensure_started();
        }
      }
    }

    // At -= X . V* + V . X*
    her2kUpdateTrailingMatrix<B>(trailing_matrix_view, mat_a, x, vt, v, xt);

    xt.reset();
    x.reset();
    wt.reset();
    w.reset();
    vt.reset();
    v.reset();
  }

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_reduction_to_band_data) {
    file->write(mat_a, "/band");
  }

  num_reduction_to_band_calls++;
#endif

  return mat_taus;
}
}
