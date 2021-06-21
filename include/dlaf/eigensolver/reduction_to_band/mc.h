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

#include <cmath>
#include <string>
#include <vector>

#include <hpx/future.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/include/util.hpp>
#include <hpx/tuple.hpp>
#include <lapack/util.hh>

#include "dlaf/blas/tile.h"
#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/communication/kernels/all_reduce.h"
#include "dlaf/communication/kernels/reduce.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/reduction_to_band/api.h"
#include "dlaf/factorization/qr.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
struct ReductionToBand<Backend::MC, Device::CPU, T> {
  static std::vector<hpx::shared_future<common::internal::vector<T>>> call(
      comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a);
};

namespace {

using matrix::Matrix;
using matrix::Tile;

template <class Type>
using MatrixT = Matrix<Type, Device::CPU>;
template <class Type>
using ConstMatrixT = MatrixT<const Type>;
template <class Type>
using TileT = Tile<Type, Device::CPU>;
template <class Type>
using ConstTileT = TileT<const Type>;

template <Coord panel_type, class T>
using PanelT = matrix::Panel<panel_type, T, Device::CPU>;
template <Coord panel_type, class T>
using ConstPanelT = PanelT<panel_type, const T>;

template <class Type>
using FutureTile = hpx::future<TileT<Type>>;
template <class Type>
using FutureConstTile = hpx::shared_future<ConstTileT<Type>>;

template <class Type>
using MemViewT = memory::MemoryView<Type, Device::CPU>;

template <class T>
T computeReflector(const comm::IndexT_MPI rank_v0, comm::Communicator& communicator,
                   const std::vector<TileT<T>>& panel, SizeType j) {
  const bool is_head_rank = rank_v0 == communicator.rank();

  // Extract x0 and compute local cumulative sum of squares of the reflector column
  T x0 = 0;
  T squares = 0;
  auto it_begin = panel.begin();
  auto it_end = panel.end();

  if (is_head_rank) {
    auto& tile_v0 = *it_begin++;

    const TileElementIndex idx_x0(j, j);
    x0 = tile_v0(idx_x0);

    T* reflector_ptr = tile_v0.ptr({idx_x0});
    squares = blas::dot(tile_v0.size().rows() - idx_x0.row(), reflector_ptr, 1, reflector_ptr, 1);
  }

  for (auto it = it_begin; it != it_end; ++it) {
    auto& tile = *it;

    T* reflector_ptr = tile.ptr({0, j});
    squares += blas::dot(tile.size().rows(), reflector_ptr, 1, reflector_ptr, 1);
  }

  // reduce local cumulative sums
  // rank_v0 will have the x0 and the total cumulative sum of squares
  comm::sync::reduceInPlace(rank_v0, communicator, MPI_SUM, common::make_data(&squares, 1));

  // rank_v0 will compute params that will be used for next computation of reflector components
  // FIXME in this case just one compute and the other will receive it
  // it may be better to compute on each one, in order to avoid a communication of few values
  // but it would benefit if all_reduce of the norm and x0 is faster than communicating params
  std::array<T, 3> params;
  if (is_head_rank) {
    const T norm = std::sqrt(squares);
    const T y = std::signbit(std::real(x0)) ? norm : -norm;
    const T tau = (y - x0) / y;

    params = {x0, y, tau};
  }

  // broadcast params
  auto params_data = common::make_data(params.data(), params.size());
  if (is_head_rank)
    comm::sync::broadcast::send(communicator, params_data);
  else
    comm::sync::broadcast::receive_from(rank_v0, communicator, params_data);

  // compute reflector
  it_begin = panel.begin();
  it_end = panel.end();

  if (is_head_rank) {
    auto& tile_v0 = *it_begin++;

    const TileElementIndex idx_x0(j, j);
    tile_v0(idx_x0) = params[1];

    if (j + 1 < tile_v0.size().rows()) {
      T* v = tile_v0.ptr({j + 1, j});
      blas::scal(tile_v0.size().rows() - (j + 1),
                 typename TypeInfo<T>::BaseType(1) / (params[0] - params[1]), v, 1);
    }
  }

  for (auto it = it_begin; it != it_end; ++it) {
    auto& tile_v = *it;
    T* v = tile_v.ptr({0, j});
    blas::scal(tile_v.size().rows(), typename TypeInfo<T>::BaseType(1) / (params[0] - params[1]), v, 1);
  }

  return params[2];
}

template <class T>
void updateTrailingPanelWithReflector(comm::IndexT_MPI rank_v0, comm::Communicator& communicator,
                                      TileT<T>& tile_w, const std::vector<TileT<T>>& panel, SizeType j,
                                      const T tau) {
  const SizeType pt_cols = tile_w.size().cols() - (j + 1);

  // for each tile in the panel, consider just the trailing panel
  // i.e. all rows (height = reflector), just columns to the right of the current reflector
  if (!(pt_cols > 0))
    return;

  lapack::laset(lapack::MatrixType::General, pt_cols, 1, 0, 0, tile_w.ptr(), tile_w.ld());

  const bool is_head_rank = rank_v0 == communicator.rank();

  // TODO this is a workaround for detecing index 0 on global tile 0
  auto has_first = [&](const auto& tile) { return is_head_rank and &tile == &panel[0]; };

  const TileElementIndex index_el_x0{j, j};

  // W = Pt * W
  for (auto& tile_a : panel) {
    const bool has_first_component = has_first(tile_a);
    const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

    // clang-format off
    TileElementIndex        pt_start  {first_element, index_el_x0.col() + 1};
    TileElementSize         pt_size   {tile_a.size().rows() - pt_start.row(), pt_cols};

    TileElementIndex        v_start   {first_element, index_el_x0.col()};
    const TileElementIndex  w_start   {0, 0};
    // clang-format on

    if (has_first_component) {
      const TileElementSize offset{1, 0};

      const T fake_v = 1;
      // clang-format off
      blas::gemv(blas::Layout::ColMajor,
          blas::Op::ConjTrans,
          offset.rows(), pt_size.cols(),
          static_cast<T>(1),
          tile_a.ptr(pt_start), tile_a.ld(),
          &fake_v, 1,
          static_cast<T>(0),
          tile_w.ptr(w_start), 1);
      // clang-format on

      pt_start = pt_start + offset;
      v_start = v_start + offset;
      pt_size = pt_size - offset;
    }

    if (pt_start.isIn(tile_a.size())) {
      // W += 1 . A* . V
      // clang-format off
      blas::gemv(blas::Layout::ColMajor,
          blas::Op::ConjTrans,
          pt_size.rows(), pt_size.cols(),
          static_cast<T>(1),
          tile_a.ptr(pt_start), tile_a.ld(),
          tile_a.ptr(v_start), 1,
          1,
          tile_w.ptr(w_start), 1);
      // clang-format on
    }
  }

  comm::sync::allReduceInPlace(communicator, MPI_SUM, common::make_data(tile_w));

  // update trailing panel
  // GER Pt = Pt - tau . v . w*
  for (auto& tile_a : panel) {
    const bool has_first_component = has_first(tile_a);
    const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

    // clang-format off
    TileElementIndex        pt_start{first_element, index_el_x0.col() + 1};
    TileElementSize         pt_size {tile_a.size().rows() - pt_start.row(), tile_a.size().cols() - pt_start.col()};

    TileElementIndex        v_start {first_element, index_el_x0.col()};
    const TileElementIndex  w_start {0, 0};
    // clang-format on

    if (has_first_component) {
      const TileElementSize offset{1, 0};

      // Pt = Pt - tau * v[0] * w*
      // clang-format off
      const T fake_v = 1;
      blas::ger(blas::Layout::ColMajor,
          1, pt_size.cols(),
          -dlaf::conj(tau),
          &fake_v, 1,
          tile_w.ptr(w_start), 1,
          tile_a.ptr(pt_start), tile_a.ld());
      // clang-format on

      pt_start = pt_start + offset;
      v_start = v_start + offset;
      pt_size = pt_size - offset;
    }

    if (pt_start.isIn(tile_a.size())) {
      // Pt = Pt - tau * v * w*
      // clang-format off
      blas::ger(blas::Layout::ColMajor,
          pt_size.rows(), pt_size.cols(),
          -dlaf::conj(tau),
          tile_a.ptr(v_start), 1,
          tile_w.ptr(w_start), 1,
          tile_a.ptr(pt_start), tile_a.ld());
      // clang-format on
    }
  }
}

template <class T>
hpx::shared_future<common::internal::vector<T>> computeAndUpdatePanel(
    comm::IndexT_MPI rank_v0, hpx::future<common::PromiseGuard<comm::Communicator>> bmpi_col_task_chain,
    MatrixT<T>& mat_a,
    const common::IterableRange2D<SizeType, dlaf::matrix::LocalTile_TAG> ai_panel_range,
    SizeType k_reflectors) {
  using hpx::util::unwrapping;

  auto panel_task =
      unwrapping([rank_v0, k_reflectors](auto fut_panel_tiles, auto communicator, auto tile_w) {
        auto panel_tiles = hpx::util::unwrap(fut_panel_tiles);

        common::internal::vector<T> taus(k_reflectors);
        for (SizeType j = 0; j < k_reflectors; ++j) {
          const auto tau = computeReflector(rank_v0, communicator.ref(), panel_tiles, j);
          taus[j] = tau;
          updateTrailingPanelWithReflector(rank_v0, communicator.ref(), tile_w, panel_tiles, j, tau);
        }
        return taus;
      });

  MatrixT<T> w({mat_a.blockSize().rows(), mat_a.blockSize().cols()}, mat_a.blockSize());
  auto panel_tiles = dlaf::matrix::select(mat_a, ai_panel_range);

  // clang-format off
  return hpx::dataflow(
      dlaf::getHpExecutor<Backend::MC>(),
      panel_task, hpx::when_all(panel_tiles), bmpi_col_task_chain, w(LocalTileIndex{0, 0}));
  // clang-format on
}

template <class T, class MatrixLikeT>
void compute_w(PanelT<Coord::Col, T>& w, MatrixLikeT& v, FutureConstTile<T> tile_t) {
  const auto ex = dlaf::getHpExecutor<Backend::MC>();

  auto trmm_func =
      hpx::util::unwrapping([](auto&& tile_w, const auto& tile_v, const auto& tile_t) -> void {
        // Note:
        // Since T can be smaller then the entire block, here its size is used to update potentially
        // just a sub-part of the resulting W tile, which currently still works as an entire block.
        // T can be of reduced-size when there are less reflectors than columns, so when the V tile
        // is also reduced in the number of rows, which currently happens just when working on the
        // last tile fully containing a reflector. This, together with the fact that V0 is well-formed
        // implies that by copying V0 to W we are also resetting W where the matrix is not going to
        // be computed.
        // TODO check this when number of reflectors is changed (i.e. skip last single-element reflector)

        dlaf::tile::lacpy(tile_v, tile_w);

        // W = V . T
        // clang-format off
        blas::trmm(blas::Layout::ColMajor,
            blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            tile_w.size().rows(), tile_t.size().rows(),
            static_cast<T>(1),
            tile_t.ptr(), tile_t.ld(),
            tile_w.ptr(), tile_w.ld());
        // clang-format on
      });

  for (const auto& index_tile_w : w.iterator()) {
    // clang-format off
    FutureTile<T>      tile_w = w(index_tile_w);
    FutureConstTile<T> tile_v = v.read(index_tile_w);
    // clang-format on

    hpx::dataflow(ex, trmm_func, std::move(tile_w), std::move(tile_v), tile_t);
  }
}

// TODO document that it stores in Xcols just the ones for which he is not on the right row,
// otherwise it directly compute also gemm2 inside Xrows
template <class T>
void compute_x(comm::IndexT_MPI reducer_col, PanelT<Coord::Col, T>& x, PanelT<Coord::Row, T>& xt,
               const LocalTileSize at_offset, ConstMatrixT<T>& a, ConstPanelT<Coord::Col, T>& w,
               ConstPanelT<Coord::Row, T>& wt, common::Pipeline<comm::Communicator>& mpi_row_task_chain,
               common::Pipeline<comm::Communicator>& mpi_col_task_chain) {
  using hpx::util::unwrapping;
  using dlaf::common::make_data;
  using dlaf::comm::sync::reduce;

  const auto ex = dlaf::getHpExecutor<Backend::MC>();
  const auto ex_mpi = dlaf::getMPIExecutor<Backend::MC>();

  const auto dist = a.distribution();
  const auto rank = dist.rankIndex();

  for (SizeType i = at_offset.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = limit - 1; j >= at_offset.cols(); --j) {
      const LocalTileIndex index_a_loc{i, j};

      const GlobalTileIndex index_a = dist.globalTileIndex(index_a_loc);

      const bool is_first = j == limit - 1;
      const bool is_diagonal_tile = (index_a.row() == index_a.col());

      if (is_diagonal_tile) {
        const LocalTileIndex index_x{index_a_loc.row(), 0};
        const LocalTileIndex index_w{index_a_loc.row(), 0};

        // clang-format off
        FutureTile<T>       tile_x = x(index_x);
        FutureConstTile<T>  tile_a = a.read(index_a_loc);
        FutureConstTile<T>  tile_w = w.read(index_w);
        // clang-format on

        hpx::dataflow(ex, unwrapping(dlaf::tile::hemm<T>), blas::Side::Left, blas::Uplo::Lower,
                      static_cast<T>(1), std::move(tile_a), std::move(tile_w),
                      static_cast<T>(is_first ? 0 : 1), std::move(tile_x));
      }
      else {
        // A . W*
        {
          // Note:
          // Since it is not a diagonal tile, otherwise it would have been managed in the previous
          // branch, the second operand is not available in W but it is accessible through the
          // support panel Wt.
          // However, since we are still computing the "straight" part, the result can be stored
          // in the "local" panel X.
          const LocalTileIndex index_x{index_a_loc.row(), 0};
          const LocalTileIndex index_wt{0, index_a_loc.col()};

          // clang-format off
          FutureTile<T>       tile_x = x(index_x);
          FutureConstTile<T>  tile_a = a.read(index_a_loc);
          FutureConstTile<T>  tile_w = wt.read(index_wt);
          // clang-format on

          hpx::dataflow(ex, unwrapping(dlaf::tile::gemm<T>), blas::Op::NoTrans, blas::Op::NoTrans,
                        static_cast<T>(1), std::move(tile_a), std::move(tile_w),
                        static_cast<T>(is_first ? 0 : 1), std::move(tile_x));
        }

        // A* . W
        {
          // Note:
          // Here we are considering the hermitian part of A, so coordinates have to be "mirrored".
          // So, first step is identifying the mirrored cell coordinate, i.e. swap row/col, together
          // with realizing if the new coord lays on an owned row or not.
          // If yes, the result can be stored in the X, otherwise Xt support panel will be used.
          // For what concerns the second operand, it can be found for sure in W. In fact, the
          // multiplication requires matching col(A) == row(W), but since coordinates are mirrored,
          // we are mathing row(A) == row(W), so it is local by construction.
          const auto owner = dist.template rankGlobalTile<Coord::Row>(index_a.col());

          const LocalTileIndex index_x{dist.template localTileFromGlobalTile<Coord::Row>(index_a.col()),
                                       0};
          const LocalTileIndex index_xt{0, index_a_loc.col()};

          const LocalTileIndex index_w{index_a_loc.row(), 0};

          const bool is_first_xt = (dist.rankIndex().row() != owner) && is_first;

          // clang-format off
          FutureTile<T>       tile_x = (dist.rankIndex().row() == owner) ? x(index_x) : xt(index_xt);
          FutureConstTile<T>  tile_a = a.read(index_a_loc);
          FutureConstTile<T>  tile_w = w.read(index_w);
          // clang-format on

          hpx::dataflow(ex, unwrapping(dlaf::tile::gemm<T>), blas::Op::ConjTrans, blas::Op::NoTrans,
                        static_cast<T>(1), std::move(tile_a), std::move(tile_w),
                        static_cast<T>(is_first_xt ? 0 : 1), std::move(tile_x));
        }
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
  for (const auto& index_xt : xt.iterator()) {
    const auto index_k = dist.template globalTileFromLocalTile<Coord::Col>(index_xt.col());
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(index_k);

    if (rank_owner_row == rank.row()) {
      // Note:
      // Since it is the owner, it has to perform the "mirroring" of the results from columns to
      // rows.
      const auto i = dist.template localTileFromGlobalTile<Coord::Row>(index_k);
      // TODO here the IN-PLACE is happening (because Xt is not used for this tile on this rank)
      comm::scheduleReduceRecvInPlace(ex_mpi, mpi_col_task_chain(), MPI_SUM, x({i, 0}));
    }
    else {
      comm::scheduleReduceSend(ex_mpi, rank_owner_row, mpi_col_task_chain(), MPI_SUM, xt.read(index_xt));
    }
  }

  // Note:
  // At this point partial results are all collected in X (Xt has been embedded in previous step),
  // so the last step needed is to reduce these last partial results in the final results.
  // The result is needed just on the column with reflectors.
  for (const auto& index_x_loc : x.iterator()) {
    if (reducer_col == rank.col())
      comm::scheduleReduceRecvInPlace(ex_mpi, mpi_row_task_chain(), MPI_SUM, x(index_x_loc));
    else
      comm::scheduleReduceSend(ex_mpi, reducer_col, mpi_row_task_chain(), MPI_SUM, x.read(index_x_loc));
  }
}

template <class T>
void compute_w2(MatrixT<T>& w2, ConstPanelT<Coord::Col, T>& w, ConstPanelT<Coord::Col, T>& x,
                common::Pipeline<comm::Communicator>& mpi_col_task_chain) {
  using hpx::util::unwrapping;
  using common::make_data;
  using namespace comm::sync;

  const auto ex = dlaf::getHpExecutor<Backend::MC>();
  const auto ex_mpi = dlaf::getMPIExecutor<Backend::MC>();

  // Note:
  // Not all ranks in the column always hold at least a tile in the panel Ai, but all ranks in
  // the column are going to participate to the reduce. For them, it is important to set the
  // partial result W2 to zero.
  dlaf::matrix::util::set(w2, [](...) { return 0; });

  // GEMM W2 = W* . X
  for (const auto& index_tile : w.iterator()) {
    const T beta = (index_tile.row() == 0) ? 0 : 1;

    // clang-format off
    FutureTile<T>       tile_w2 = w2(LocalTileIndex{0, 0});
    FutureConstTile<T>  tile_w  = w.read(index_tile);
    FutureConstTile<T>  tile_x  = x.read(index_tile);
    // clang-format on

    hpx::dataflow(ex, unwrapping(dlaf::tile::gemm<T>), blas::Op::ConjTrans, blas::Op::NoTrans,
                  static_cast<T>(1), std::move(tile_w), std::move(tile_x), beta, std::move(tile_w2));
  }

  // all-reduce instead of computing it on each node, everyone in the panel should have it
  FutureTile<T> tile_w2 = w2(LocalTileIndex{0, 0});
  comm::scheduleAllReduceInPlace(ex_mpi, mpi_col_task_chain(), MPI_SUM, std::move(tile_w2));
}

template <class T, class MatrixLikeT>
void update_x(PanelT<Coord::Col, T>& x, ConstMatrixT<T>& w2, MatrixLikeT& v) {
  using hpx::util::unwrapping;

  const auto ex = dlaf::getHpExecutor<Backend::MC>();

  // GEMM X = X - 0.5 . V . W2
  for (const auto& index_row : v.iterator()) {
    // clang-format off
    FutureTile<T>       tile_x  = x(index_row);
    FutureConstTile<T>  tile_v  = v.read(index_row);
    FutureConstTile<T>  tile_w2 = w2.read(LocalTileIndex{0, 0});
    // clang-format on

    hpx::dataflow(ex, unwrapping(dlaf::tile::gemm<T>), blas::Op::NoTrans, blas::Op::NoTrans,
                  static_cast<T>(-0.5), std::move(tile_v), std::move(tile_w2), static_cast<T>(1),
                  std::move(tile_x));
  }
}

template <class T>
void update_a(const LocalTileSize& at_start, MatrixT<T>& a, ConstPanelT<Coord::Col, T>& x,
              ConstPanelT<Coord::Row, T>& vt, ConstPanelT<Coord::Col, T>& v,
              ConstPanelT<Coord::Row, T>& xt) {
  using hpx::util::unwrapping;

  const auto dist = a.distribution();

  for (SizeType i = at_start.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = at_start.cols(); j < limit; ++j) {
      const LocalTileIndex index_at{i, j};

      const GlobalTileIndex index_a = dist.globalTileIndex(index_at);  // TODO possible FIXME
      const bool is_diagonal_tile = (index_a.row() == index_a.col());

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto& ex = (j == at_start.cols()) ? dlaf::getHpExecutor<Backend::MC>()
                                              : dlaf::getNpExecutor<Backend::MC>();

      if (is_diagonal_tile) {
        // HER2K
        const LocalTileIndex index_x{index_at.row(), 0};

        // clang-format off
        FutureTile<T>       tile_a = a(index_at);
        FutureConstTile<T>  tile_v = v.read(index_x);
        FutureConstTile<T>  tile_x = x.read(index_x);
        // clang-format on

        const T alpha = -1;  // TODO T must be a signed type
        hpx::dataflow(ex, unwrapping(dlaf::tile::her2k<T>), blas::Uplo::Lower, blas::Op::NoTrans, alpha,
                      std::move(tile_v), std::move(tile_x),
                      static_cast<typename TypeInfo<T>::BaseType>(1), std::move(tile_a));
      }
      else {
        // GEMM A: X . V*
        {
          const LocalTileIndex index_x{index_at.row(), 0};

          // clang-format off
          FutureTile<T>       tile_a = a(index_at);
          FutureConstTile<T>  tile_x = x.read(index_x);
          FutureConstTile<T>  tile_v = vt.read({0, index_at.col()});
          // clang-format on

          const T alpha = -1;  // TODO T must be a sigend type
          hpx::dataflow(ex, unwrapping(dlaf::tile::gemm<T>), blas::Op::NoTrans, blas::Op::ConjTrans,
                        alpha, std::move(tile_x), std::move(tile_v), static_cast<T>(1),
                        std::move(tile_a));
        }

        // GEMM A: V . X*
        {
          // clang-format off
          FutureTile<T>       tile_a = a(index_at);
          FutureConstTile<T>  tile_v = v.read({index_at.row(), 0});
          FutureConstTile<T>  tile_x = xt.read({0, index_at.col()});
          // clang-format on

          const T alpha = -1;  // TODO T must be a sigend type
          hpx::dataflow(ex, unwrapping(dlaf::tile::gemm<T>), blas::Op::NoTrans, blas::Op::ConjTrans,
                        alpha, std::move(tile_v), std::move(tile_x), static_cast<T>(1),
                        std::move(tile_a));
        }
      }
    }
  }
}

}

/// Distributed implementation of reduction to band
/// @return a list of shared futures of vectors, where each vector contains a block of taus
template <class T>
std::vector<hpx::shared_future<common::internal::vector<T>>> ReductionToBand<
    Backend::MC, Device::CPU, T>::call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using namespace comm;
  using namespace comm::sync;

  using hpx::util::unwrapping;

  using common::iterate_range2d;
  using common::make_data;

  using factorization::internal::computeTFactor;

  DLAF_ASSERT(equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(square_size(mat_a), mat_a.size());
  DLAF_ASSERT(square_blocksize(mat_a), mat_a.blockSize());

  const auto ex_mpi = dlaf::getMPIExecutor<Backend::MC>();

  common::Pipeline<comm::Communicator> bmpi_col_task_chain(grid.colCommunicator());
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  const SizeType nb = mat_a.blockSize().rows();
  // TODO not yet implemented for the moment the panel is tile-wide
  // const SizeType band_size = nb;

  std::vector<hpx::shared_future<common::internal::vector<T>>> taus;

  constexpr std::size_t n_workspaces = 1;
  common::RoundRobin<PanelT<Coord::Col, T>> panels_v(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_vt(n_workspaces, dist);

  common::RoundRobin<PanelT<Coord::Col, T>> panels_w(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_wt(n_workspaces, dist);

  common::RoundRobin<PanelT<Coord::Col, T>> panels_x(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_xt(n_workspaces, dist);

  for (SizeType j_panel = 0; j_panel < (dist.nrTiles().cols() - 1); ++j_panel) {
    PanelT<Coord::Col, T>& v = panels_v.nextResource();
    PanelT<Coord::Row, T>& vt = panels_vt.nextResource();
    PanelT<Coord::Col, T>& w = panels_w.nextResource();
    PanelT<Coord::Row, T>& wt = panels_wt.nextResource();
    PanelT<Coord::Col, T>& x = panels_x.nextResource();
    PanelT<Coord::Row, T>& xt = panels_xt.nextResource();

    const GlobalTileIndex ai_start{GlobalTileIndex{j_panel, j_panel} + GlobalTileSize{1, 0}};
    const GlobalTileIndex at_start{ai_start + GlobalTileSize{0, 1}};

    const comm::Index2D rank_v0 = dist.rankGlobalTile(ai_start);

    const LocalTileSize ai_offset{
        dist.template nextLocalTileFromGlobalTile<Coord::Row>(ai_start.row()),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(ai_start.col()),
    };
    const LocalTileSize at_offset{
        ai_offset.rows(),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(at_start.col()),
    };

    const auto ai_panel =
        iterate_range2d(LocalTileIndex{0, 0} + ai_offset,
                        LocalTileSize{dist.localNrTiles().rows() - ai_offset.rows(), 1});

    const bool is_panel_rank_col = rank_v0.col() == rank.col();

    v.setRangeStart(at_offset);

    // TODO it can be improved, because the very last reflector of size 1 is not worth the effort
    const SizeType k_reflectors = [v0_size = mat_a.tileSize(ai_start)]() {
      return std::min(v0_size.cols(), v0_size.rows());
    }();

    const LocalTileIndex t_idx(0, 0);
    // TODO used just by the column, maybe we can re-use a panel tile?
    MatrixT<T> t({k_reflectors, k_reflectors}, dist.blockSize());

    // 1. PANEL
    if (is_panel_rank_col) {
      // Note:
      // for each column in the panel, compute reflector and update panel
      // if this block has the last reflector, that would be just the first 1, skip the last column
      auto taus_panel =
          computeAndUpdatePanel(rank_v0.row(), bmpi_col_task_chain(), mat_a, ai_panel, k_reflectors);

      taus.push_back(taus_panel);

      // Prepare T and V for the next step

      computeTFactor<Backend::MC>(k_reflectors, mat_a, ai_start, taus_panel, t(t_idx),
                                  mpi_col_task_chain);

      // Note:
      // Reflectors are stored in the lower triangular part of the A matrix leading to sharing memory
      // between reflectors and results, which are in the upper triangular part. The problem exists only
      // for the first tile (of the V, i.e. band excluded). Since refelectors will be used in next
      // computations, they should be well-formed, i.e. a unit lower trapezoidal matrix. For this reason,
      // a support tile is used, where just the reflectors values are copied, the diagonal is set to 1
      // and the rest is zeroed out.
      if (rank_v0 == rank) {
        auto setup_V0_func = unwrapping([](auto&& tile_v, const auto& tile_a) {
          dlaf::tile::lacpy(tile_a, tile_v);

          // set upper part to zero and 1 on diagonal (reflectors)
          // clang-format off
          lapack::laset(lapack::MatrixType::Upper,
              tile_v.size().rows(), tile_v.size().cols(),
              T(0), // off diag
              T(1), // on  diag
              tile_v.ptr(), tile_v.ld());
          // clang-format on
        });

        const LocalTileIndex v0_index(ai_offset.rows(), ai_offset.cols());
        hpx::dataflow(dlaf::getHpExecutor<Backend::MC>(), setup_V0_func, v(v0_index),
                      mat_a.read(v0_index));
      }

      // The rest of the V panel of reflectors can just point to the values in A, since they are
      // well formed in-place.
      for (auto row = dist.template nextLocalTileFromGlobalTile<Coord::Row>(ai_start.row() + 1);
           row < dist.localNrTiles().rows(); ++row) {
        const LocalTileIndex idx{row, ai_offset.cols()};
        v.setTile(idx, mat_a.read(idx));
      }
    }

    vt.setRangeStart(at_offset);
    comm::broadcast(ex_mpi, rank_v0.col(), v, vt, mpi_row_task_chain, mpi_col_task_chain);

    // UPDATE TRAILING MATRIX

    // COMPUTE W
    // W = V . T
    w.setRangeStart(at_offset);

    if (is_panel_rank_col)
      compute_w(w, v, t.read(t_idx));

    wt.setRangeStart(at_offset);
    comm::broadcast(ex_mpi, rank_v0.col(), w, wt, mpi_row_task_chain, mpi_col_task_chain);

    // COMPUTE X
    // X = At . W
    // Since At is hermitian, just the lower part is referenced.
    // When the tile is not part of the main diagonal, the same tile has to be used for two computations
    // that will contribute to two different rows of X: the ones indexed with row and col.
    // This is achieved by storing the two results in two different workspaces: X and X_conj respectively.

    // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
    // "initialized" during computation, so they should not contribute with any spurious value to the final result
    x.setRangeStart(at_offset);
    xt.setRangeStart(at_offset);

    x.clear();
    xt.clear();

    compute_x(rank_v0.col(), x, xt, at_offset, mat_a, w, wt, mpi_row_task_chain, mpi_col_task_chain);

    // Now the intermediate result for X is available on the panel column rank,
    // which has locally all the needed stuff for updating X and finalize the result

    // W2 can be computed by the panel column rank only, it is the only one that has the X
    if (is_panel_rank_col) {
      // Note:
      // w2 could in theory re-use T-factor tile, but currently T may have a different size.
      // Indeed, T size depends on the number of the reflectors in this step, while W2 is still
      // working on the full size tile (it can be improved in the future)
      MatrixT<T> w2({nb, nb}, dist.blockSize());

      compute_w2(w2, w, x, mpi_col_task_chain);

      update_x(x, w2, v);
    }

    comm::broadcast(ex_mpi, rank_v0.col(), x, xt, mpi_row_task_chain, mpi_col_task_chain);

    // UPDATE
    // At = At - X . V* + V . X*
    update_a(at_offset, mat_a, x, vt, v, xt);

    xt.reset();
    x.reset();

    wt.reset();
    w.reset();

    vt.reset();
    v.reset();
  }

  return taus;
}

/// ---- ETI
#define DLAF_EIGENSOLVER_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct ReductionToBand<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_MC_ETI(extern, float)
DLAF_EIGENSOLVER_MC_ETI(extern, double)
DLAF_EIGENSOLVER_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_MC_ETI(extern, std::complex<double>)

}
}
}
