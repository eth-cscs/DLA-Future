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
#include <type_traits>
#include <vector>

#include <hpx/future.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/include/util.hpp>
#include <hpx/tuple.hpp>

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
#include "dlaf/matrix/tile.h"
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

template <class T>
T computeReflector(const comm::IndexT_MPI rank_v0, comm::Communicator& communicator,
                   const std::vector<TileT<T>>& panel, SizeType j) {
  const bool is_head_rank = rank_v0 == communicator.rank();

  // Extract x0 and compute local cumulative sum of squares of the reflector column
  std::array<T, 2> x0_and_squares{0, 0};
  auto it_begin = panel.begin();
  auto it_end = panel.end();

  if (is_head_rank) {
    auto& tile_v0 = *it_begin++;

    const TileElementIndex idx_x0(j, j);
    x0_and_squares[0] = tile_v0(idx_x0);

    T* reflector_ptr = tile_v0.ptr({idx_x0});
    x0_and_squares[1] =
        blas::dot(tile_v0.size().rows() - idx_x0.row(), reflector_ptr, 1, reflector_ptr, 1);
  }

  for (auto it = it_begin; it != it_end; ++it) {
    const auto& tile = *it;

    T* reflector_ptr = tile.ptr({0, j});
    x0_and_squares[1] += blas::dot(tile.size().rows(), reflector_ptr, 1, reflector_ptr, 1);
  }

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
                               common::make_data(x0_and_squares.data(), x0_and_squares.size()));

  const T norm = std::sqrt(x0_and_squares[1]);
  const T x0 = x0_and_squares[0];
  const T y = std::signbit(std::real(x0_and_squares[0])) ? norm : -norm;
  const T tau = (y - x0) / y;

  // compute reflector
  it_begin = panel.begin();
  it_end = panel.end();

  if (is_head_rank) {
    const auto& tile_v0 = *it_begin++;

    const TileElementIndex idx_x0(j, j);
    tile_v0(idx_x0) = y;

    if (j + 1 < tile_v0.size().rows()) {
      T* v = tile_v0.ptr({j + 1, j});
      blas::scal(tile_v0.size().rows() - (j + 1), T(1) / (x0 - y), v, 1);
    }
  }

  for (auto it = it_begin; it != it_end; ++it) {
    auto& tile_v = *it;
    T* v = tile_v.ptr({0, j});
    blas::scal(tile_v.size().rows(), T(1) / (x0 - y), v, 1);
  }

  return tau;
}

template <class T>
void updateTrailingPanelWithReflector(comm::IndexT_MPI rank_v0, comm::Communicator& communicator,
                                      const std::vector<TileT<T>>& panel, SizeType j,
                                      const SizeType pt_cols, const T tau) {
  // for each tile in the panel, consider just the trailing panel
  // i.e. all rows (height = reflector), just columns to the right of the current reflector
  if (!(pt_cols > 0))
    return;

  std::vector<T> w(pt_cols, 0);

  const bool is_head_rank = rank_v0 == communicator.rank();

  // TODO this is a workaround for detecting index 0 on global tile 0
  auto has_first = [&](const auto& tile) { return is_head_rank and &tile == &panel[0]; };

  const TileElementIndex index_el_x0(j, j);

  // W = Pt * V
  for (auto& tile_a : panel) {
    const bool has_first_component = has_first(tile_a);
    const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

    // clang-format off
    TileElementIndex        pt_start  {first_element, index_el_x0.col() + 1};
    TileElementSize         pt_size   {tile_a.size().rows() - pt_start.row(), pt_cols};

    TileElementIndex        v_start   {first_element, index_el_x0.col()};
    // clang-format on

    if (has_first_component) {
      const TileElementSize offset{1, 0};

      const T fake_v = 1;
      blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, offset.rows(), pt_size.cols(), T(1),
                 tile_a.ptr(pt_start), tile_a.ld(), &fake_v, 1, T(0), w.data(), 1);

      pt_start = pt_start + offset;
      v_start = v_start + offset;
      pt_size = pt_size - offset;
    }

    if (pt_start.isIn(tile_a.size())) {
      // W += 1 . A* . V
      blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, pt_size.rows(), pt_size.cols(), T(1),
                 tile_a.ptr(pt_start), tile_a.ld(), tile_a.ptr(v_start), 1, T(1), w.data(), 1);
    }
  }

  comm::sync::allReduceInPlace(communicator, MPI_SUM, common::make_data(w.data(), pt_cols));

  // update trailing panel
  // GER Pt = Pt - tau . v . w*
  for (auto& tile_a : panel) {
    const bool has_first_component = has_first(tile_a);
    const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

    // clang-format off
    TileElementIndex        pt_start{first_element, index_el_x0.col() + 1};
    TileElementSize         pt_size {tile_a.size().rows() - pt_start.row(), tile_a.size().cols() - pt_start.col()};

    TileElementIndex        v_start {first_element, index_el_x0.col()};
    // clang-format on

    if (has_first_component) {
      const TileElementSize offset{1, 0};

      // Pt = Pt - tau * v[0] * w*
      const T fake_v = 1;
      blas::ger(blas::Layout::ColMajor, 1, pt_size.cols(), -dlaf::conj(tau), &fake_v, 1, w.data(), 1,
                tile_a.ptr(pt_start), tile_a.ld());

      pt_start = pt_start + offset;
      v_start = v_start + offset;
      pt_size = pt_size - offset;
    }

    if (pt_start.isIn(tile_a.size())) {
      // Pt = Pt - tau * v * w*
      blas::ger(blas::Layout::ColMajor, pt_size.rows(), pt_size.cols(), -dlaf::conj(tau),
                tile_a.ptr(v_start), 1, w.data(), 1, tile_a.ptr(pt_start), tile_a.ld());
    }
  }
}

template <class T>
hpx::shared_future<common::internal::vector<T>> computePanelReflectors(
    hpx::future<void> trigger, comm::IndexT_MPI rank_v0,
    hpx::future<common::PromiseGuard<comm::Communicator>> mpi_col_chain_panel, MatrixT<T>& mat_a,
    const common::IterableRange2D<SizeType, matrix::LocalTile_TAG> ai_panel_range, SizeType nrefls) {
  auto panel_task = hpx::unwrapping(
      [rank_v0, nrefls, cols = mat_a.blockSize().cols()](auto fut_panel_tiles, auto communicator) {
        const auto panel_tiles = hpx::unwrap(fut_panel_tiles);

        common::internal::vector<T> taus;
        taus.reserve(nrefls);
        for (SizeType j = 0; j < nrefls; ++j) {
          taus.emplace_back(computeReflector(rank_v0, communicator.ref(), panel_tiles, j));
          updateTrailingPanelWithReflector(rank_v0, communicator.ref(), panel_tiles, j, cols - (j + 1),
                                           taus.back());
        }
        return taus;
      });

  auto panel_tiles = hpx::when_all(matrix::select(mat_a, ai_panel_range));

  return hpx::dataflow(getHpExecutor<Backend::MC>(), std::move(panel_task), std::move(panel_tiles),
                       mpi_col_chain_panel, std::move(trigger));
}

template <class T>
void setupReflectorPanelV(comm::IndexT_MPI rank_v0, const LocalTileSize& ai_offset,
                          const SizeType nrefls, PanelT<Coord::Col, T>& v, MatrixT<const T>& mat_a) {
  const auto rank = mat_a.rankIndex().row();

  // Note:
  // Reflectors are stored in the lower triangular part of the A matrix leading to sharing memory
  // between reflectors and results, which are in the upper triangular part. The problem exists only
  // for the first tile (of the V, i.e. band excluded). Since refelectors will be used in next
  // computations, they should be well-formed, i.e. a unit lower trapezoidal matrix. For this reason,
  // a support tile is used, where just the reflectors values are copied, the diagonal is set to 1
  // and the rest is zeroed out.
  auto it_begin = v.iteratorLocal().begin();
  auto it_end = v.iteratorLocal().end();

  if (rank_v0 == rank) {
    auto setupV0 = hpx::unwrapping([](auto&& tile_v, const auto& tile_a) {
      copy(tile_a, tile_v);
      tile::laset(lapack::MatrixType::Upper, T(0), T(1), tile_v);
    });

    // Note:
    // If the number of reflectors are limited by height (|reflector| > 1), the panel is narrower than
    // the blocksize, leading to just using a part of A
    const auto v0_index = indexFromOrigin(ai_offset);
    const auto& tile_a = splitTile(mat_a.read(v0_index), {{0, 0}, {nrefls, nrefls}});

    hpx::dataflow(getHpExecutor<Backend::MC>(), std::move(setupV0), v(v0_index), tile_a);

    ++it_begin;
  }

  // The rest of the V panel of reflectors can just point to the values in A, since they are
  // well formed in-place.
  for (auto it = it_begin; it < it_end; ++it) {
    const LocalTileIndex idx(it->row(), ai_offset.cols());
    v.setTile(idx, mat_a.read(idx));
  }
}

template <class T, class MatrixLikeT>
void trmmComputeW(PanelT<Coord::Col, T>& w, MatrixLikeT& v, FutureConstTile<T> tile_t) {
  const auto ex = getHpExecutor<Backend::MC>();

  auto trmm_func = hpx::unwrapping([](auto&& tile_w, const auto& tile_v, const auto& tile_t) -> void {
    // Note:
    // Since V0 is well-formed, by copying V0 to W we are also resetting W where the matrix is not going
    // to be computed.
    // TODO check this when number of reflectors is changed (i.e. skip last single-element reflector)
    copy(tile_v, tile_w);

    // W = V . T
    using namespace blas;
    tile::trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), tile_t, tile_w);
  });

  for (const auto& index_tile_w : w.iteratorLocal()) {
    // clang-format off
    FutureTile<T>      tile_w = w(index_tile_w);
    FutureConstTile<T> tile_v = v.read(index_tile_w);
    // clang-format on

    hpx::dataflow(ex, trmm_func, std::move(tile_w), std::move(tile_v), tile_t);
  }
}

template <class T>
void hemmComputeX(comm::IndexT_MPI reducer_col, PanelT<Coord::Col, T>& x, PanelT<Coord::Row, T>& xt,
                  const LocalTileSize at_offset, ConstMatrixT<T>& a, ConstPanelT<Coord::Col, T>& w,
                  ConstPanelT<Coord::Row, T>& wt, common::Pipeline<comm::Communicator>& mpi_row_chain,
                  common::Pipeline<comm::Communicator>& mpi_col_chain) {
  using matrix::unwrapExtendTiles;
  using tile::hemm_o;
  using tile::gemm_o;

  const auto ex = getHpExecutor<Backend::MC>();
  const auto ex_mpi = getMPIExecutor<Backend::MC>();

  const auto dist = a.distribution();
  const auto rank = dist.rankIndex();

  // Note:
  // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
  // "initialized" during computation, so they should not contribute with any spurious value to the final
  // result.
  //
  // TODO set0 can be "embedded" in the logic but currently it will be a bit cumbersome.
  x.clear();
  xt.clear();

  for (SizeType i = at_offset.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = limit - 1; j >= at_offset.cols(); --j) {
      const LocalTileIndex ij_local{i, j};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      if (is_diagonal_tile) {
        // clang-format off
        FutureTile<T>       tile_x = x(ij_local);
        FutureConstTile<T>  tile_a = a.read(ij_local);
        FutureConstTile<T>  tile_w = w.read(ij_local);
        // clang-format on

        hpx::dataflow(ex, unwrapExtendTiles(hemm_o), blas::Side::Left, blas::Uplo::Lower, T(1),
                      std::move(tile_a), std::move(tile_w), T(1), std::move(tile_x));
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

          // clang-format off
          FutureTile<T>       tile_x = x(ij_local);
          FutureConstTile<T>  tile_a = a.read(ij_local);
          FutureConstTile<T>  tile_w = wt.read(ij_local);
          // clang-format on

          hpx::dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::NoTrans, blas::Op::NoTrans, T(1),
                        std::move(tile_a), std::move(tile_w), T(1), std::move(tile_x));
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
          const auto owner = dist.template rankGlobalTile<Coord::Row>(ij.col());

          const LocalTileIndex index_x{dist.template localTileFromGlobalTile<Coord::Row>(ij.col()), 0};
          const LocalTileIndex index_xt{0, ij_local.col()};

          // clang-format off
          FutureTile<T>       tile_x = (dist.rankIndex().row() == owner) ? x(index_x) : xt(index_xt);
          FutureConstTile<T>  tile_a = a.read(ij_local);
          FutureConstTile<T>  tile_w = w.read(ij_local);
          // clang-format on

          hpx::dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::ConjTrans, blas::Op::NoTrans, T(1),
                        std::move(tile_a), std::move(tile_w), T(1), std::move(tile_x));
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
      comm::scheduleReduceRecvInPlace(ex_mpi, mpi_col_chain(), MPI_SUM, x({i, 0}));
    }
    else {
      comm::scheduleReduceSend(ex_mpi, rank_owner_row, mpi_col_chain(), MPI_SUM, xt.read(index_xt));
    }
  }

  // Note:
  // At this point partial results are all collected in X (Xt has been embedded in previous step),
  // so the last step needed is to reduce these last partial results in the final results.
  // The result is needed just on the column with reflectors.
  for (const auto& index_x : x.iteratorLocal()) {
    if (reducer_col == rank.col())
      comm::scheduleReduceRecvInPlace(ex_mpi, mpi_row_chain(), MPI_SUM, x(index_x));
    else
      comm::scheduleReduceSend(ex_mpi, reducer_col, mpi_row_chain(), MPI_SUM, x.read(index_x));
  }
}

template <class T>
void gemmComputeW2(MatrixT<T>& w2, ConstPanelT<Coord::Col, T>& w, ConstPanelT<Coord::Col, T>& x,
                   common::Pipeline<comm::Communicator>& mpi_col_chain) {
  using matrix::unwrapExtendTiles;
  using tile::gemm_o;

  const auto ex = getHpExecutor<Backend::MC>();
  const auto ex_mpi = getMPIExecutor<Backend::MC>();

  // Note:
  // Not all ranks in the column always hold at least a tile in the panel Ai, but all ranks in
  // the column are going to participate to the reduce. For them, it is important to set the
  // partial result W2 to zero.
  bool isW2initializedialized = false;

  // GEMM W2 = W* . X
  for (const auto& index_tile : w.iteratorLocal()) {
    isW2initializedialized = true;  // with C++20 this can be moved into the for init-statement

    const T beta = (index_tile.row() == w.rangeStartLocal()) ? 0 : 1;

    // clang-format off
    FutureTile<T>       tile_w2 = w2(LocalTileIndex{0, 0});
    FutureConstTile<T>  tile_w  = w.read(index_tile);
    FutureConstTile<T>  tile_x  = x.read(index_tile);
    // clang-format on

    hpx::dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::ConjTrans, blas::Op::NoTrans, T(1),
                  std::move(tile_w), std::move(tile_x), beta, std::move(tile_w2));
  }

  if (!isW2initializedialized)
    hpx::dataflow(ex, unwrapExtendTiles(tile::set0<T>), w2(LocalTileIndex(0, 0)));

  FutureTile<T> tile_w2 = w2(LocalTileIndex{0, 0});
  comm::scheduleAllReduceInPlace(ex_mpi, mpi_col_chain(), MPI_SUM, std::move(tile_w2));
}

template <class T, class MatrixLikeT>
void gemmUpdateX(PanelT<Coord::Col, T>& x, ConstMatrixT<T>& w2, MatrixLikeT& v) {
  using matrix::unwrapExtendTiles;
  using tile::gemm_o;

  const auto ex = getHpExecutor<Backend::MC>();

  // GEMM X = X - 0.5 . V . W2
  for (const auto& index_row : v.iteratorLocal()) {
    // clang-format off
    FutureTile<T>       tile_x  = x(index_row);
    FutureConstTile<T>  tile_v  = v.read(index_row);
    FutureConstTile<T>  tile_w2 = w2.read(LocalTileIndex{0, 0});
    // clang-format on

    hpx::dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::NoTrans, blas::Op::NoTrans, T(-0.5),
                  std::move(tile_v), std::move(tile_w2), T(1), std::move(tile_x));
  }
}

template <class T>
void her2kUpdateTrailingMatrix(const LocalTileSize& at_start, MatrixT<T>& a,
                               ConstPanelT<Coord::Col, T>& x, ConstPanelT<Coord::Row, T>& vt,
                               ConstPanelT<Coord::Col, T>& v, ConstPanelT<Coord::Row, T>& xt) {
  static_assert(std::is_signed<BaseType<T>>::value, "alpha in computations requires to be -1");

  using hpx::dataflow;
  using matrix::unwrapExtendTiles;
  using tile::her2k_o;
  using tile::gemm_o;

  const auto dist = a.distribution();

  for (SizeType i = at_start.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = at_start.cols(); j < limit; ++j) {
      const LocalTileIndex ij_local{i, j};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto& ex =
          (j == at_start.cols()) ? getHpExecutor<Backend::MC>() : getNpExecutor<Backend::MC>();

      if (is_diagonal_tile) {
        // clang-format off
        FutureTile<T>       tile_a = a(ij_local);
        FutureConstTile<T>  tile_v = v.read(ij_local);
        FutureConstTile<T>  tile_x = x.read(ij_local);
        // clang-format on

        dataflow(ex, unwrapExtendTiles(her2k_o), blas::Uplo::Lower, blas::Op::NoTrans, T(-1),
                 std::move(tile_v), std::move(tile_x), BaseType<T>(1), std::move(tile_a));
      }
      else {
        // GEMM A: X . V*
        {
          // clang-format off
          FutureTile<T>       tile_a = a(ij_local);
          FutureConstTile<T>  tile_x = x.read(ij_local);
          FutureConstTile<T>  tile_v = vt.read(ij_local);
          // clang-format on

          dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::NoTrans, blas::Op::ConjTrans, T(-1),
                   std::move(tile_x), std::move(tile_v), T(1), std::move(tile_a));
        }

        // GEMM A: V . X*
        {
          // clang-format off
          FutureTile<T>       tile_a = a(ij_local);
          FutureConstTile<T>  tile_v = v.read(ij_local);
          FutureConstTile<T>  tile_x = xt.read(ij_local);
          // clang-format on

          dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::NoTrans, blas::Op::ConjTrans, T(-1),
                   std::move(tile_v), std::move(tile_x), T(1), std::move(tile_a));
        }
      }
    }
  }
}
}

/// Distributed implementation of reduction to band
/// @return a vector of shared futures of vectors, where each inner vector contains a block of taus
template <class T>
std::vector<hpx::shared_future<common::internal::vector<T>>> ReductionToBand<
    Backend::MC, Device::CPU, T>::call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using common::iterate_range2d;
  using factorization::internal::computeTFactor;

  DLAF_ASSERT(equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(square_size(mat_a), mat_a.size());
  DLAF_ASSERT(square_blocksize(mat_a), mat_a.blockSize());

  const auto ex_mpi = getMPIExecutor<Backend::MC>();

  common::Pipeline<comm::Communicator> mpi_col_chain_panel(grid.colCommunicator());
  common::Pipeline<comm::Communicator> mpi_row_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_chain(grid.colCommunicator().clone());

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  std::vector<hpx::shared_future<common::internal::vector<T>>> taus;
  // TODO taus.reserve(); it's a minor optimization

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<PanelT<Coord::Col, T>> panels_v(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_vt(n_workspaces, dist);

  common::RoundRobin<PanelT<Coord::Col, T>> panels_w(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_wt(n_workspaces, dist);

  common::RoundRobin<PanelT<Coord::Col, T>> panels_x(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_xt(n_workspaces, dist);

  hpx::future<void> trigger_panel = hpx::make_ready_future<void>();
  for (SizeType j_panel = 0; j_panel < (dist.nrTiles().cols() - 1); ++j_panel) {
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

    const auto v0_size = mat_a.tileSize(ai_start);
    const SizeType nrefls = [v0_size]() {
      // TODO FIXME it can be improved, because the very last reflector of size 1 is not worth the effort
      return std::min(v0_size.cols(), v0_size.rows());
    }();

    PanelT<Coord::Col, T>& v = panels_v.nextResource();
    PanelT<Coord::Row, T>& vt = panels_vt.nextResource();

    v.setRangeStart(at_start);
    vt.setRangeStart(at_start);

    v.setWidth(nrefls);
    vt.setHeight(nrefls);

    const LocalTileIndex t_idx(0, 0);
    // TODO used just by the column, maybe we can re-use a panel tile?
    // TODO or we can keep just the sh_future and allocate just inside if (is_panel_rank_col)
    MatrixT<T> t({nrefls, nrefls}, dist.blockSize());

    // PANEL
    if (is_panel_rank_col) {
      taus.emplace_back(computePanelReflectors(std::move(trigger_panel), rank_v0.row(),
                                               mpi_col_chain_panel(), mat_a, ai_panel, nrefls));
      computeTFactor<Backend::MC>(nrefls, mat_a, ai_start, taus.back(), t(t_idx), mpi_col_chain);
      setupReflectorPanelV(rank_v0.row(), ai_offset, nrefls, v, mat_a);
    }

    // PREPARATION FOR TRAILING MATRIX UPDATE
    comm::broadcast(ex_mpi, rank_v0.col(), v, vt, mpi_row_chain, mpi_col_chain);

    // W = V . T
    PanelT<Coord::Col, T>& w = panels_w.nextResource();
    PanelT<Coord::Row, T>& wt = panels_wt.nextResource();

    w.setRangeStart(at_start);
    wt.setRangeStart(at_start);

    w.setWidth(nrefls);
    wt.setHeight(nrefls);

    if (is_panel_rank_col)
      trmmComputeW(w, v, t.read(t_idx));

    comm::broadcast(ex_mpi, rank_v0.col(), w, wt, mpi_row_chain, mpi_col_chain);

    // X = At . W
    PanelT<Coord::Col, T>& x = panels_x.nextResource();
    PanelT<Coord::Row, T>& xt = panels_xt.nextResource();

    x.setRangeStart(at_start);
    xt.setRangeStart(at_start);

    x.setWidth(nrefls);
    xt.setHeight(nrefls);

    // Note:
    // Since At is hermitian, just the lower part is referenced.
    // When the tile is not part of the main diagonal, the same tile has to be used for two computations
    // that will contribute to two different rows of X: the ones indexed with row and col.
    // This is achieved by storing the two results in two different workspaces: X and X_conj respectively.
    //
    // On exit, x will contain a valid result just on ranks belonging to the column panel.
    // For what concerns xt, it is just used as support and it contains junk data on all ranks.
    hemmComputeX(rank_v0.col(), x, xt, at_offset, mat_a, w, wt, mpi_row_chain, mpi_col_chain);

    // In the next section the next two operations are performed
    // A) W2 = W* . X
    // B) X -= 1/2 . V . W2

    // Note:
    // Now the intermediate result for X is available on the panel column ranks,
    // which have locally all the needed stuff for updating X and finalize the result
    if (is_panel_rank_col) {
      // Note:
      // T can be re-used because it is not needed anymore in this step and it has the same shape
      MatrixT<T> w2 = std::move(t);

      gemmComputeW2(w2, w, x, mpi_col_chain);

      gemmUpdateX(x, w2, v);
    }

    comm::broadcast(ex_mpi, rank_v0.col(), x, xt, mpi_row_chain, mpi_col_chain);

    // TRAILING MATRIX UPDATE

    // Note:
    // This is a checkpoint that it is used to trigger the computation of the next iteration.
    //
    // It is a checkpoint to ensure advancements in specific edge cases, due to the usage of
    // blocking MPI calls inside the panel computation.
    // If there is just a single worker thread (MPI excluded) per rank, and one of the ranks ends up
    // having an empty panel, it does not have any dependency so it is ready to start the computation
    // of this empty panel, but at the same time, since it has to partecipate to the collective blocking
    // calls, it may start and block the only worker thread available. If it starts before this point of
    // the previous iteration is reached, then a deadlock is created. Indeed, the offending rank is
    // blocked waiting to do nothing on the next iteration (computePanel), while other ranks would be
    // stuck waiting for it for completing steps of the previous iteration, needed for the update of the
    // trailing matrix that will unlock the next iteration.
    trigger_panel = hpx::when_all(selectRead(x, x.iteratorLocal()), selectRead(xt, xt.iteratorLocal()));

    // At -= X . V* + V . X*
    her2kUpdateTrailingMatrix(at_offset, mat_a, x, vt, v, xt);

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
