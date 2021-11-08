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
#include <vector>

#include <hpx/local/future.hpp>
#include <hpx/local/unwrap.hpp>

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
  static std::vector<hpx::shared_future<common::internal::vector<T>>> call(Matrix<T, Device::CPU>& mat_a);
  static std::vector<hpx::shared_future<common::internal::vector<T>>> call(
      comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a);
};

namespace red2band {
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

// Extract x0 and compute local cumulative sum of squares of the reflector column
template <class T>
std::array<T, 2> computeX0AndSquares(const bool has_head, const std::vector<TileT<T>>& panel,
                                     SizeType j) {
  std::array<T, 2> x0_and_squares{0, 0};
  auto it_begin = panel.begin();
  auto it_end = panel.end();

  if (has_head) {
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
  return x0_and_squares;
}

template <class T>
T computeReflectorAndTau(const bool has_head, const std::vector<TileT<T>>& panel, const SizeType j,
                         std::array<T, 2> x0_and_squares) {
  const T norm = std::sqrt(x0_and_squares[1]);
  const T x0 = x0_and_squares[0];
  const T y = std::signbit(std::real(x0_and_squares[0])) ? norm : -norm;
  const T tau = (y - x0) / y;

  auto it_begin = panel.begin();
  auto it_end = panel.end();

  if (has_head) {
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
std::vector<T> computeWTrailingPanel(const bool has_head, const std::vector<TileT<T>>& panel, SizeType j,
                                     const SizeType pt_cols) {
  // for each tile in the panel, consider just the trailing panel
  // i.e. all rows (height = reflector), just columns to the right of the current reflector
  if (!(pt_cols > 0))
    return {};

  std::vector<T> w(to_sizet(pt_cols), 0);

  const TileElementIndex index_el_x0(j, j);
  bool has_first_component = has_head;

  // W = Pt * V
  for (auto& tile_a : panel) {
    const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

    TileElementIndex pt_start{first_element, index_el_x0.col() + 1};
    TileElementSize pt_size{tile_a.size().rows() - pt_start.row(), pt_cols};
    TileElementIndex v_start{first_element, index_el_x0.col()};

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

    has_first_component = false;
  }

  return w;
}

template <class T>
void updateTrailingPanel(const bool has_head, const std::vector<TileT<T>>& panel, SizeType j,
                         const std::vector<T>& w, const T tau) {
  const TileElementIndex index_el_x0(j, j);

  bool has_first_component = has_head;

  // GER Pt = Pt - tau . v . w*
  for (auto& tile_a : panel) {
    const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

    TileElementIndex pt_start{first_element, index_el_x0.col() + 1};
    TileElementSize pt_size{tile_a.size().rows() - pt_start.row(),
                            tile_a.size().cols() - pt_start.col()};
    TileElementIndex v_start{first_element, index_el_x0.col()};

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

    has_first_component = false;
  }
}

template <class Executor, class T>
void hemmDiag(const Executor& ex, hpx::shared_future<TileT<const T>> tile_a,
              hpx::shared_future<TileT<const T>> tile_w, hpx::future<TileT<T>> tile_x) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::hemm_o), blas::Side::Left,
                blas::Uplo::Lower, T(1), std::move(tile_a), std::move(tile_w), T(1), std::move(tile_x));
}

// X += op(A) * W
template <class Executor, class T>
void hemmOffDiag(const Executor& ex, blas::Op op, hpx::shared_future<TileT<const T>> tile_a,
                 hpx::shared_future<TileT<const T>> tile_w, hpx::future<TileT<T>> tile_x) {
  hpx::dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), op, blas::Op::NoTrans, T(1),
                std::move(tile_a), std::move(tile_w), T(1), std::move(tile_x));
}

template <class Executor, class T>
void her2kDiag(const Executor& ex, hpx::shared_future<TileT<const T>> tile_v,
               hpx::shared_future<TileT<const T>> tile_x, hpx::future<TileT<T>> tile_a) {
  dataflow(ex, matrix::unwrapExtendTiles(tile::internal::her2k_o), blas::Uplo::Lower, blas::Op::NoTrans,
           T(-1), std::move(tile_v), std::move(tile_x), BaseType<T>(1), std::move(tile_a));
}

// C -= A . B*
template <class Executor, class T>
void her2kOffDiag(const Executor& ex, hpx::shared_future<TileT<const T>> tile_a,
                  hpx::shared_future<TileT<const T>> tile_b, hpx::future<TileT<T>> tile_c) {
  dataflow(ex, matrix::unwrapExtendTiles(tile::internal::gemm_o), blas::Op::NoTrans, blas::Op::ConjTrans,
           T(-1), std::move(tile_a), std::move(tile_b), T(1), std::move(tile_c));
}

namespace local {

template <class T>
T computeReflector(const std::vector<TileT<T>>& panel, SizeType j) {
  constexpr bool has_head = true;

  std::array<T, 2> x0_and_squares = computeX0AndSquares(has_head, panel, j);

  auto tau = computeReflectorAndTau(has_head, panel, j, std::move(x0_and_squares));

  return tau;
}

template <class T>
void updateTrailingPanelWithReflector(const std::vector<TileT<T>>& panel, SizeType j,
                                      const SizeType pt_cols, const T tau) {
  constexpr bool has_head = true;
  std::vector<T> w = computeWTrailingPanel(has_head, panel, j, pt_cols);

  if (w.size() == 0)
    return;

  updateTrailingPanel(has_head, panel, j, w, tau);
}

template <class T>
hpx::shared_future<common::internal::vector<T>> computePanelReflectors(
    MatrixT<T>& mat_a, const common::IterableRange2D<SizeType, matrix::LocalTile_TAG> ai_panel_range,
    SizeType nrefls) {
  auto panel_task = hpx::unwrapping([nrefls, cols = mat_a.blockSize().cols()](auto fut_panel_tiles) {
    const auto panel_tiles = hpx::unwrap(fut_panel_tiles);

    common::internal::vector<T> taus;
    taus.reserve(nrefls);
    for (SizeType j = 0; j < nrefls; ++j) {
      taus.emplace_back(computeReflector(panel_tiles, j));
      updateTrailingPanelWithReflector(panel_tiles, j, cols - (j + 1), taus.back());
    }
    return taus;
  });

  auto panel_tiles = hpx::when_all(matrix::select(mat_a, ai_panel_range));

  return hpx::dataflow(getHpExecutor<Backend::MC>(), std::move(panel_task), std::move(panel_tiles));
}

template <class T>
void setupReflectorPanelV(bool has_head, const LocalTileSize& ai_offset, const SizeType nrefls,
                          PanelT<Coord::Col, T>& v, MatrixT<const T>& mat_a) {
  // Note:
  // Reflectors are stored in the lower triangular part of the A matrix leading to sharing memory
  // between reflectors and results, which are in the upper triangular part. The problem exists only
  // for the first tile (of the V, i.e. band excluded). Since refelectors will be used in next
  // computations, they should be well-formed, i.e. a unit lower trapezoidal matrix. For this reason,
  // a support tile is used, where just the reflectors values are copied, the diagonal is set to 1
  // and the rest is zeroed out.
  auto it_begin = v.iteratorLocal().begin();
  auto it_end = v.iteratorLocal().end();

  if (has_head) {
    auto setupV0 = hpx::unwrapping([](auto&& tile_v, const auto& tile_a) {
      matrix::internal::copy(tile_a, tile_v);
      tile::internal::laset(lapack::MatrixType::Upper, T(0), T(1), tile_v);
    });

    // Note:
    // If the number of reflectors are limited by height (|reflector| > 1), the panel is narrower than
    // the blocksize, leading to just using a part of A (first full nrefls columns)
    const auto malformed_v0_idx = indexFromOrigin(ai_offset);
    const auto nrows = mat_a.tileSize(GlobalTileIndex(Coord::Row, v.rangeStart())).rows();
    const auto& tile_a = splitTile(mat_a.read(malformed_v0_idx), {{0, 0}, {nrows, nrefls}});

    hpx::dataflow(getHpExecutor<Backend::MC>(), std::move(setupV0), v(malformed_v0_idx), tile_a);

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
void trmmComputeW(PanelT<Coord::Col, T>& w, MatrixLikeT& v, hpx::shared_future<ConstTileT<T>> tile_t) {
  const auto ex = getHpExecutor<Backend::MC>();

  auto trmm_func = hpx::unwrapping([](auto&& tile_w, const auto& tile_v, const auto& tile_t) -> void {
    // Note:
    // Since V0 is well-formed, by copying V0 to W we are also resetting W where the matrix is not going
    // to be computed.
    matrix::internal::copy(tile_v, tile_w);

    // W = V . T
    using namespace blas;
    tile::internal::trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), tile_t, tile_w);
  });

  for (const auto& index_i : w.iteratorLocal())
    hpx::dataflow(ex, trmm_func, w(index_i), v.read(index_i), tile_t);
}

template <class T, class MatrixLikeT>
void gemmUpdateX(PanelT<Coord::Col, T>& x, ConstMatrixT<T>& w2, MatrixLikeT& v) {
  using matrix::unwrapExtendTiles;
  using tile::internal::gemm_o;

  const auto ex = getHpExecutor<Backend::MC>();

  // GEMM X = X - 0.5 . V . W2
  for (const auto& index_i : v.iteratorLocal())
    hpx::dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::NoTrans, blas::Op::NoTrans, T(-0.5),
                  v.read(index_i), w2.read(LocalTileIndex(0, 0)), T(1), x(index_i));
}

template <class T>
void hemmComputeX(PanelT<Coord::Col, T>& x, const LocalTileSize at_offset, ConstMatrixT<T>& a,
                  ConstPanelT<Coord::Col, T>& w) {
  const auto ex = getHpExecutor<Backend::MC>();

  const auto dist = a.distribution();

  // Note:
  // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
  // "initialized" during computation, so they should not contribute with any spurious value to the final
  // result.
  //
  // TODO set0 can be "embedded" in the logic but currently it will be a bit cumbersome.
  matrix::util::set0(ex, x);

  for (SizeType i = at_offset.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = i + 1;
    for (SizeType j = limit - 1; j >= at_offset.cols(); --j) {
      const LocalTileIndex ij{i, j};

      const bool is_diagonal_tile = (ij.row() == ij.col());

      if (is_diagonal_tile) {
        hemmDiag(ex, a.read(ij), w.read(ij), x(ij));
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
          hemmOffDiag(ex, blas::Op::NoTrans, a.read(ij), w.read(index_w), x(index_x));
        }

        {
          const LocalTileIndex index_pretended = transposed(ij);
          const LocalTileIndex index_x(Coord::Row, index_pretended.row());
          const LocalTileIndex index_w(Coord::Row, index_pretended.col());
          hemmOffDiag(ex, blas::Op::ConjTrans, a.read(ij), w.read(index_w), x(index_x));
        }
      }
    }
  }
}

template <class T>
void gemmComputeW2(MatrixT<T>& w2, ConstPanelT<Coord::Col, T>& w, ConstPanelT<Coord::Col, T>& x) {
  using matrix::unwrapExtendTiles;
  using tile::internal::gemm_o;

  const auto ex = getHpExecutor<Backend::MC>();

  // Note:
  // Not all ranks in the column always hold at least a tile in the panel Ai, but all ranks in
  // the column are going to participate to the reduce. For them, it is important to set the
  // partial result W2 to zero.
  hpx::dataflow(ex, unwrapExtendTiles(tile::internal::set0_o), w2(LocalTileIndex(0, 0)));

  // GEMM W2 = W* . X
  for (const auto& index_tile : w.iteratorLocal())
    hpx::dataflow(ex, unwrapExtendTiles(gemm_o), blas::Op::ConjTrans, blas::Op::NoTrans, T(1),
                  w.read(index_tile), x.read(index_tile), T(1), w2(LocalTileIndex(0, 0)));
}

template <class T>
void her2kUpdateTrailingMatrix(const LocalTileSize& at_start, MatrixT<T>& a,
                               ConstPanelT<Coord::Col, T>& x, ConstPanelT<Coord::Col, T>& v) {
  static_assert(std::is_signed<BaseType<T>>::value, "alpha in computations requires to be -1");

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
        her2kDiag(ex, v.read(ij_local), x.read(ij_local), a(ij_local));
      }
      else {
        // A -= X . V*
        her2kOffDiag(ex, x.read(ij_local), v.read(transposed(ij_local)), a(ij_local));

        // A -= V . X*
        her2kOffDiag(ex, v.read(ij_local), x.read(transposed(ij_local)), a(ij_local));
      }
    }
  }
}

}

namespace distributed {
template <class T>
T computeReflector(const bool has_head, comm::Communicator& communicator,
                   const std::vector<TileT<T>>& panel, SizeType j) {
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

template <class T>
void updateTrailingPanelWithReflector(const bool has_head, comm::Communicator& communicator,
                                      const std::vector<TileT<T>>& panel, SizeType j,
                                      const SizeType pt_cols, const T tau) {
  std::vector<T> w = computeWTrailingPanel(has_head, panel, j, pt_cols);

  if (w.size() == 0)
    return;

  comm::sync::allReduceInPlace(communicator, MPI_SUM, common::make_data(w.data(), pt_cols));

  updateTrailingPanel(has_head, panel, j, w, tau);
}

template <class T>
hpx::shared_future<common::internal::vector<T>> computePanelReflectors(
    hpx::future<void> trigger, comm::IndexT_MPI rank_v0,
    hpx::future<common::PromiseGuard<comm::Communicator>> mpi_col_chain_panel, MatrixT<T>& mat_a,
    const common::IterableRange2D<SizeType, matrix::LocalTile_TAG> ai_panel_range, SizeType nrefls) {
  auto panel_task = hpx::unwrapping(
      [rank_v0, nrefls, cols = mat_a.blockSize().cols()](auto fut_panel_tiles, auto comm_wrapper) {
        auto communicator = comm_wrapper.ref();
        const bool has_head = communicator.rank() == rank_v0;

        const auto panel_tiles = hpx::unwrap(fut_panel_tiles);

        common::internal::vector<T> taus;
        taus.reserve(nrefls);
        for (SizeType j = 0; j < nrefls; ++j) {
          taus.emplace_back(computeReflector(has_head, communicator, panel_tiles, j));
          updateTrailingPanelWithReflector(has_head, communicator, panel_tiles, j, cols - (j + 1),
                                           taus.back());
        }
        return taus;
      });

  auto panel_tiles = hpx::when_all(matrix::select(mat_a, ai_panel_range));

  return hpx::dataflow(getHpExecutor<Backend::MC>(), std::move(panel_task), std::move(panel_tiles),
                       mpi_col_chain_panel, std::move(trigger));
}

template <class T>
void hemmComputeX(comm::IndexT_MPI reducer_col, PanelT<Coord::Col, T>& x, PanelT<Coord::Row, T>& xt,
                  const LocalTileSize at_offset, ConstMatrixT<T>& a, ConstPanelT<Coord::Col, T>& w,
                  ConstPanelT<Coord::Row, T>& wt, common::Pipeline<comm::Communicator>& mpi_row_chain,
                  common::Pipeline<comm::Communicator>& mpi_col_chain) {
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
  matrix::util::set0(ex, x);
  matrix::util::set0(ex, xt);

  for (SizeType i = at_offset.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = limit - 1; j >= at_offset.cols(); --j) {
      const LocalTileIndex ij_local{i, j};
      const GlobalTileIndex ij = dist.globalTileIndex(ij_local);

      const bool is_diagonal_tile = (ij.row() == ij.col());

      if (is_diagonal_tile) {
        hemmDiag(ex, a.read(ij_local), w.read(ij_local), x(ij_local));
      }
      else {
        // Note:
        // Since it is not a diagonal tile, otherwise it would have been managed in the previous
        // branch, the second operand is not available in W but it is accessible through the
        // support panel Wt.
        // However, since we are still computing the "straight" part, the result can be stored
        // in the "local" panel X.
        hemmOffDiag(ex, blas::Op::NoTrans, a.read(ij_local), wt.read(ij_local), x(ij_local));

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

        auto tile_x = (dist.rankIndex().row() == owner) ? x(index_x) : xt(index_xt);

        hemmOffDiag(ex, blas::Op::ConjTrans, a.read(ij_local), w.read(ij_local), std::move(tile_x));
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
void her2kUpdateTrailingMatrix(const LocalTileSize& at_start, MatrixT<T>& a,
                               ConstPanelT<Coord::Col, T>& x, ConstPanelT<Coord::Row, T>& vt,
                               ConstPanelT<Coord::Col, T>& v, ConstPanelT<Coord::Row, T>& xt) {
  static_assert(std::is_signed<BaseType<T>>::value, "alpha in computations requires to be -1");

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
        her2kDiag(ex, v.read(ij_local), x.read(ij_local), a(ij_local));
      }
      else {
        // A -= X . V*
        her2kOffDiag(ex, x.read(ij_local), vt.read(ij_local), a(ij_local));

        // A -= V . X*
        her2kOffDiag(ex, v.read(ij_local), xt.read(ij_local), a(ij_local));
      }
    }
  }
}
}
}

/// Local implementation of reduction to band
/// @return a vector of shared futures of vectors, where each inner vector contains a block of taus
template <class T>
std::vector<hpx::shared_future<common::internal::vector<T>>> ReductionToBand<
    Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_a) {
  using namespace red2band::local;
  using red2band::MatrixT;
  using red2band::PanelT;

  using common::iterate_range2d;
  using factorization::internal::computeTFactor;

  const auto dist = mat_a.distribution();

  const SizeType nblocks = dist.nrTiles().cols() - 1;
  std::vector<hpx::shared_future<common::internal::vector<T>>> taus;
  taus.reserve(to_sizet(nblocks));

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<PanelT<Coord::Col, T>> panels_v(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Col, T>> panels_w(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Col, T>> panels_x(n_workspaces, dist);

  for (SizeType j_block = 0; j_block < nblocks; ++j_block) {
    const GlobalTileIndex ai_start{GlobalTileIndex{j_block, j_block} + GlobalTileSize{1, 0}};
    const GlobalTileIndex at_start{ai_start + GlobalTileSize{0, 1}};

    const LocalTileSize ai_offset{ai_start.row(), ai_start.col()};
    const LocalTileSize at_offset{at_start.row(), at_start.col()};

    const auto ai_panel =
        iterate_range2d(indexFromOrigin(ai_offset),
                        LocalTileSize(dist.localNrTiles().rows() - ai_offset.rows(), 1));

    const auto v0_size = mat_a.tileSize(ai_start);
    const SizeType nrefls = [ai_start, nrTiles = mat_a.nrTiles(), v0_size]() {
      if (ai_start.row() != nrTiles.rows() - 1)
        return v0_size.cols();
      else
        return std::min(v0_size.rows(), v0_size.cols()) - 1;
    }();

    if (nrefls == 0)
      break;

    PanelT<Coord::Col, T>& v = panels_v.nextResource();
    v.setRangeStart(ai_start);
    v.setWidth(nrefls);

    const LocalTileIndex t_idx(0, 0);
    // TODO used just by the column, maybe we can re-use a panel tile?
    MatrixT<T> t({nrefls, nrefls}, dist.blockSize());

    // PANEL
    constexpr bool has_reflector_head = true;
    taus.emplace_back(computePanelReflectors(mat_a, ai_panel, nrefls));
    computeTFactor<Backend::MC>(nrefls, mat_a, ai_start, taus.back(), t(t_idx));
    setupReflectorPanelV(has_reflector_head, ai_offset, nrefls, v, mat_a);

    // PREPARATION FOR TRAILING MATRIX UPDATE

    // W = V . T
    PanelT<Coord::Col, T>& w = panels_w.nextResource();
    w.setRangeStart(at_start);
    w.setWidth(nrefls);

    trmmComputeW(w, v, t.read(t_idx));

    // X = At . W
    PanelT<Coord::Col, T>& x = panels_x.nextResource();
    x.setRangeStart(at_start);
    x.setWidth(nrefls);

    // Note:
    // Since At is hermitian, just the lower part is referenced.
    // When the tile is not part of the main diagonal, the same tile has to be used for two computations
    // that will contribute to two different rows of X: the ones indexed with row and col.
    hemmComputeX(x, at_offset, mat_a, w);

    // In the next section the next two operations are performed
    // A) W2 = W* . X
    // B) X -= 1/2 . V . W2

    // Note:
    // T can be re-used because it is not needed anymore in this step and it has the same shape
    MatrixT<T> w2 = std::move(t);

    gemmComputeW2(w2, w, x);
    gemmUpdateX(x, w2, v);

    // TRAILING MATRIX UPDATE

    // At -= X . V* + V . X*
    her2kUpdateTrailingMatrix(at_offset, mat_a, x, v);

    x.reset();
    w.reset();
    v.reset();
  }

  return taus;
}

/// Distributed implementation of reduction to band
/// @return a vector of shared futures of vectors, where each inner vector contains a block of taus
template <class T>
std::vector<hpx::shared_future<common::internal::vector<T>>> ReductionToBand<
    Backend::MC, Device::CPU, T>::call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using namespace red2band::distributed;
  using red2band::MatrixT;
  using red2band::PanelT;

  using common::iterate_range2d;
  using factorization::internal::computeTFactor;

  const auto ex_mpi = getMPIExecutor<Backend::MC>();

  common::Pipeline<comm::Communicator> mpi_col_chain_panel(grid.colCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_row_chain(grid.rowCommunicator());
  common::Pipeline<comm::Communicator> mpi_col_chain(grid.colCommunicator());

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  const SizeType nblocks = dist.nrTiles().cols() - 1;
  std::vector<hpx::shared_future<common::internal::vector<T>>> taus;
  taus.reserve(to_sizet(nblocks));

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<PanelT<Coord::Col, T>> panels_v(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_vt(n_workspaces, dist);

  common::RoundRobin<PanelT<Coord::Col, T>> panels_w(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_wt(n_workspaces, dist);

  common::RoundRobin<PanelT<Coord::Col, T>> panels_x(n_workspaces, dist);
  common::RoundRobin<PanelT<Coord::Row, T>> panels_xt(n_workspaces, dist);

  hpx::future<void> trigger_panel = hpx::make_ready_future<void>();
  for (SizeType j_block = 0; j_block < nblocks; ++j_block) {
    const GlobalTileIndex ai_start{GlobalTileIndex{j_block, j_block} + GlobalTileSize{1, 0}};
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
        iterate_range2d(indexFromOrigin(ai_offset),
                        LocalTileSize(dist.localNrTiles().rows() - ai_offset.rows(), 1));

    const bool is_panel_rank_col = rank_v0.col() == rank.col();

    const auto v0_size = mat_a.tileSize(ai_start);
    const SizeType nrefls = [ai_start, nrTiles = mat_a.nrTiles(), v0_size]() {
      if (ai_start.row() != nrTiles.rows() - 1)
        return v0_size.cols();
      else
        return std::min(v0_size.rows(), v0_size.cols()) - 1;
    }();

    if (nrefls == 0)
      break;

    PanelT<Coord::Col, T>& v = panels_v.nextResource();
    PanelT<Coord::Row, T>& vt = panels_vt.nextResource();

    v.setRangeStart(ai_start);
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
      red2band::local::setupReflectorPanelV(rank.row() == rank_v0.row(), ai_offset, nrefls, v, mat_a);
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
      red2band::local::trmmComputeW(w, v, t.read(t_idx));

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

      red2band::local::gemmComputeW2(w2, w, x);
      comm::scheduleAllReduceInPlace(ex_mpi, mpi_col_chain(), MPI_SUM, w2(LocalTileIndex(0, 0)));

      red2band::local::gemmUpdateX(x, w2, v);
    }

    // Note:
    // xt has been used previously as workspace for hemmComputeX, so it has to be reset, because now it
    // will be used for accessing the broadcasted version of x
    xt.reset();
    xt.setRangeStart(at_start);
    xt.setHeight(nrefls);

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
