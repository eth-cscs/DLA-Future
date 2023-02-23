//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include <pika/future.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/assert.h"
#include "dlaf/common/barrier.h"
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
#include "dlaf/communication/rdma.h"
#include "dlaf/eigensolver/get_red2band_panel_nworkers.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/matrix/views.h"
#include "dlaf/schedulers.h"
#include "dlaf/sender/keep_future.h"
#include "dlaf/sender/traits.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/reduction_to_band/api.h"
#include "dlaf/factorization/qr.h"

namespace dlaf::eigensolver::internal {

// Given a vector of vectors, reduce all vectors in the first one using sum operation
template <class T>
void reduceColumnVectors(std::vector<common::internal::vector<T>>& columnVectors) {
  for (std::size_t i = 1; i < columnVectors.size(); ++i) {
    DLAF_ASSERT_HEAVY(columnVectors[0].size() == columnVectors[i].size(), columnVectors[0].size(),
                      columnVectors[i].size());
    for (SizeType j = 0; j < columnVectors[0].size(); ++j)
      columnVectors[0][j] += columnVectors[i][j];
  }
}

namespace red2band {

// Extract x0 and compute local cumulative sum of squares of the reflector column
template <Device D, class T>
std::array<T, 2> computeX0AndSquares(const bool has_head, const std::vector<matrix::Tile<T, D>>& panel,
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

template <Device D, class T>
T computeReflectorAndTau(const bool has_head, const std::vector<matrix::Tile<T, D>>& panel,
                         const SizeType j, std::array<T, 2> x0_and_squares) {
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

template <Device D, class T>
void computeWTrailingPanel(const bool has_head, const std::vector<matrix::Tile<T, D>>& panel,
                           common::internal::vector<T>& w, SizeType j, const SizeType pt_cols,
                           const std::size_t begin, const std::size_t end) {
  // for each tile in the panel, consider just the trailing panel
  // i.e. all rows (height = reflector), just columns to the right of the current reflector
  if (!(pt_cols > 0))
    return;

  const TileElementIndex index_el_x0(j, j);
  bool has_first_component = has_head;

  // W = Pt* . V
  for (auto index = begin; index < end; ++index) {
    const matrix::Tile<const T, D>& tile_a = panel[index];
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

      has_first_component = false;
    }

    if (pt_start.isIn(tile_a.size())) {
      // W += 1 . A* . V
      blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, pt_size.rows(), pt_size.cols(), T(1),
                 tile_a.ptr(pt_start), tile_a.ld(), tile_a.ptr(v_start), 1, T(1), w.data(), 1);
    }
  }
}

template <Device D, class T>
void updateTrailingPanel(const bool has_head, const std::vector<matrix::Tile<T, D>>& panel, SizeType j,
                         const std::vector<T>& w, const T tau, const std::size_t begin,
                         const std::size_t end) {
  const TileElementIndex index_el_x0(j, j);

  bool has_first_component = has_head;

  // GER Pt = Pt - tau . v . w*
  for (auto index = begin; index < end; ++index) {
    const matrix::Tile<T, D>& tile_a = panel[index];
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

      has_first_component = false;
    }

    if (pt_start.isIn(tile_a.size())) {
      // Pt = Pt - tau * v * w*
      blas::ger(blas::Layout::ColMajor, pt_size.rows(), pt_size.cols(), -dlaf::conj(tau),
                tile_a.ptr(v_start), 1, w.data(), 1, tile_a.ptr(pt_start), tile_a.ld());
    }
  }
}

template <Backend B, typename ASender, typename WSender, typename XSender>
void hemmDiag(pika::execution::thread_priority priority, ASender&& tile_a, WSender&& tile_w,
              XSender&& tile_x) {
  using T = dlaf::internal::SenderElementType<ASender>;
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, T(1),
                                  std::forward<ASender>(tile_a), std::forward<WSender>(tile_w), T(1),
                                  std::forward<XSender>(tile_x)) |
      tile::hemm(dlaf::internal::Policy<B>(priority)));
}

// X += op(A) * W
template <Backend B, typename ASender, typename WSender, typename XSender>
void hemmOffDiag(pika::execution::thread_priority priority, blas::Op op, ASender&& tile_a,
                 WSender&& tile_w, XSender&& tile_x) {
  using T = dlaf::internal::SenderElementType<ASender>;
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(op, blas::Op::NoTrans, T(1), std::forward<ASender>(tile_a),
                                  std::forward<WSender>(tile_w), T(1), std::forward<XSender>(tile_x)) |
      tile::gemm(dlaf::internal::Policy<B>(priority)));
}

template <Backend B, typename VSender, typename XSender, typename ASender>
void her2kDiag(pika::execution::thread_priority priority, VSender&& tile_v, XSender&& tile_x,
               ASender&& tile_a) {
  using T = dlaf::internal::SenderElementType<VSender>;
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Lower, blas::Op::NoTrans, T(-1),
                                  std::forward<VSender>(tile_v), std::forward<XSender>(tile_x),
                                  BaseType<T>(1), std::forward<ASender>(tile_a)) |
      tile::her2k(dlaf::internal::Policy<B>(priority)));
}

// C -= A . B*
template <Backend B, typename ASender, typename BSender, typename CSender>
void her2kOffDiag(pika::execution::thread_priority priority, ASender&& tile_a, BSender&& tile_b,
                  CSender&& tile_c) {
  using T = dlaf::internal::SenderElementType<ASender>;
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, T(-1),
                                  std::forward<ASender>(tile_a), std::forward<BSender>(tile_b), T(1),
                                  std::forward<CSender>(tile_c)) |
      tile::gemm(dlaf::internal::Policy<B>(priority)));
}

namespace local {

template <Device D, class T>
T computeReflector(const std::vector<matrix::Tile<T, D>>& panel, SizeType j) {
  constexpr bool has_head = true;

  std::array<T, 2> x0_and_squares = computeX0AndSquares(has_head, panel, j);

  auto tau = computeReflectorAndTau(has_head, panel, j, std::move(x0_and_squares));

  return tau;
}

template <Device D, class T>
void updateTrailingPanelWithReflector(const std::vector<matrix::Tile<T, D>>& panel, const SizeType j,
                                      const SizeType pt_cols, const T tau) {
  if (pt_cols == 0)
    return;

  constexpr bool has_head = true;
  common::internal::vector<T> w(pt_cols, 0);
  computeWTrailingPanel(has_head, panel, w, j, pt_cols, 0, panel.size());

  updateTrailingPanel(has_head, panel, j, w, tau, 0, panel.size());
}

template <class MatrixLike>
auto computePanelReflectors(MatrixLike& mat_a, const matrix::SubPanelView& panel_view,
                            const SizeType nrefls) {
  static Device constexpr D = MatrixLike::device;
  using T = typename MatrixLike::ElementType;
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<pika::future<matrix::Tile<T, D>>> panel_tiles;
  panel_tiles.reserve(
      to_sizet(std::distance(panel_view.iteratorLocal().begin(), panel_view.iteratorLocal().end())));
  for (const auto& i : panel_view.iteratorLocal()) {
    const matrix::SubTileSpec& spec = panel_view(i);
    panel_tiles.emplace_back(matrix::splitTile(mat_a(i), spec));
  }

  const size_t nthreads = getReductionToBandPanelNWorkers();
  return ex::when_all(ex::just(std::make_shared<barrier_t>(nthreads)),
                      ex::just(std::vector<common::internal::vector<T>>{}),  // w (interally required)
                      ex::just(common::internal::vector<T>{}),               // taus
                      ex::when_all_vector(std::move(panel_tiles))) |
         ex::transfer(di::getBackendScheduler<Backend::MC>(pika::execution::thread_priority::high)) |
         ex::bulk(nthreads,
                  [nthreads, nrefls, cols = panel_view.cols()](const std::size_t index,
                                                               auto& barrier_ptr, auto& w, auto& taus,
                                                               auto& tiles) {
                    const std::size_t batch_size = util::ceilDiv(tiles.size(), nthreads);
                    const std::size_t begin = index * batch_size;
                    const std::size_t end = std::min(index * batch_size + batch_size, tiles.size());

                    if (index == 0) {
                      taus.reserve(nrefls);
                      w.resize(nthreads);
                    }

                    for (SizeType j = 0; j < nrefls; ++j) {
                      // STEP1: compute tau and reflector (single-thread)
                      if (index == 0)
                        taus.emplace_back(computeReflector(tiles, j));
                      barrier_ptr->arrive_and_wait();

                      // STEP2a: compute w (multi-threaded)
                      const SizeType pt_cols = cols - (j + 1);
                      if (pt_cols == 0)
                        break;
                      const bool has_head = (index == 0);

                      w[index] = common::internal::vector<T>(pt_cols, 0);
                      computeWTrailingPanel(has_head, tiles, w[index], j, pt_cols, begin, end);
                      barrier_ptr->arrive_and_wait();

                      // STEP2b: reduce w results (single-threaded)
                      if (index == 0)
                        dlaf::eigensolver::internal::reduceColumnVectors(w);
                      barrier_ptr->arrive_and_wait();

                      // STEP3: update trailing panel (multi-threaded)
                      updateTrailingPanel(has_head, tiles, j, w[0], taus.back(), begin, end);
                      barrier_ptr->arrive_and_wait();
                    }
                  }) |
         ex::then([](auto barrier_ptr, auto w, auto taus, auto tiles) {
           di::silenceUnusedWarningFor(barrier_ptr, w, tiles);
           return taus;
         }) |
         ex::make_future();
}

template <Backend B, Device D, class T>
void setupReflectorPanelV(bool has_head, const matrix::SubPanelView& panel_view, const SizeType nrefls,
                          matrix::Panel<Coord::Col, T, D>& v, matrix::Matrix<const T, D>& mat_a,
                          bool force_copy = false) {
  namespace ex = pika::execution::experimental;

  using dlaf::internal::keepFuture;
  using pika::execution::thread_priority;

  // Note:
  // Reflectors are stored in the lower triangular part of the A matrix leading to sharing memory
  // between reflectors and results, which are in the upper triangular part. The problem exists only
  // for the first tile (of the V, i.e. band excluded). Since refelectors will be used in next
  // computations, they should be well-formed, i.e. a unit lower trapezoidal matrix. For this reason,
  // a support tile is used, where just the reflectors values are copied, the diagonal is set to 1
  // and the rest is zeroed out.
  auto it_begin = panel_view.iteratorLocal().begin();
  auto it_end = panel_view.iteratorLocal().end();

  if (has_head) {
    const LocalTileIndex i = *it_begin;
    matrix::SubTileSpec spec = panel_view(i);

    // Note:
    // If the number of reflectors are limited by height (|reflector| > 1), the panel is narrower than
    // the blocksize, leading to just using a part of A (first full nrefls columns)
    spec.size = {spec.size.rows(), std::min(nrefls, spec.size.cols())};

    // Note:
    // copy + laset is done in two independent tasks, but it could be theoretically merged to into a
    // single task doing both.
    const auto p = dlaf::internal::Policy<B>(thread_priority::high);
    ex::start_detached(
        dlaf::internal::whenAllLift(keepFuture(splitTile(mat_a.read(i), spec)), v.readwrite_sender(i)) |
        matrix::copy(p));
    ex::start_detached(
        dlaf::internal::whenAllLift(blas::Uplo::Upper, T(0), T(1), v.readwrite_sender(i)) |
        tile::laset(p));

    ++it_begin;
  }

  // The rest of the V panel of reflectors can just point to the values in A, since they are
  // well formed in-place.
  for (auto it = it_begin; it < it_end; ++it) {
    const LocalTileIndex idx = *it;
    const matrix::SubTileSpec& spec = panel_view(idx);

    // Note:  This is a workaround for the deadlock problem with sub-tiles.
    //        Without this copy, during matrix update the same tile would get accessed at the same
    //        time both in readonly mode (for reflectors) and in readwrite mode (for updating the
    //        matrix). This would result in a deadlock, so instead of linking the panel to an external
    //        tile, memory provided internally by the panel is used as support. In this way, the two
    //        subtiles used in the operation belong to different tiles.
    if (force_copy)
      ex::start_detached(
          ex::when_all(keepFuture(matrix::splitTile(mat_a.read(idx), spec)), v.readwrite_sender(idx)) |
          matrix::copy(dlaf::internal::Policy<B>(thread_priority::high)));
    else
      v.setTile(idx, matrix::splitTile(mat_a.read(idx), spec));
  }
}

template <Backend B, Device D, class T>
void trmmComputeW(matrix::Panel<Coord::Col, T, D>& w, matrix::Panel<Coord::Col, T, D>& v,
                  pika::shared_future<matrix::Tile<const T, D>> tile_t) {
  namespace ex = pika::execution::experimental;

  using dlaf::internal::keepFuture;

  using pika::execution::thread_priority;
  using namespace blas;

  for (const auto& index_i : w.iteratorLocal())
    ex::start_detached(dlaf::internal::whenAllLift(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                                                   T(1), keepFuture(tile_t), v.read_sender(index_i),
                                                   w.readwrite_sender(index_i)) |
                       tile::trmm3(dlaf::internal::Policy<B>(thread_priority::high)));
}

template <Backend B, Device D, class T>
void gemmUpdateX(matrix::Panel<Coord::Col, T, D>& x, matrix::Matrix<const T, D>& w2,
                 matrix::Panel<Coord::Col, const T, D>& v) {
  namespace ex = pika::execution::experimental;

  using pika::execution::thread_priority;
  using namespace blas;

  // GEMM X = X - 0.5 . V . W2
  for (const auto& index_i : v.iteratorLocal())
    ex::start_detached(dlaf::internal::whenAllLift(Op::NoTrans, Op::NoTrans, T(-0.5),
                                                   v.read_sender(index_i),
                                                   w2.read_sender(LocalTileIndex(0, 0)), T(1),
                                                   x.readwrite_sender(index_i)) |
                       tile::gemm(dlaf::internal::Policy<B>(thread_priority::high)));
}

template <Backend B, Device D, class T>
void hemmComputeX(matrix::Panel<Coord::Col, T, D>& x, const matrix::SubMatrixView& view,
                  matrix::Matrix<const T, D>& a, matrix::Panel<Coord::Col, const T, D>& w) {
  namespace ex = pika::execution::experimental;

  using dlaf::internal::keepFuture;
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
        hemmDiag<B>(thread_priority::high, keepFuture(tile_a), w.read_sender(ij),
                    x.readwrite_sender(ij));
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
          hemmOffDiag<B>(thread_priority::high, blas::Op::NoTrans, keepFuture(tile_a),
                         w.read_sender(index_w), x.readwrite_sender(index_x));
        }

        {
          const LocalTileIndex index_pretended = transposed(ij);
          const LocalTileIndex index_x(Coord::Row, index_pretended.row());
          const LocalTileIndex index_w(Coord::Row, index_pretended.col());
          hemmOffDiag<B>(thread_priority::high, blas::Op::ConjTrans, keepFuture(tile_a),
                         w.read_sender(index_w), x.readwrite_sender(index_x));
        }
      }
    }
  }
}

template <Backend B, Device D, class T>
void gemmComputeW2(matrix::Matrix<T, D>& w2, matrix::Panel<Coord::Col, const T, D>& w,
                   matrix::Panel<Coord::Col, const T, D>& x) {
  using pika::execution::thread_priority;

  namespace ex = pika::execution::experimental;

  // Note:
  // Not all ranks in the column always hold at least a tile in the panel Ai, but all ranks in
  // the column are going to participate to the reduce. For them, it is important to set the
  // partial result W2 to zero.
  ex::start_detached(w2.readwrite_sender(LocalTileIndex(0, 0)) |
                     tile::set0(dlaf::internal::Policy<B>(thread_priority::high)));

  using namespace blas;
  // GEMM W2 = W* . X
  for (const auto& index_tile : w.iteratorLocal())
    ex::start_detached(dlaf::internal::whenAllLift(Op::ConjTrans, Op::NoTrans, T(1),
                                                   w.read_sender(index_tile), x.read_sender(index_tile),
                                                   T(1), w2.readwrite_sender(LocalTileIndex(0, 0))) |
                       tile::gemm(dlaf::internal::Policy<B>(thread_priority::high)));
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

      auto tile_a = a(ij_local);
      auto getSubA = [&view, ij_local](auto& tile_a) { return splitTile(tile_a, view(ij_local)); };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority = (j == at_start.col()) ? thread_priority::high : thread_priority::normal;

      if (is_diagonal_tile) {
        her2kDiag<B>(priority, v.read_sender(ij_local), x.read_sender(ij_local), getSubA(tile_a));
      }
      else {
        // A -= X . V*
        her2kOffDiag<B>(priority, x.read_sender(ij_local), v.read_sender(transposed(ij_local)),
                        getSubA(tile_a));

        // A -= V . X*
        her2kOffDiag<B>(priority, v.read_sender(ij_local), x.read_sender(transposed(ij_local)),
                        getSubA(tile_a));
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

template <Device D, class T>
void updateTrailingPanelWithReflector(const bool has_head, comm::Communicator& communicator,
                                      const std::vector<matrix::Tile<T, D>>& panel, SizeType j,
                                      const SizeType pt_cols, const T tau) {
  if (pt_cols == 0)
    return;

  common::internal::vector<T> w(pt_cols, 0);
  computeWTrailingPanel(has_head, panel, w, j, pt_cols, 0, panel.size());

  comm::sync::allReduceInPlace(communicator, MPI_SUM, common::make_data(w.data(), pt_cols));

  updateTrailingPanel(has_head, panel, j, w, tau, 0, panel.size());
}

template <class MatrixLike, class TriggerSender, class CommSender>
auto computePanelReflectors(TriggerSender&& trigger, comm::IndexT_MPI rank_v0,
                            CommSender&& mpi_col_chain_panel, MatrixLike& mat_a,
                            const matrix::SubPanelView& panel_view, const SizeType nrefls) {
  static Device constexpr D = MatrixLike::device;
  using T = typename MatrixLike::ElementType;
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<pika::future<matrix::Tile<T, D>>> panel_tiles;
  panel_tiles.reserve(
      to_sizet(std::distance(panel_view.iteratorLocal().begin(), panel_view.iteratorLocal().end())));
  for (const auto& i : panel_view.iteratorLocal()) {
    const matrix::SubTileSpec& spec = panel_view(i);
    panel_tiles.emplace_back(matrix::splitTile(mat_a(i), spec));
  }

  const size_t nthreads = getReductionToBandPanelNWorkers();
  return ex::when_all(ex::just(std::make_shared<barrier_t>(nthreads)),
                      ex::just(std::vector<common::internal::vector<T>>{}),  // w (interally required)
                      ex::just(common::internal::vector<T>{}),               // taus
                      ex::when_all_vector(std::move(panel_tiles)),
                      std::forward<CommSender>(mpi_col_chain_panel),
                      std::forward<TriggerSender>(trigger)) |
         ex::transfer(di::getBackendScheduler<Backend::MC>(pika::execution::thread_priority::high)) |
         ex::bulk(nthreads,
                  [nthreads, nrefls, rank_v0,
                   cols = panel_view.cols()](const std::size_t index, auto& barrier_ptr, auto& w,
                                             auto& taus, auto& tiles, auto&& pcomm) {
                    const bool rankHasHead = rank_v0 == pcomm.ref().rank();

                    const std::size_t batch_size = util::ceilDiv(tiles.size(), nthreads);
                    const std::size_t begin = index * batch_size;
                    const std::size_t end = std::min(index * batch_size + batch_size, tiles.size());

                    if (index == 0) {
                      taus.reserve(nrefls);
                      w.resize(nthreads);
                    }

                    for (SizeType j = 0; j < nrefls; ++j) {
                      // STEP1: compute tau and reflector (single-thread)
                      if (index == 0) {
                        const bool has_head = rankHasHead;
                        taus.emplace_back(computeReflector(has_head, pcomm.ref(), tiles, j));
                      }
                      barrier_ptr->arrive_and_wait();

                      // STEP2a: compute w (multi-threaded)
                      const SizeType pt_cols = cols - (j + 1);
                      if (pt_cols == 0)
                        break;

                      const bool has_head = rankHasHead && (index == 0);

                      w[index] = common::internal::vector<T>(pt_cols, 0);
                      computeWTrailingPanel(has_head, tiles, w[index], j, pt_cols, begin, end);
                      barrier_ptr->arrive_and_wait();

                      // STEP2b: reduce w results (single-threaded)
                      if (index == 0) {
                        dlaf::eigensolver::internal::reduceColumnVectors(w);
                        comm::sync::allReduceInPlace(pcomm.ref(), MPI_SUM,
                                                     common::make_data(w[0].data(), pt_cols));
                      }
                      barrier_ptr->arrive_and_wait();

                      // STEP3: update trailing panel (multi-threaded)
                      updateTrailingPanel(has_head, tiles, j, w[0], taus.back(), begin, end);
                      barrier_ptr->arrive_and_wait();
                    }
                  }) |
         ex::then([](auto barrier_ptr, auto w, auto taus, auto tiles, auto pcomm) {
           di::silenceUnusedWarningFor(barrier_ptr, w, tiles, pcomm);
           return taus;
         }) |
         ex::make_future();
}

template <Backend B, Device D, class T>
void hemmComputeX(comm::IndexT_MPI reducer_col, matrix::Panel<Coord::Col, T, D>& x,
                  matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>& xt,
                  const matrix::SubMatrixView& view, matrix::Matrix<const T, D>& a,
                  matrix::Panel<Coord::Col, const T, D>& w,
                  matrix::Panel<Coord::Row, const T, D, matrix::StoreTransposed::Yes>& wt,
                  common::Pipeline<comm::Communicator>& mpi_row_chain,
                  common::Pipeline<comm::Communicator>& mpi_col_chain) {
  namespace ex = pika::execution::experimental;

  using dlaf::internal::keepFuture;
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

      const auto& tile_a = splitTile(a.read(ij), view(ij_local));

      if (is_diagonal_tile) {
        hemmDiag<B>(thread_priority::high, keepFuture(tile_a), w.read_sender(ij_local),
                    x.readwrite_sender(ij_local));
      }
      else {
        // Note:
        // Since it is not a diagonal tile, otherwise it would have been managed in the previous
        // branch, the second operand is not available in W but it is accessible through the
        // support panel Wt.
        // However, since we are still computing the "straight" part, the result can be stored
        // in the "local" panel X.
        hemmOffDiag<B>(thread_priority::high, blas::Op::NoTrans, keepFuture(tile_a),
                       wt.read_sender(ij_local), x.readwrite_sender(ij_local));

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

        auto tile_x = (dist.rankIndex().row() == owner) ? x.readwrite_sender(index_x)
                                                        : xt.readwrite_sender(index_xt);

        hemmOffDiag<B>(thread_priority::high, blas::Op::ConjTrans, keepFuture(tile_a),
                       w.read_sender(ij_local), std::move(tile_x));
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
      ex::start_detached(
          comm::scheduleReduceRecvInPlace(mpi_col_chain(), MPI_SUM,
                                          ex::make_unique_any_sender(x.readwrite_sender({i, 0}))));
    }
    else {
      ex::start_detached(comm::scheduleReduceSend(mpi_col_chain(), rank_owner_row, MPI_SUM,
                                                  ex::make_unique_any_sender(xt.read_sender(index_xt))));
    }
  }

  // Note:
  // At this point partial results are all collected in X (Xt has been embedded in previous step),
  // so the last step needed is to reduce these last partial results in the final results.
  // The result is needed just on the column with reflectors.
  for (const auto& index_x : x.iteratorLocal()) {
    if (reducer_col == rank.col())
      ex::start_detached(
          comm::scheduleReduceRecvInPlace(mpi_row_chain(), MPI_SUM,
                                          ex::make_unique_any_sender(x.readwrite_sender(index_x))));
    else
      ex::start_detached(comm::scheduleReduceSend(mpi_row_chain(), reducer_col, MPI_SUM,
                                                  ex::make_unique_any_sender(x.read_sender(index_x))));
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

      auto tile_a = a(ij_local);
      auto getSubA = [&view, ij_local](auto& tile_a) { return splitTile(tile_a, view(ij_local)); };

      // The first column of the trailing matrix (except for the very first global tile) has to be
      // updated first, in order to unlock the next iteration as soon as possible.
      const auto priority = (j == at_start.col()) ? thread_priority::high : thread_priority::normal;

      if (is_diagonal_tile) {
        her2kDiag<B>(priority, v.read_sender(ij_local), x.read_sender(ij_local), getSubA(tile_a));
      }
      else {
        // A -= X . V*
        her2kOffDiag<B>(priority, x.read_sender(ij_local), vt.read_sender(ij_local), getSubA(tile_a));

        // A -= V . X*
        her2kOffDiag<B>(priority, v.read_sender(ij_local), xt.read_sender(ij_local), getSubA(tile_a));
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

  auto call(Matrix<T, Device::CPU>& mat_a, const matrix::SubPanelView& panel_view,
            const SizeType nrefls_block) {
    using red2band::local::computePanelReflectors;
    return computePanelReflectors(mat_a, panel_view, nrefls_block);
  }

  template <Device D, class CommSender, class TriggerSender>
  auto call(TriggerSender&& trigger, comm::IndexT_MPI rank_v0, CommSender&& mpi_col_chain_panel,
            Matrix<T, D>& mat_a, const matrix::SubPanelView& panel_view, SizeType nrefls) {
    using red2band::distributed::computePanelReflectors;
    return computePanelReflectors(std::forward<TriggerSender>(trigger), rank_v0,
                                  std::forward<CommSender>(mpi_col_chain_panel), mat_a, panel_view,
                                  nrefls);
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct ComputePanelHelper<Backend::GPU, Device::GPU, T> {
  ComputePanelHelper(const std::size_t n_workspaces, matrix::Distribution dist_a)
      : panels_v(n_workspaces, dist_a) {}

  auto call(Matrix<T, Device::GPU>& mat_a, const matrix::SubPanelView& panel_view,
            const SizeType nrefls_block) {
    using red2band::local::computePanelReflectors;

    namespace ex = pika::execution::experimental;

    // Note:
    // - copy panel_view from GPU to CPU
    // - computePanelReflectors on CPU (on a matrix like, with just a panel)
    // - copy back matrix "panel" from CPU to GPU

    auto& v = panels_v.nextResource();

    copyToCPU(panel_view, mat_a, v);
    auto taus = computePanelReflectors(v, panel_view, nrefls_block);
    copyFromCPU(panel_view, v, mat_a);

    return taus;
  }

  template <Device D, class CommSender, class TriggerSender>
  auto call(TriggerSender&& trigger, comm::IndexT_MPI rank_v0, CommSender&& mpi_col_chain_panel,
            Matrix<T, D>& mat_a, const matrix::SubPanelView& panel_view, const SizeType nrefls) {
    auto& v = panels_v.nextResource();

    // copy to CPU
    copyToCPU(panel_view, mat_a, v);

    // compute on CPU
    using dlaf::eigensolver::internal::red2band::distributed::computePanelReflectors;
    auto taus =
        computePanelReflectors(std::forward<TriggerSender>(trigger), rank_v0,
                               std::forward<CommSender>(mpi_col_chain_panel), v, panel_view, nrefls);

    // copy back to GPU
    copyFromCPU(panel_view, v, mat_a);

    return taus;
  }

protected:
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panels_v;

  void copyToCPU(const matrix::SubPanelView panel_view, matrix::Matrix<T, Device::GPU>& mat_a,
                 matrix::Panel<Coord::Col, T, Device::CPU>& v) {
    namespace ex = pika::execution::experimental;

    using dlaf::internal::keepFuture;
    using dlaf::internal::Policy;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;

    for (const auto& i : panel_view.iteratorLocal()) {
      auto spec = panel_view(i);
      auto tmp_tile = v.readwrite_sender(i);
      ex::start_detached(
          ex::when_all(keepFuture(splitTile(mat_a.read(i), spec)), splitTile(tmp_tile, spec)) |
          matrix::copy(Policy<CopyBackend_v<Device::GPU, Device::CPU>>(thread_priority::high)));
    }
  }

  void copyFromCPU(const matrix::SubPanelView panel_view, matrix::Panel<Coord::Col, T, Device::CPU>& v,
                   matrix::Matrix<T, Device::GPU>& mat_a) {
    namespace ex = pika::execution::experimental;

    using dlaf::internal::keepFuture;
    using dlaf::internal::Policy;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;

    for (const auto& i : panel_view.iteratorLocal()) {
      auto spec = panel_view(i);
      auto tile_a = mat_a.readwrite_sender(i);
      ex::start_detached(
          ex::when_all(keepFuture(splitTile(v.read(i), spec)), splitTile(tile_a, spec)) |
          matrix::copy(Policy<CopyBackend_v<Device::CPU, Device::GPU>>(thread_priority::high)));
    }
  }
};
#endif

}

/// Local implementation of reduction to band
/// @return a vector of shared futures of vectors, where each inner vector contains a block of taus
template <Backend B, Device D, class T>
common::internal::vector<pika::shared_future<common::internal::vector<T>>> ReductionToBand<B, D, T>::call(
    Matrix<T, D>& mat_a, const SizeType band_size) {
  using dlaf::matrix::Matrix;
  using dlaf::matrix::Panel;

  using namespace red2band::local;

  using common::iterate_range2d;
  using factorization::internal::computeTFactor;

  const auto dist_a = mat_a.distribution();
  const matrix::Distribution dist({mat_a.size().rows(), band_size},
                                  {dist_a.blockSize().rows(), band_size});

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real nor complex)
  const SizeType nrefls = std::max<SizeType>(0, dist_a.size().rows() - band_size - 1);

  common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus;

  if (nrefls == 0)
    return taus;

  const SizeType nblocks = (nrefls - 1) / band_size + 1;
  taus.reserve(nblocks);

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

  for (SizeType j_sub = 0; j_sub < nblocks; ++j_sub) {
    const auto i_sub = j_sub + 1;

    const GlobalElementIndex ij_offset(i_sub * band_size, j_sub * band_size);

    const SizeType nrefls_block = [=]() {
      const bool is_last = j_sub == nblocks - 1;
      if (!is_last)
        return band_size;

      const SizeType nrefls_last = nrefls % band_size;
      return nrefls_last == 0 ? band_size : nrefls_last;
    }();

    const bool isPanelIncomplete = (nrefls_block != band_size);

    // Note: if this is running, it must have at least one valid reflector (i.e. with size > 1)
    DLAF_ASSERT_HEAVY(nrefls_block != 0, nrefls_block);

    // Note:  SubPanelView is (at most) band_size wide, but it may contain a smaller number of
    //        reflectors (i.e. at the end when last reflector size is 1)
    const matrix::SubPanelView panel_view(dist_a, ij_offset, band_size);

    Panel<Coord::Col, T, D>& v = panels_v.nextResource();
    v.setRangeStart(ij_offset);
    if (isPanelIncomplete)
      v.setWidth(nrefls_block);

    // PANEL
    taus.emplace_back(compute_panel_helper.call(mat_a, panel_view, nrefls_block));

    // Note:
    // - has_reflector_head tells if this rank owns the first tile of the panel (being local, always true)
    // - if !is_full_band it has to force copy as a workaround, otherwise in update matrix it would deadlock
    // due to tile shared between panel and trailing matrix
    constexpr bool has_reflector_head = true;
    setupReflectorPanelV<B, D, T>(has_reflector_head, panel_view, nrefls_block, v, mat_a, !is_full_band);

    const LocalTileIndex t_idx(0, 0);
    // TODO used just by the column, maybe we can re-use a panel tile?
    // TODO probably the first one in any panel is ok?
    Matrix<T, D> t({nrefls_block, nrefls_block}, dist.blockSize());

    computeTFactor<B>(v, taus.back(), t.readwrite_sender(t_idx));

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
      w.setWidth(nrefls_block);

    trmmComputeW<B>(w, v, t.read(t_idx));

    // X = At . W
    Panel<Coord::Col, T, D>& x = panels_x.nextResource();
    x.setRangeStart(at_offset);
    if (isPanelIncomplete)
      x.setWidth(nrefls_block);

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

  return taus;
}

/// Distributed implementation of reduction to band
/// @return a vector of shared futures of vectors, where each inner vector contains a block of taus
template <Backend B, Device D, class T>
common::internal::vector<pika::shared_future<common::internal::vector<T>>> ReductionToBand<B, D, T>::call(
    comm::CommunicatorGrid grid, Matrix<T, D>& mat_a, const SizeType band_size) {
  using namespace red2band::distributed;

  using common::iterate_range2d;
  using factorization::internal::computeTFactor;

  namespace ex = pika::execution::experimental;

  // Note:
  // This is a temporary workaround.
  // See issue https://github.com/eth-cscs/DLA-Future/issues/729
  pika::threads::get_thread_manager().wait();

  common::Pipeline<comm::Communicator> mpi_col_chain_panel(grid.colCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_row_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_chain(grid.colCommunicator().clone());

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  // Note:
  // Reflector of size = 1 is not considered whatever T is (i.e. neither real nor complex)
  const SizeType nrefls = std::max<SizeType>(0, dist.size().rows() - band_size - 1);

  common::internal::vector<pika::shared_future<common::internal::vector<T>>> taus;

  if (nrefls == 0)
    return taus;

  const SizeType nblocks = (nrefls - 1) / band_size + 1;
  taus.reserve(nblocks);

  const bool is_full_band = (band_size == dist.blockSize().cols());

  constexpr std::size_t n_workspaces = 2;
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_v(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>>
      panels_vt(n_workspaces, dist);

  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_w(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>>
      panels_wt(n_workspaces, dist);

  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_x(n_workspaces, dist);
  common::RoundRobin<matrix::Panel<Coord::Row, T, D, matrix::StoreTransposed::Yes>>
      panels_xt(n_workspaces, dist);

  red2band::ComputePanelHelper<B, D, T> compute_panel_helper(n_workspaces, dist);

  ex::unique_any_sender<> trigger_panel{ex::just()};
  for (SizeType j_sub = 0; j_sub < nblocks; ++j_sub) {
    const SizeType i_sub = j_sub + 1;

    const GlobalElementIndex ij_offset(i_sub * band_size, j_sub * band_size);
    const GlobalElementIndex at_offset(i_sub * band_size, (j_sub + 1) * band_size);

    const comm::Index2D rank_v0{
        dist.template rankGlobalElement<Coord::Row>(ij_offset.row()),
        dist.template rankGlobalElement<Coord::Col>(ij_offset.col()),
    };

    const bool is_panel_rank_col = rank_v0.col() == rank.col();

    const SizeType nrefls_block = [=]() {
      const bool is_last = j_sub == nblocks - 1;
      if (!is_last)
        return band_size;

      const SizeType nrefls_last = nrefls % band_size;
      return nrefls_last == 0 ? band_size : nrefls_last;
    }();

    if (nrefls_block == 0)
      break;

    auto& v = panels_v.nextResource();
    auto& vt = panels_vt.nextResource();

    v.setRangeStart(at_offset);
    vt.setRangeStart(at_offset);

    v.setWidth(nrefls_block);
    vt.setHeight(nrefls_block);

    const LocalTileIndex t_idx(0, 0);
    // TODO used just by the column, maybe we can re-use a panel tile?
    // TODO or we can keep just the sh_future and allocate just inside if (is_panel_rank_col)
    matrix::Matrix<T, D> t({nrefls_block, nrefls_block}, dist.blockSize());

    // PANEL
    const matrix::SubPanelView panel_view(dist, ij_offset, band_size);

    if (is_panel_rank_col) {
      taus.emplace_back(compute_panel_helper.call(std::move(trigger_panel), rank_v0.row(),
                                                  mpi_col_chain_panel(), mat_a, panel_view,
                                                  nrefls_block));

      // Note:
      // - has_reflector_head tells if this rank owns the first tile of the panel
      // - if !is_full_band it has to force copy as a workaround, otherwise in update matrix it would
      // deadlock due to tile shared between panel and trailing matrix
      red2band::local::setupReflectorPanelV<B, D, T>(rank.row() == rank_v0.row(), panel_view,
                                                     nrefls_block, v, mat_a, !is_full_band);
      computeTFactor<B>(v, taus.back(), t.readwrite_sender(t_idx), mpi_col_chain);
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

    w.setWidth(nrefls_block);
    wt.setHeight(nrefls_block);

    if (is_panel_rank_col)
      red2band::local::trmmComputeW<B, D>(w, v, t.read(t_idx));

    comm::broadcast(rank_v0.col(), w, wt, mpi_row_chain, mpi_col_chain);

    // X = At . W
    auto& x = panels_x.nextResource();
    auto& xt = panels_xt.nextResource();

    x.setRangeStart(at_offset);
    xt.setRangeStart(at_offset);

    x.setWidth(nrefls_block);
    xt.setHeight(nrefls_block);

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
      ex::start_detached(comm::scheduleAllReduceInPlace(mpi_col_chain(), MPI_SUM,
                                                        ex::make_unique_any_sender(
                                                            w2.readwrite_sender(LocalTileIndex(0, 0)))));

      red2band::local::gemmUpdateX<B, D>(x, w2, v);
    }

    // Note:
    // xt has been used previously as workspace for hemmComputeX, so it has to be reset, because now it
    // will be used for accessing the broadcasted version of x
    xt.reset();
    xt.setRangeStart(at_offset);
    xt.setHeight(nrefls_block);

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
    // a buffer more suitable for the communication (e.g. GPU -> CPU if RDMA is not available).
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
            ex::start_detached(comm::scheduleSendBcast(ex::make_unique_any_sender(mpi_col_chain()),
                                                       ex::make_unique_any_sender(xt.read_sender(at))));
        }
        else {
          if (dist.commGridSize().rows() > 1)
            ex::start_detached(
                comm::scheduleRecvBcast(ex::make_unique_any_sender(mpi_col_chain()), owner,
                                        ex::make_unique_any_sender(xt.readwrite_sender(at))));
        }
      }

      if constexpr (dlaf::comm::CommunicationDevice_v<D> == D) {
        // Note:
        // if there is no need for additional buffers, we can just wait that xt[0] is ready for
        // reading.
        if (rank.row() == rank_v0.row()) {
          trigger_panel = xt.read_sender(at) | ex::drop_value();
        }
        else {
          // Note:
          // Conservatively ensure that xt[0] needed for updating the first column has been
          // received. Just wait for xt because communication of x happens over rows, while the
          // pivot rank can just block rank in the same column.
          trigger_panel = xt.read_sender(at) | ex::drop_value();
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
          trigger_panel = x.readwrite_sender(at) | ex::drop_value();
        }
        else {
          // Note:
          // Conservatively ensure that xt[0] needed for updating the first column has been
          // received. Just wait for xt because communication of x happens over rows, while the
          // pivot rank can just block rank in the same column.
          trigger_panel = xt.read_sender(at) | ex::drop_value();
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

  return taus;
}
}
