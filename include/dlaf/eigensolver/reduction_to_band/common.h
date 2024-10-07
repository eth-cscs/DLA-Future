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
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <pika/barrier.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/common/vector.h>
#include <dlaf/communication/kernels/reduce.h>
#include <dlaf/eigensolver/internal/get_red2band_barrier_busy_wait.h>
#include <dlaf/eigensolver/internal/get_red2band_panel_nworkers.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/matrix/views.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform.h>

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

  common::internal::SingleThreadedBlasScope single;

  if (has_head) {
    auto& tile_v0 = *it_begin++;

    const TileElementIndex idx_x0(j, j);
    x0_and_squares[0] = tile_v0(idx_x0);

    T* reflector_ptr = tile_v0.ptr(idx_x0);
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
  if (x0_and_squares[1] == T(0))
    return T(0);

  const T norm = std::sqrt(x0_and_squares[1]);
  const T x0 = x0_and_squares[0];
  const T y = std::signbit(std::real(x0_and_squares[0])) ? norm : -norm;
  const T tau = (y - x0) / y;

  auto it_begin = panel.begin();
  auto it_end = panel.end();

  common::internal::SingleThreadedBlasScope single;

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

  common::internal::SingleThreadedBlasScope single;

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

  common::internal::SingleThreadedBlasScope single;

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
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, T(1),
                                  std::forward<ASender>(tile_a), std::forward<WSender>(tile_w), T(1),
                                  std::forward<XSender>(tile_x)) |
      tile::hemm(dlaf::internal::Policy<B>(priority, thread_stacksize::nostack)));
}

// X += op(A) * W
template <Backend B, typename ASender, typename WSender, typename XSender>
void hemmOffDiag(pika::execution::thread_priority priority, blas::Op op, ASender&& tile_a,
                 WSender&& tile_w, XSender&& tile_x) {
  using T = dlaf::internal::SenderElementType<ASender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(op, blas::Op::NoTrans, T(1), std::forward<ASender>(tile_a),
                                  std::forward<WSender>(tile_w), T(1), std::forward<XSender>(tile_x)) |
      tile::gemm(dlaf::internal::Policy<B>(priority, thread_stacksize::nostack)));
}

template <Backend B, typename VSender, typename XSender, typename ASender>
void her2kDiag(pika::execution::thread_priority priority, VSender&& tile_v, XSender&& tile_x,
               ASender&& tile_a) {
  using T = dlaf::internal::SenderElementType<VSender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Uplo::Lower, blas::Op::NoTrans, T(-1),
                                  std::forward<VSender>(tile_v), std::forward<XSender>(tile_x),
                                  BaseType<T>(1), std::forward<ASender>(tile_a)) |
      tile::her2k(dlaf::internal::Policy<B>(priority, thread_stacksize::nostack)));
}

// C -= A . B*
template <Backend B, typename ASender, typename BSender, typename CSender>
void her2kOffDiag(pika::execution::thread_priority priority, ASender&& tile_a, BSender&& tile_b,
                  CSender&& tile_c) {
  using T = dlaf::internal::SenderElementType<ASender>;
  using pika::execution::thread_stacksize;

  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::ConjTrans, T(-1),
                                  std::forward<ASender>(tile_a), std::forward<BSender>(tile_b), T(1),
                                  std::forward<CSender>(tile_c)) |
      tile::gemm(dlaf::internal::Policy<B>(priority, thread_stacksize::nostack)));
}

}

namespace red2band::local {

template <Device D, class T>
T computeReflector(const std::vector<matrix::Tile<T, D>>& panel, SizeType j) {
  constexpr bool has_head = true;

  std::array<T, 2> x0_and_squares = computeX0AndSquares(has_head, panel, j);

  auto tau = computeReflectorAndTau(has_head, panel, j, std::move(x0_and_squares));

  return tau;
}

template <class T, Device D, class MatrixLikeA>
void computePanelReflectors(MatrixLikeA& mat_a, matrix::ReadWriteTileSender<T, D> tile_tau,
                            const matrix::SubPanelView& panel_view) {
  static_assert(D == MatrixLikeA::device);
  static_assert(std::is_same_v<T, typename MatrixLikeA::ElementType>);

  using pika::execution::thread_priority;
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<matrix::ReadWriteTileSender<T, D>> panel_tiles;
  const auto panel_range = panel_view.iteratorLocal();
  const std::size_t panel_ntiles = to_sizet(std::distance(panel_range.begin(), panel_range.end()));

  if (panel_ntiles == 0) {
    return;
  }

  panel_tiles.reserve(panel_ntiles);
  for (const auto& i : panel_range) {
    const matrix::SubTileSpec& spec = panel_view(i);
    panel_tiles.emplace_back(matrix::splitTile(mat_a.readwrite(i), spec));
  }

  const std::size_t nthreads = getReductionToBandPanelNWorkers();
  ex::start_detached(
      ex::when_all(ex::just(std::make_unique<pika::barrier<>>(nthreads),
                            std::vector<common::internal::vector<T>>{}),  // w (internally required)
                   std::move(tile_tau), ex::when_all_vector(std::move(panel_tiles))) |
      ex::transfer(di::getBackendScheduler<Backend::MC>(thread_priority::high)) |
      ex::bulk(nthreads, [nthreads, cols = panel_view.cols()](const std::size_t index, auto& barrier_ptr,
                                                              auto& w, auto& taus, auto& tiles) {
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
            taus({j, 0}) = computeReflector(tiles, j);
          }

          barrier_ptr->arrive_and_wait(barrier_busy_wait);

          // STEP2a: compute w (multi-threaded)
          const SizeType pt_cols = cols - (j + 1);
          if (pt_cols == 0)
            break;
          const bool has_head = (index == 0);

          w[index] = common::internal::vector<T>(pt_cols, 0);
          computeWTrailingPanel(has_head, tiles, w[index], j, pt_cols, begin, end);
          barrier_ptr->arrive_and_wait(barrier_busy_wait);

          // STEP2b: reduce w results (single-threaded)
          if (index == 0)
            dlaf::eigensolver::internal::reduceColumnVectors(w);
          barrier_ptr->arrive_and_wait(barrier_busy_wait);

          // STEP3: update trailing panel (multi-threaded)
          updateTrailingPanel(has_head, tiles, j, w[0], taus({j, 0}), begin, end);
          barrier_ptr->arrive_and_wait(barrier_busy_wait);
        }
      }));
}

template <Backend B, Device D, class T>
void setupReflectorPanelV(bool has_head, const matrix::SubPanelView& panel_view, const SizeType nrefls,
                          matrix::Panel<Coord::Col, T, D>& v, matrix::Matrix<const T, D>& mat_a,
                          bool force_copy = false) {
  namespace ex = pika::execution::experimental;

  using pika::execution::thread_priority;
  using pika::execution::thread_stacksize;

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
    const auto p = dlaf::internal::Policy<B>(thread_priority::high, thread_stacksize::nostack);
    ex::start_detached(dlaf::internal::whenAllLift(splitTile(mat_a.read(i), spec), v.readwrite(i)) |
                       matrix::copy(p));
    ex::start_detached(dlaf::internal::whenAllLift(blas::Uplo::Upper, T(0), T(1), v.readwrite(i)) |
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
      ex::start_detached(ex::when_all(matrix::splitTile(mat_a.read(idx), spec), v.readwrite(idx)) |
                         matrix::copy(dlaf::internal::Policy<B>(thread_priority::high,
                                                                thread_stacksize::nostack)));
    else
      v.setTile(idx, matrix::splitTile(mat_a.read(idx), spec));
  }
}

template <Backend B, Device D, class T, Coord C, matrix::StoreTransposed S>
void trmmComputeW(matrix::Panel<C, T, D, S>& w, matrix::Panel<C, T, D, S>& v,
                  matrix::ReadOnlyTileSender<T, D> tile_t) {
  namespace ex = pika::execution::experimental;

  using pika::execution::thread_priority;
  using pika::execution::thread_stacksize;
  using namespace blas;

  auto it = w.iteratorLocal();

  for (const auto& index_i : it) {
    ex::start_detached(dlaf::internal::whenAllLift(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                                                   T(1), tile_t, v.read(index_i), w.readwrite(index_i)) |
                       tile::trmm3(dlaf::internal::Policy<B>(thread_priority::high,
                                                             thread_stacksize::nostack)));
  }

  if (it.empty()) {
    ex::start_detached(std::move(tile_t));
  }
}

template <Backend B, Device D, class T>
void gemmComputeW2(matrix::Matrix<T, D>& w2, matrix::Panel<Coord::Col, const T, D>& w,
                   matrix::Panel<Coord::Col, const T, D>& x) {
  using pika::execution::thread_priority;
  using pika::execution::thread_stacksize;

  namespace ex = pika::execution::experimental;

  // Note:
  // Not all ranks in the column always hold at least a tile in the panel Ai, but all ranks in
  // the column are going to participate to the reduce. For them, it is important to set the
  // partial result W2 to zero.
  ex::start_detached(w2.readwrite(LocalTileIndex(0, 0)) |
                     tile::set0(dlaf::internal::Policy<B>(thread_priority::high,
                                                          thread_stacksize::nostack)));

  using namespace blas;
  // GEMM W2 = W* . X
  for (const auto& index_tile : w.iteratorLocal())
    ex::start_detached(
        dlaf::internal::whenAllLift(Op::ConjTrans, Op::NoTrans, T(1), w.read(index_tile),
                                    x.read(index_tile), T(1), w2.readwrite(LocalTileIndex(0, 0))) |
        tile::gemm(dlaf::internal::Policy<B>(thread_priority::high, thread_stacksize::nostack)));
}

template <Backend B, Device D, class T>
void gemmUpdateX(matrix::Panel<Coord::Col, T, D>& x, matrix::Matrix<const T, D>& w2,
                 matrix::Panel<Coord::Col, const T, D>& v) {
  namespace ex = pika::execution::experimental;

  using pika::execution::thread_priority;
  using pika::execution::thread_stacksize;
  using namespace blas;

  // GEMM X = X - 0.5 . V . W2
  for (const auto& index_i : v.iteratorLocal())
    ex::start_detached(
        dlaf::internal::whenAllLift(Op::NoTrans, Op::NoTrans, T(-0.5), v.read(index_i),
                                    w2.read(LocalTileIndex(0, 0)), T(1), x.readwrite(index_i)) |
        tile::gemm(dlaf::internal::Policy<B>(thread_priority::high, thread_stacksize::nostack)));
}

}

}
