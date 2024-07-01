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

#include <vector>

#include <pika/barrier.hpp>

#include <dlaf/common/index2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/common/vector.h>
#include <dlaf/eigensolver/internal/get_red2band_barrier_busy_wait.h>
#include <dlaf/eigensolver/internal/get_red2band_panel_nworkers.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/views.h>
#include <dlaf/schedulers.h>

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

}

namespace red2band::local {

template <Device D, class T>
T computeReflector(const std::vector<matrix::Tile<T, D>>& panel, SizeType j) {
  constexpr bool has_head = true;

  std::array<T, 2> x0_and_squares = computeX0AndSquares(has_head, panel, j);

  auto tau = computeReflectorAndTau(has_head, panel, j, std::move(x0_and_squares));

  return tau;
}

template <class MatrixLikeA, class MatrixLikeTaus>
void computePanelReflectors(MatrixLikeA& mat_a, MatrixLikeTaus& mat_taus, const SizeType j_sub,
                            const matrix::SubPanelView& panel_view) {
  static Device constexpr D = MatrixLikeA::device;
  using T = typename MatrixLikeA::ElementType;
  using pika::execution::thread_priority;
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<matrix::ReadWriteTileSender<T, D>> panel_tiles;
  const auto panel_range = panel_view.iteratorLocal();
  const std::size_t panel_ntiles = to_sizet(std::distance(panel_range.begin(), panel_range.end()));

  // if (panel_ntiles == 0) {
  //   return;
  // }

  panel_tiles.reserve(panel_ntiles);
  for (const auto& i : panel_range) {
    const matrix::SubTileSpec& spec = panel_view(i);
    panel_tiles.emplace_back(matrix::splitTile(mat_a.readwrite(i), spec));
  }

  const std::size_t nthreads = getReductionToBandPanelNWorkers();
  ex::start_detached(
      ex::when_all(ex::just(std::make_unique<pika::barrier<>>(nthreads),
                            std::vector<common::internal::vector<T>>{}),  // w (internally required)
                   mat_taus.readwrite(LocalTileIndex(j_sub, 0)),
                   ex::when_all_vector(std::move(panel_tiles))) |
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

}
}
