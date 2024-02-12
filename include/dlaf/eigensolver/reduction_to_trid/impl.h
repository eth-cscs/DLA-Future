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

#include <pika/execution.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/eigensolver/reduction_to_trid/api.h>
#include <dlaf/eigensolver/reduction_utils/misc.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/views.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/transform.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
TridiagResult1Stage<T, D> ReductionToTrid<B, D, T>::call(Matrix<T, D>& mat_a) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  using matrix::internal::MatrixRef;

  const auto dist_a = mat_a.distribution();

  // TODO complex/real: real can be one less
  const SizeType nrefls = std::max<SizeType>(0, dist_a.size().rows() - 1);

  Matrix<BaseType<T>, Device::CPU> mat_trid({dist_a.size().rows(), 2}, {dist_a.tile_size().rows(), 2});

  // Row-vector that is distributed over columns, but exists locally on all rows of the grid
  Matrix<T, Device::CPU> mat_taus(matrix::Distribution(GlobalElementSize(nrefls, 1),
                                                       TileElementSize(mat_a.blockSize().cols(), 1),
                                                       comm::Size2D(mat_a.commGridSize().cols(), 1),
                                                       comm::Index2D(mat_a.rankIndex().col(), 0),
                                                       comm::Index2D(mat_a.sourceRankIndex().col(), 0)));

  if (dist_a.size().isEmpty())
    return {std::move(mat_taus), std::move(mat_trid)};

  constexpr std::size_t n_workspaces = 2;
  const matrix::Distribution dist({mat_a.size().rows(), mat_a.tile_size().cols()}, dist_a.tile_size());
  common::RoundRobin<matrix::Panel<Coord::Col, T, D>> panels_w(n_workspaces, dist);

  for (SizeType j = 0; j < dist_a.nr_tiles().cols(); ++j) {
    auto& W = panels_w.nextResource();

    W.setRangeStart(GlobalTileIndex{j, 0});

    // TODO probably not needed
    matrix::util::set0<B>(pika::execution::thread_priority::high, W);

    // PANEL
    const bool is_last_tile = j == dist_a.nr_tiles().cols() - 1;

    const SizeType panel_width = dist_a.template tile_size_of<Coord::Col>(j);
    for (SizeType j_el_tl = 0; !is_last_tile && j_el_tl < panel_width; ++j_el_tl) {
      const SizeType j_el =
          dist_a.template global_element_from_global_tile_and_tile_element<Coord::Col>(j, j_el_tl);

      const SizeType i_el = j_el + 1;
      const SizeType i = dist_a.template global_tile_from_global_element<Coord::Row>(i_el);
      const SizeType i_el_tl = dist_a.template tile_element_from_global_element<Coord::Row>(i_el);

      // Note:
      // This is a mask for capturing all the left part of the panel, from first non-off diag of the
      // first column of this panel.
      const matrix::SubPanelView panel_uptonow(dist_a, dist_a.global_element_index({j, j}, {0, 0}),
                                               j_el_tl + 1);

      std::vector<matrix::ReadWriteTileSender<T, D>> v_panel_rw;
      v_panel_rw.reserve(to_sizet(std::distance(panel_uptonow.iteratorLocal().begin(),
                                                panel_uptonow.iteratorLocal().end())));

      std::vector<matrix::ReadOnlyTileSender<T, D>> w_panel_ro;
      w_panel_ro.reserve(to_sizet(std::distance(panel_uptonow.iteratorLocal().begin(),
                                                panel_uptonow.iteratorLocal().end())));

      for (const auto& it : panel_uptonow.iteratorLocal()) {
        const auto spec = panel_uptonow(it);
        v_panel_rw.emplace_back(splitTile(mat_a.readwrite(it), spec));
        w_panel_ro.emplace_back(splitTile(W.read(it), spec));
      }

      auto tau = ex::ensure_started(
          ex::when_all(mat_taus.readwrite(GlobalTileIndex{j, 0}),
                       mat_trid.readwrite(GlobalTileIndex{j, 0}),
                       ex::when_all_vector(std::move(w_panel_ro)),
                       ex::when_all_vector(std::move(v_panel_rw))) |
          di::transform(
              di::Policy<B>(),
              [dist_a, offset = i - j, j, i_el, i_el_tl, j_el_tl](auto&& panel_taus, auto&& panel_trid,
                                                                  auto&& w_tiles, auto&& v_tiles) -> T {
                // panel update
                if (j_el_tl > 0) {
                  const std::size_t i_first =
                      to_sizet(dist_a.template global_tile_from_global_element<Coord::Row>(i_el - 1) -
                               j);
                  const SizeType i_first_el_tl =
                      dist_a.template tile_element_from_global_element<Coord::Row>(i_el - 1);
                  for (std::size_t i = i_first; i < v_tiles.size(); ++i) {
                    // Note: this is needed because first tile has to align with mask without "diagonal"
                    const SizeType i_first_tl = (i == i_first) ? i_first_el_tl : 0;

                    // Note:
                    // Here we are going to do a matrix-vector multiplication with GEMM instead of GEMV.
                    // The reason is that we have to conjugate transpose the vector, and with the GEMV
                    // we can workaround the problem for real values using ld as gap between elements of
                    // the vector, but for complex type it does not allow to conjugate the value. So, we
                    // ended up using the more generic GEMM.

                    {
                      auto&& tile_v = v_tiles[i];
                      auto&& tile_wt_snd = w_tiles[0];
                      auto&& tile_wt = tile_wt_snd.get();
                      auto&& col_out = v_tiles[i];

                      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
                                 tile_v.size().rows() - i_first_tl, 1, j_el_tl, T(-1),
                                 tile_v.ptr({i_first_tl, 0}), tile_v.ld(), tile_wt.ptr({j_el_tl, 0}),
                                 tile_wt.ld(), T(1), col_out.ptr({i_first_tl, j_el_tl}), col_out.ld());
                    }

                    {
                      auto&& tile_w_snd = w_tiles[i];
                      auto&& tile_w = tile_w_snd.get();
                      auto&& tile_vt = v_tiles[0];
                      auto&& col_out = v_tiles[i];

                      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
                                 tile_w.size().rows() - i_first_tl, 1, j_el_tl, T(-1),
                                 tile_w.ptr({i_first_tl, 0}), tile_w.ld(), tile_vt.ptr({j_el_tl, 0}),
                                 tile_vt.ld(), T(1), col_out.ptr({i_first_tl, j_el_tl}), col_out.ld());
                    }
                  }
                }

                // compute reflector
                constexpr bool has_head = true;
                const auto tau = computeReflectorAndTau(
                    has_head, v_tiles, i_el_tl, j_el_tl,
                    computeX0AndSquares(has_head, v_tiles, i_el_tl, j_el_tl, offset), offset);
                panel_taus({j_el_tl, 0}) = tau;

                // Note: V is set in-place and off-diagonal is stored out-of-place in the result
                T& head = v_tiles[to_sizet(offset)]({i_el_tl, j_el_tl});

                const T& diag_value = v_tiles[0]({j_el_tl, j_el_tl});

                BaseType<T>& diag = panel_trid({j_el_tl, 0});
                BaseType<T>& offdiag = panel_trid({j_el_tl, 1});
                if constexpr (isComplex_v<T>) {
                  diag = diag_value.real();
                  offdiag = head.real();
                }
                else {
                  diag = diag_value;
                  offdiag = head;
                }

                head = T(1.0);

                return tau;
              }));

      // W = At @ v
      const SizeType mt = dist_a.size().rows() - i_el;
      MatrixRef<const T, D> V(mat_a, {{i_el, j_el}, {mt, 1}});
      MatrixRef<const T, D> At(mat_a, {{i_el, j_el + 1}, {mt, mt}});

      for (auto ij : common::iterate_range2d(At.nr_tiles())) {
        const LocalTileIndex ij_lc = At.distribution().local_tile_index(ij);

        // Note: this is for W which starts from the beginning
        // TODO fix for last reflector of the panel
        const SizeType i_first_tl = ij.row() == 0 ? i_el_tl : 0;
        auto&& tile_w =
            W.readwrite({to_SizeType(i) + ij_lc.row(), 0});  // TODO workaround to align At and W

        // Note: V is aligned with At, but it must be accessed with the col due to mul from right
        auto&& tile_v = V.read(GlobalTileIndex{ij.col(), 0});

        // Note: this is for managing the access "transposed" for the hemv
        const bool is_lower = ij.row() > ij.col();
        const GlobalTileIndex ij_lower = is_lower ? ij : transposed(ij);
        auto&& tile_at = At.read(ij_lower);

        ex::start_detached(ex::when_all(std::move(tile_at), std::move(tile_v), std::move(tile_w)) |
                           di::transform(di::Policy<B>(), [ij, is_lower, i_first_tl,
                                                           j_el_tl](auto&& at, auto&& v, auto&& tile_w) {
                             auto&& tile_v = v;
                             auto&& tile_a = at;

                             if (ij.row() == ij.col()) {
                               blas::hemv(blas::Layout::ColMajor, blas::Uplo::Lower,
                                          tile_a.size().rows(), T(1), tile_a.ptr(), tile_a.ld(),
                                          tile_v.ptr({0, 0}), 1, T(1), tile_w.ptr({i_first_tl, j_el_tl}),
                                          1);
                             }
                             else {
                               const blas::Op op = is_lower ? blas::Op::NoTrans : blas::Op::ConjTrans;
                               blas::gemv(blas::Layout::ColMajor, op, tile_a.size().rows(),
                                          tile_a.size().cols(), T(1), tile_a.ptr(), tile_a.ld(),
                                          tile_v.ptr({0, 0}), 1, T(1), tile_w.ptr({i_first_tl, j_el_tl}),
                                          1);
                             }
                           }));
      }

      // compute W
      std::vector<matrix::ReadOnlyTileSender<T, D>> v_panel_ro;
      v_panel_ro.reserve(to_sizet(std::distance(panel_uptonow.iteratorLocal().begin(),
                                                panel_uptonow.iteratorLocal().end())));

      std::vector<matrix::ReadWriteTileSender<T, D>> w_panel_rw;
      w_panel_rw.reserve(to_sizet(std::distance(panel_uptonow.iteratorLocal().begin(),
                                                panel_uptonow.iteratorLocal().end())));

      for (const auto& it : panel_uptonow.iteratorLocal()) {
        const auto spec = panel_uptonow(it);
        v_panel_ro.emplace_back(splitTile(mat_a.read(it), spec));
        w_panel_rw.emplace_back(splitTile(W.readwrite(it), spec));
      }

      ex::start_detached(
          ex::when_all(ex::when_all_vector(std::move(w_panel_rw)),
                       ex::when_all_vector(std::move(v_panel_ro)), std::move(tau)) |
          di::transform(di::Policy<B>(), [i_first = to_sizet(i - j), i_el_tl,
                                          j_el_tl](auto&& w_tiles, auto&& v_tiles, auto&& tau) {
            // computeW mods
            if (j_el_tl > 0) {
              auto&& w_up = w_tiles[0].ptr({0, j_el_tl});

              // w_up = W* . v
              for (std::size_t i = i_first; i < v_tiles.size(); ++i) {
                const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;

                auto&& tile_v = v_tiles[i].get();
                auto&& tile_w = w_tiles[i];

                auto&& w = tile_w.ptr({i_first_tl, 0});
                auto&& v_col = tile_v.ptr({i_first_tl, j_el_tl});

                // TODO this shouldn't be needed: probably w is computed outside of needed bounds
                const T beta = (i == i_first) ? 0 : 1;
                blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans,
                           tile_v.size().rows() - i_first_tl, j_el_tl, T(1), w, tile_w.ld(), v_col, 1,
                           beta, w_up, 1);
              }

              // w = w - V . w_up
              for (std::size_t i = i_first; i < v_tiles.size(); ++i) {
                const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;

                auto&& tile_v = v_tiles[i].get();
                auto&& tile_w = w_tiles[i];

                auto&& v = tile_v.ptr({i_first_tl, 0});
                auto&& w_col = tile_w.ptr({i_first_tl, j_el_tl});

                blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, tile_v.size().rows() - i_first_tl,
                           j_el_tl, T(-1), v, tile_v.ld(), w_up, 1, T(1), w_col, 1);
              }

              // w_up = V* . v
              for (std::size_t i = i_first; i < v_tiles.size(); ++i) {
                const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;

                auto&& tile_v = v_tiles[i].get();

                auto&& v = tile_v.ptr({i_first_tl, 0});
                auto&& v_col = tile_v.ptr({i_first_tl, j_el_tl});

                const T beta = (i == i_first) ? 0 : 1;
                blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans,
                           tile_v.size().rows() - i_first_tl, j_el_tl, T(1), v, tile_v.ld(), v_col, 1,
                           beta, w_up, 1);
              }

              // w = w - W* . w_up
              for (std::size_t i = i_first; i < v_tiles.size(); ++i) {
                const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;

                auto&& tile_w = w_tiles[i];

                auto&& w = tile_w.ptr({i_first_tl, 0});
                auto&& w_col = tile_w.ptr({i_first_tl, j_el_tl});

                blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, tile_w.size().rows() - i_first_tl,
                           j_el_tl, T(-1), w, tile_w.ld(), w_up, 1, T(1), w_col, 1);
              }
            }

            for (std::size_t i = i_first; i < w_tiles.size(); ++i) {
              const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;
              auto&& tile_w = w_tiles[i];
              blas::scal(tile_w.size().rows() - i_first_tl, tau, tile_w.ptr({i_first_tl, j_el_tl}), 1);
            }

            T alpha = 0;
            for (std::size_t i = i_first; i < w_tiles.size(); ++i) {
              const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;
              auto&& tile_w = w_tiles[i];
              auto&& tile_v_snd = v_tiles[i];
              auto&& tile_v = tile_v_snd.get();
              alpha += blas::dot(tile_w.size().rows() - i_first_tl, tile_w.ptr({i_first_tl, j_el_tl}), 1,
                                 tile_v.ptr({i_first_tl, j_el_tl}), 1);
            }
            alpha *= T(-0.5) * tau;

            for (std::size_t i = i_first; i < w_tiles.size(); ++i) {
              const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;
              auto&& tile_w = w_tiles[i];
              auto&& tile_v_snd = v_tiles[i];
              auto&& tile_v = tile_v_snd.get();
              blas::axpy(tile_w.size().rows() - i_first_tl, alpha, tile_v.ptr({i_first_tl, j_el_tl}), 1,
                         tile_w.ptr({i_first_tl, j_el_tl}), 1);
            }
          }));
    }
    if (is_last_tile && dist_a.template tile_size_of<Coord::Col>(j) != 1) {
      ex::start_detached(ex::when_all(mat_taus.readwrite(GlobalTileIndex{j, 0}),
                                      mat_trid.readwrite(GlobalTileIndex{j, 0}),
                                      mat_a.readwrite(GlobalTileIndex{j, j})) |
                         ex::then([](auto&& tile_taus, auto&& tile_tri, auto&& tile_a) {
                           // TODO create tile wrapper
                           lapack::hetrd(lapack::Uplo::Lower, tile_a.size().rows(), tile_a.ptr(),
                                         tile_a.ld(), tile_tri.ptr({0, 0}), tile_tri.ptr({0, 1}),
                                         tile_taus.ptr());
                         }));
    }

    // TRAILING MATRIX

    // no trailing matrix, the last panel computed was the last one
    if (is_last_tile)
      break;

    const GlobalElementIndex at_offset = dist_a.global_element_index({j + 1, j + 1}, {0, 0});
    const matrix::SubMatrixView at_view(dist_a, at_offset);

    auto i_v = [j](const SizeType i) -> LocalTileIndex { return {i, j}; };
    auto i_w = [](const SizeType i) -> LocalTileIndex { return {i, 0}; };  // TODO check if W is aligned

    for (const auto& ij : at_view.iteratorLocal()) {
      if (ij.row() < ij.col())
        continue;

      auto get_rw_tile_at = [&mat_a, &at_view, ij]() {
        return splitTile(mat_a.readwrite(ij), at_view(ij));  // TODO splitTile should not be needed
      };

      const auto priority = (ij.col() == j + 1) ? pika::execution::thread_priority::high
                                                : pika::execution::thread_priority::normal;

      if (ij.row() == ij.col())
        her2kDiag<B>(priority, mat_a.read(i_v(ij.row())), W.read(i_w(ij.row())), get_rw_tile_at());
      else {
        her2kOffDiag<B>(priority, mat_a.read(i_v(ij.row())), W.read(i_w(ij.col())), get_rw_tile_at());
        her2kOffDiag<B>(priority, W.read(i_w(ij.row())), mat_a.read(i_v(ij.col())), get_rw_tile_at());
      }
    }

    W.reset();
  }

  // Note: copy last diagonal element
  const SizeType j_last = dist_a.nrTiles().cols() - 1;
  ex::start_detached(ex::when_all(mat_a.read(GlobalTileIndex{j_last, j_last}),
                                  mat_trid.readwrite(GlobalTileIndex{j_last, 0})) |
                     ex::then([](auto&& a, auto&& tri) {
                       const SizeType j_tl = tri.size().rows() - 1;
                       if constexpr (isComplex_v<T>)
                         tri({j_tl, 0}) = a.get()({j_tl, j_tl}).real();
                       else
                         tri({j_tl, 0}) = a.get()({j_tl, j_tl});
                     }));

  return {std::move(mat_taus), std::move(mat_trid)};
}
}
