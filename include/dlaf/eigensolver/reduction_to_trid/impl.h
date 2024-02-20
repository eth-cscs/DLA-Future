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
struct Helper;

template <class T>
struct Helper<Backend::MC, Device::CPU, T> {
  static constexpr Backend B = Backend::MC;
  static constexpr Device D = Device::CPU;

  Helper(const std::size_t, const matrix::Distribution&) {}

  void align(const GlobalTileIndex&) {}
  void reset() {}

  template <class SenderTau, class SenderTrid, class MatrixLike>
  auto call(const matrix::Distribution& dist_a, const GlobalTileIndex& ij, const SizeType i_el,
            const TileElementIndex ij_el_tl, SenderTau&& taus, SenderTrid&& trid,
            const matrix::SubPanelView& panel_uptonow, matrix::Panel<Coord::Col, const T, D>& W,
            MatrixLike& mat_a) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    const auto i = ij.row();
    const auto j = ij.col();

    const std::size_t ntiles = to_sizet(std::distance(panel_uptonow.iteratorLocal().begin(),
                                                      panel_uptonow.iteratorLocal().end()));

    std::vector<matrix::ReadWriteTileSender<T, D>> v_panel_rw;
    v_panel_rw.reserve(ntiles);

    std::vector<matrix::ReadOnlyTileSender<T, D>> w_panel_ro;
    w_panel_ro.reserve(ntiles);

    for (const auto& it : panel_uptonow.iteratorLocal()) {
      const auto spec = panel_uptonow(it);
      v_panel_rw.emplace_back(splitTile(mat_a.readwrite(it), spec));
      w_panel_ro.emplace_back(splitTile(W.read(it), spec));
    }

    auto kernelUpdateVandComputeTau = [dist_a, offset = i - j, j, i_el, i_el_tl = ij_el_tl.row(),
                                       j_el_tl = ij_el_tl.col()](auto&& panel_taus, auto&& panel_trid,
                                                                 auto&& w_tiles, auto&& v_tiles) -> T {
      // panel update
      if (j_el_tl > 0) {
        const std::size_t i_first =
            to_sizet(dist_a.template global_tile_from_global_element<Coord::Row>(i_el - 1) - j);
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
            auto&& tile_wt = w_tiles[0].get();
            auto&& col_out = v_tiles[i];

            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
                       tile_v.size().rows() - i_first_tl, 1, j_el_tl, T(-1), tile_v.ptr({i_first_tl, 0}),
                       tile_v.ld(), tile_wt.ptr({j_el_tl, 0}), tile_wt.ld(), T(1),
                       col_out.ptr({i_first_tl, j_el_tl}), col_out.ld());
          }

          {
            auto&& tile_w = w_tiles[i].get();
            auto&& tile_vt = v_tiles[0];
            auto&& col_out = v_tiles[i];

            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
                       tile_w.size().rows() - i_first_tl, 1, j_el_tl, T(-1), tile_w.ptr({i_first_tl, 0}),
                       tile_w.ld(), tile_vt.ptr({j_el_tl, 0}), tile_vt.ld(), T(1),
                       col_out.ptr({i_first_tl, j_el_tl}), col_out.ld());
          }
        }
      }

      // compute reflector
      constexpr bool has_head = true;
      const auto tau =
          computeReflectorAndTau(has_head, v_tiles, i_el_tl, j_el_tl,
                                 computeX0AndSquares(has_head, v_tiles, i_el_tl, j_el_tl, offset),
                                 offset);
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
    };

    return ex::ensure_started(ex::when_all(std::forward<SenderTau>(taus), std::forward<SenderTrid>(trid),
                                           ex::when_all_vector(std::move(w_panel_ro)),
                                           ex::when_all_vector(std::move(v_panel_rw))) |
                              di::transform(di::Policy<B>(), std::move(kernelUpdateVandComputeTau)));
  }

  template <class SenderTau, class MatrixLike>
  void setupW(const GlobalTileIndex& ij, const TileElementIndex& ij_el_tl,
              const matrix::SubPanelView& panel_uptonow, MatrixLike& mat_a,
              matrix::Panel<Coord::Col, T, Device::CPU>& W, SenderTau&& tau) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    const SizeType i = ij.row();
    const SizeType j = ij.col();

    const SizeType i_el_tl = ij_el_tl.row();
    const SizeType j_el_tl = ij_el_tl.col();

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
                     ex::when_all_vector(std::move(v_panel_ro)), std::forward<SenderTau>(tau)) |
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
              blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, tile_v.size().rows() - i_first_tl,
                         j_el_tl, T(1), w, tile_w.ld(), v_col, 1, beta, w_up, 1);
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
              blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, tile_v.size().rows() - i_first_tl,
                         j_el_tl, T(1), v, tile_v.ld(), v_col, 1, beta, w_up, 1);
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
            auto&& tile_v = v_tiles[i].get();
            alpha += blas::dot(tile_w.size().rows() - i_first_tl, tile_w.ptr({i_first_tl, j_el_tl}), 1,
                               tile_v.ptr({i_first_tl, j_el_tl}), 1);
          }
          alpha *= T(-0.5) * tau;

          for (std::size_t i = i_first; i < w_tiles.size(); ++i) {
            const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;
            auto&& tile_w = w_tiles[i];
            auto&& tile_v = v_tiles[i].get();
            blas::axpy(tile_w.size().rows() - i_first_tl, alpha, tile_v.ptr({i_first_tl, j_el_tl}), 1,
                       tile_w.ptr({i_first_tl, j_el_tl}), 1);
          }
        }));
  }

  template <class SenderTau, class SenderTri, class SenderA>
  void computeLastTile(SenderTau&& tau, SenderTri&& mat_tri, SenderA&& mat_a) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    ex::start_detached(
        ex::when_all(std::forward<SenderTau>(tau), std::forward<SenderTri>(mat_tri),
                     std::forward<SenderA>(mat_a)) |
        di::transform(di::Policy<B>(), [](auto&& tile_taus, auto&& tile_tri, auto&& tile_a) {
          // TODO create tile wrapper
          lapack::hetrd(lapack::Uplo::Lower, tile_a.size().rows(), tile_a.ptr(), tile_a.ld(),
                        tile_tri.ptr({0, 0}), tile_tri.ptr({0, 1}), tile_taus.ptr());
        }));
  }

  template <class SenderA, class SenderTri>
  void copyLastElement(SenderA&& mat_a, SenderTri&& mat_tri) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    // Note: copy last diagonal element
    ex::start_detached(ex::when_all(std::forward<SenderA>(mat_a), std::forward<SenderTri>(mat_tri)) |
                       ex::then([](auto&& a, auto&& tri) {
                         const SizeType j_tl = tri.size().rows() - 1;
                         if constexpr (isComplex_v<T>)
                           tri({j_tl, 0}) = a.get()({j_tl, j_tl}).real();
                         else
                           tri({j_tl, 0}) = a.get()({j_tl, j_tl});
                       }));
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct Helper<Backend::GPU, Device::GPU, T> : Helper<Backend::MC, Device::CPU, T> {
  static constexpr Backend B = Backend::GPU;
  static constexpr Device D = Device::GPU;

  Helper(const std::size_t n_workspaces, const matrix::Distribution& panel_dist)
      : Helper<Backend::MC, Device::CPU, T>(n_workspaces, panel_dist),
        panels_v(n_workspaces, panel_dist), panels_w(n_workspaces, panel_dist) {}

  void align(const GlobalTileIndex& ij) {
    Helper<Backend::MC, Device::CPU, T>::align(ij);
    auto& Vh = panels_v.nextResource();
    auto& Wh = panels_w.nextResource();

    Vh.setRangeStart(ij);
    Wh.setRangeStart(ij);
  }

  // TODO we can think about having just align there probably
  void reset() {
    Helper<Backend::MC, Device::CPU, T>::reset();
    auto& Vh = panels_v.currentResource();
    auto& Wh = panels_w.currentResource();

    Vh.reset();
    Wh.reset();
  }

  template <class SenderTau, class SenderTrid>
  auto call(const matrix::Distribution& dist_a, const GlobalTileIndex& ij, const SizeType i_el,
            const TileElementIndex ij_el_tl, SenderTau&& taus, SenderTrid&& trid,
            const matrix::SubPanelView& panel_uptonow, matrix::Panel<Coord::Col, const T, D>& W,
            Matrix<T, D>& mat_a) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;
    using pika::execution::thread_stacksize;

    auto& Vh = panels_v.currentResource();
    auto& Wh = panels_w.currentResource();

    // Copy GPU to CPU
    // Note: for both V and W just the last column (i.e. the only one that got updated)
    for (const auto& it : panel_uptonow.iteratorLocal()) {
      const dlaf::matrix::SubTileSpec spec = panel_uptonow(it);
      const dlaf::matrix::SubTileSpec spec_col{{spec.origin.row(),
                                                spec.origin.col() + spec.size.cols() - 1},
                                               {spec.size.rows(), 1}};

      ex::start_detached(
          ex::when_all(splitTile(mat_a.read(it), spec_col), splitTile(Vh.readwrite(it), spec_col)) |
          matrix::copy(di::Policy<CopyBackend_v<Device::GPU, Device::CPU>>(thread_priority::high,
                                                                           thread_stacksize::nostack)));
      ex::start_detached(
          ex::when_all(splitTile(W.read(it), spec_col), splitTile(Wh.readwrite(it), spec_col)) |
          matrix::copy(di::Policy<CopyBackend_v<Device::GPU, Device::CPU>>(thread_priority::high,
                                                                           thread_stacksize::nostack)));
    }

    // Compute on CPU
    auto&& tau =
        Helper<Backend::MC, Device::CPU, T>::call(dist_a, ij, i_el, ij_el_tl,
                                                  std::forward<SenderTau>(taus),
                                                  std::forward<SenderTrid>(trid), panel_uptonow, Wh, Vh);

    // Copy back from CPU to GPU
    // Note: just V which was RW, and just the updated column
    for (const auto& it : panel_uptonow.iteratorLocal()) {
      const dlaf::matrix::SubTileSpec spec = panel_uptonow(it);
      const dlaf::matrix::SubTileSpec spec_col{{spec.origin.row(),
                                                spec.origin.col() + spec.size.cols() - 1},
                                               {spec.size.rows(), 1}};

      ex::start_detached(
          ex::when_all(splitTile(Vh.read(it), spec_col), splitTile(mat_a.readwrite(it), spec_col)) |
          matrix::copy(di::Policy<CopyBackend_v<Device::CPU, Device::GPU>>(thread_priority::high,
                                                                           thread_stacksize::nostack)));
    }

    return std::move(tau);
  }

  template <class SenderTau, class MatrixLike>
  void setupW(const GlobalTileIndex& ij, const TileElementIndex& ij_el_tl,
              const matrix::SubPanelView& panel_uptonow, MatrixLike& mat_a,
              matrix::Panel<Coord::Col, T, D>& W, SenderTau&& tau) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;
    using pika::execution::thread_stacksize;

    auto& Vh = panels_v.currentResource();
    auto& Wh = panels_w.currentResource();

    // Copy GPU to CPU
    // Note: for both V and W just the last column (i.e. the only one that got updated)
    for (const auto& it : panel_uptonow.iteratorLocal()) {
      const dlaf::matrix::SubTileSpec spec = panel_uptonow(it);
      // TODO probably we can
      const dlaf::matrix::SubTileSpec spec_col{{spec.origin.row(),
                                                spec.origin.col() + spec.size.cols() - 1},
                                               {spec.size.rows(), 1}};

      ex::start_detached(
          ex::when_all(splitTile(mat_a.read(it), spec_col), splitTile(Vh.readwrite(it), spec_col)) |
          matrix::copy(di::Policy<CopyBackend_v<Device::GPU, Device::CPU>>(thread_priority::high,
                                                                           thread_stacksize::nostack)));
      ex::start_detached(
          ex::when_all(splitTile(W.read(it), spec_col), splitTile(Wh.readwrite(it), spec_col)) |
          matrix::copy(di::Policy<CopyBackend_v<Device::GPU, Device::CPU>>(thread_priority::high,
                                                                           thread_stacksize::nostack)));
    }

    Helper<Backend::MC, Device::CPU, T>::setupW(ij, ij_el_tl, panel_uptonow, Vh, Wh,
                                                std::forward<SenderTau>(tau));

    // Copy back from CPU to GPU
    // Note: just W which was RW, and just the updated column
    for (const auto& it : panel_uptonow.iteratorLocal()) {
      const dlaf::matrix::SubTileSpec spec = panel_uptonow(it);
      const dlaf::matrix::SubTileSpec spec_col{{spec.origin.row(),
                                                spec.origin.col() + spec.size.cols() - 1},
                                               {spec.size.rows(), 1}};

      ex::start_detached(
          ex::when_all(splitTile(Wh.read(it), spec_col), splitTile(W.readwrite(it), spec_col)) |
          matrix::copy(di::Policy<CopyBackend_v<Device::CPU, Device::GPU>>(thread_priority::high,
                                                                           thread_stacksize::nostack)));
    }
  }

  template <class SenderTau, class SenderTri>
  void computeLastTile(SenderTau&& tau, SenderTri&& mat_tri, Matrix<T, D>& mat_a, SizeType j) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;
    using pika::execution::thread_stacksize;

    auto& Vh = panels_v.currentResource();
    const LocalTileIndex top_lc(*Vh.iteratorLocal().begin());

    ex::start_detached(ex::when_all(mat_a.read(GlobalTileIndex{j, j}), Vh.readwrite(top_lc)) |
                       matrix::copy(di::Policy<CopyBackend_v<Device::GPU, Device::CPU>>(
                           thread_priority::high, thread_stacksize::nostack)));

    Helper<Backend::MC, Device::CPU, T>::computeLastTile(std::forward<SenderTau>(tau),
                                                         std::forward<SenderTri>(mat_tri),
                                                         Vh.readwrite(top_lc));

    ex::start_detached(ex::when_all(Vh.read(top_lc), mat_a.readwrite(GlobalTileIndex{j, j})) |
                       matrix::copy(di::Policy<CopyBackend_v<Device::CPU, Device::GPU>>(
                           thread_priority::high, thread_stacksize::nostack)));
  }

  template <class SenderA, class SenderTri>
  void copyLastElement(SenderA&& mat_a, SenderTri&& mat_tri) {
    dlaf::internal::silenceUnusedWarningFor(mat_tri, mat_a);
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;
    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;
    using pika::execution::thread_stacksize;

    auto& Vh = panels_v.currentResource();

    ex::start_detached(
        ex::when_all(std::forward<SenderA>(mat_a), Vh.readwrite(*Vh.iteratorLocal().begin())) |
        matrix::copy(di::Policy<CopyBackend_v<Device::GPU, Device::CPU>>(thread_priority::high,
                                                                         thread_stacksize::nostack)));

    Helper<Backend::MC, Device::CPU, T>::copyLastElement(Vh.read(*Vh.iteratorLocal().begin()),
                                                         std::forward<SenderTri>(mat_tri));
  }

protected:
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panels_v;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panels_w;
};
#endif

template <Backend B, class T>
struct kernelHEMV;

template <class T>
struct kernelHEMV<Backend::MC, T> {
  template <class SenderAt, class SenderV, class SenderW>
  void operator()(const SizeType chunk_id, const TileElementIndex& ij_el_tl, SenderAt&& at_tiles,
                  SenderV&& v_tiles, SenderW&& tile_w) {
    const SizeType it = chunk_id;
    const SizeType i_el_tl = ij_el_tl.row();
    const SizeType j_el_tl = ij_el_tl.col();

    DLAF_ASSERT_HEAVY(at_tiles.size() = v_tiles.size(), at_tiles.size(), v_tiles.size());
    for (std::size_t index = 0; index < at_tiles.size(); ++index) {
      auto&& tile_a = at_tiles[index].get();
      auto&& tile_v = v_tiles[index].get();

      const SizeType i_first_tl = it == 0 ? i_el_tl : 0;

      if (to_SizeType(index) == it) {
        blas::hemv(blas::Layout::ColMajor, blas::Uplo::Lower, tile_a.size().rows(), T(1), tile_a.ptr(),
                   tile_a.ld(), tile_v.ptr({0, 0}), 1, T(1), tile_w.ptr({i_first_tl, j_el_tl}), 1);
      }
      else {
        const blas::Op op = (to_SizeType(index) < it) ? blas::Op::NoTrans : blas::Op::ConjTrans;
        blas::gemv(blas::Layout::ColMajor, op, tile_a.size().rows(), tile_a.size().cols(), T(1),
                   tile_a.ptr(), tile_a.ld(), tile_v.ptr({0, 0}), 1, T(1),
                   tile_w.ptr({i_first_tl, j_el_tl}), 1);
      }
    }
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct kernelHEMV<Backend::GPU, T> {
  template <class SenderAt, class SenderV, class SenderW>
  void operator()(cublasHandle_t handle, const SizeType chunk_id, const TileElementIndex& ij_el_tl,
                  SenderAt&& at_tiles, SenderV&& v_tiles, SenderW&& tile_w) {
    const SizeType it = chunk_id;
    const SizeType i_el_tl = ij_el_tl.row();
    const SizeType j_el_tl = ij_el_tl.col();

    DLAF_ASSERT_HEAVY(at_tiles.size() = v_tiles.size(), at_tiles.size(), v_tiles.size());
    for (std::size_t index = 0; index < at_tiles.size(); ++index) {
      auto&& tile_a = at_tiles[index].get();
      auto&& tile_v = v_tiles[index].get();

      const SizeType i_first_tl = it == 0 ? i_el_tl : 0;

      using namespace dlaf::util;
      if (to_SizeType(index) == it) {
        const T alpha = 1;
        const T beta = 1;
        gpublas::internal::Hemv<T>::call(handle, blasToCublas(blas::Uplo::Lower),
                                         to_int(tile_a.size().rows()), blasToCublasCast(&alpha),
                                         blasToCublasCast(tile_a.ptr()), to_int(tile_a.ld()),
                                         blasToCublasCast(tile_v.ptr()), 1, blasToCublasCast(&beta),
                                         blasToCublasCast(tile_w.ptr({i_first_tl, j_el_tl})), to_int(1));
      }
      else {
        const blas::Op op = (to_SizeType(index) < it) ? blas::Op::NoTrans : blas::Op::ConjTrans;
        const T alpha = 1;
        const T beta = 1;
        gpublas::internal::Gemv<T>::call(handle, blasToCublas(op), to_int(tile_a.size().rows()),
                                         to_int(tile_a.size().cols()), blasToCublasCast(&alpha),
                                         blasToCublasCast(tile_a.ptr()), to_int(tile_a.ld()),
                                         blasToCublasCast(tile_v.ptr()), 1, blasToCublasCast(&beta),
                                         blasToCublasCast(tile_w.ptr({i_first_tl, j_el_tl})), to_int(1));
      }
    }
  }
};
#endif

template <Backend B, class T, Device D>
void hemvPanelColumn(const GlobalTileIndex& ij, const TileElementIndex& ij_el_tl,
                     dlaf::matrix::internal::MatrixRef<const T, D>& At,
                     dlaf::matrix::internal::MatrixRef<const T, D>& V,
                     dlaf::matrix::Panel<Coord::Col, T, D>& W) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType i = ij.row();
  const SizeType i_el_tl = ij_el_tl.row();
  const SizeType j_el_tl = ij_el_tl.col();

  for (SizeType it = 0; it < At.nr_tiles().rows(); ++it) {
    const SizeType i_lc = At.distribution().template local_tile_from_global_tile<Coord::Row>(it);

    // Note: this is for W which starts from the beginning
    // TODO fix for last reflector of the panel
    auto&& tile_w = W.readwrite({to_SizeType(i) + i_lc, 0});  // TODO workaround to align At and W

    auto row_ij =
        common::iterate_range2d(GlobalTileIndex{it, 0}, GlobalTileSize{1, At.nr_tiles().cols()});

    // Note: V is aligned with At, but it must be accessed with the col due to mul from right
    std::vector<matrix::ReadOnlyTileSender<T, D>> v_chunk;
    v_chunk.reserve(to_sizet(At.nr_tiles().cols()));
    for (const auto& ij : row_ij)
      v_chunk.emplace_back(V.read(GlobalTileIndex{ij.col(), 0}));

    std::vector<GlobalTileIndex> row_ij_lower;
    std::transform(row_ij.begin(), row_ij.end(), std::back_inserter(row_ij_lower), [](auto&& ij) {
      const bool is_lower = ij.row() > ij.col();
      return is_lower ? ij : transposed(ij);
    });

    // Note: At has to be accessed just in the lower part
    std::vector<matrix::ReadOnlyTileSender<T, D>> at_chunk;
    at_chunk.reserve(to_sizet(At.nr_tiles().cols()));
    for (const auto& ij_lower : row_ij_lower)
      at_chunk.emplace_back(At.read(ij_lower));

    ex::start_detached(ex::when_all(ex::just(it), ex::just(TileElementIndex{i_el_tl, j_el_tl}),
                                    ex::when_all_vector(std::move(at_chunk)),
                                    ex::when_all_vector(std::move(v_chunk)), std::move(tile_w)) |
                       di::transform<di::TransformDispatchType::Blas>(di::Policy<B>(),
                                                                      kernelHEMV<B, T>{}));
  }
}

template <Backend B, Device D, class T>
TridiagResult1Stage<T> ReductionToTrid<B, D, T>::call(Matrix<T, D>& mat_a) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  using matrix::internal::MatrixRef;

  const auto dist_a = mat_a.distribution();

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

  Helper<B, D, T> helper(n_workspaces, dist);

  for (SizeType j = 0; j < dist_a.nr_tiles().cols(); ++j) {
    auto& W = panels_w.nextResource();

    W.setRangeStart(GlobalTileIndex{j, 0});
    helper.align(GlobalTileIndex{j, 0});

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

      auto tau = helper.call(dist_a, GlobalTileIndex{i, j}, i_el, TileElementIndex{i_el_tl, j_el_tl},
                             mat_taus.readwrite(GlobalTileIndex{j, 0}),
                             mat_trid.readwrite(GlobalTileIndex{j, 0}), panel_uptonow, W, mat_a);

      // W = At @ v
      const SizeType mt = dist_a.size().rows() - i_el;
      MatrixRef<const T, D> V(mat_a, {{i_el, j_el}, {mt, 1}});
      MatrixRef<const T, D> At(mat_a, {{i_el, j_el + 1}, {mt, mt}});

      hemvPanelColumn<B>(GlobalTileIndex{i, j}, TileElementIndex{i_el_tl, j_el_tl}, At, V, W);

      // compute W
      helper.setupW(GlobalTileIndex{i, j}, TileElementIndex{i_el_tl, j_el_tl}, panel_uptonow, mat_a, W,
                    std::move(tau));
    }
    if (is_last_tile) {
      if (dist_a.template tile_size_of<Coord::Col>(j) > 1) {
        if constexpr (B == Backend::MC)
          helper.computeLastTile(mat_taus.readwrite(GlobalTileIndex{j, 0}),
                                 mat_trid.readwrite(GlobalTileIndex{j, 0}),
                                 mat_a.readwrite(GlobalTileIndex{j, j}));
        if constexpr (B == Backend::GPU)
          helper.computeLastTile(mat_taus.readwrite(GlobalTileIndex{j, 0}),
                                 mat_trid.readwrite(GlobalTileIndex{j, 0}), mat_a, j);
      }
      else
        helper.copyLastElement(mat_a.read(GlobalTileIndex{j, j}),
                               mat_trid.readwrite(GlobalTileIndex{j, 0}));
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

      const auto priority = (ij.col() == j + 1) ? pika::execution::thread_priority::high
                                                : pika::execution::thread_priority::normal;

      if (ij.row() == ij.col())
        her2kDiag<B>(priority, mat_a.read(i_v(ij.row())), W.read(i_w(ij.row())), mat_a.readwrite(ij));
      else {
        her2kOffDiag<B>(priority, mat_a.read(i_v(ij.row())), W.read(i_w(ij.col())), mat_a.readwrite(ij));
        her2kOffDiag<B>(priority, W.read(i_w(ij.row())), mat_a.read(i_v(ij.col())), mat_a.readwrite(ij));
      }
    }

    W.reset();
    helper.reset();
  }

  return {std::move(mat_taus), std::move(mat_trid)};
}
}
