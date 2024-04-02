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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <pika/execution.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/eigensolver/reduction_to_trid/api.h>
#include <dlaf/eigensolver/reduction_to_trid/gpu/kernels.h>
#include <dlaf/eigensolver/reduction_to_trid/impl.h>
#include <dlaf/eigensolver/reduction_utils/misc.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/matrix/panel.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/matrix/views.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/transform.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/types.h>
#include <dlaf/util_cublas.h>
#include <dlaf/util_matrix.h>

#ifdef DLAF_WITH_GPU
namespace dlaf::eigensolver::internal {

template <class T>
struct kernelSetupW<Backend::GPU, T> {
  template <class SenderW, class SenderV, class SenderTau>
  void operator()(cublasHandle_t handle, const std::size_t i_first, const TileElementIndex ij_el_tl,
                  const SizeType panel_height, SenderW&& w_tiles, SenderV&& v_tiles, SenderTau&& tau) {
    const SizeType i_el_tl = ij_el_tl.row();
    const SizeType j_el_tl = ij_el_tl.col();

    // computeW mods
    if (j_el_tl > 0) {
      auto&& w_up = w_tiles[0].ptr({0, j_el_tl});

      // w_up = W* . v
      {
        auto&& tile_v = v_tiles[i_first].get();
        auto&& tile_w = w_tiles[i_first];

        auto&& w = tile_w.ptr({i_el_tl, 0});
        auto&& v_col = tile_v.ptr({i_el_tl, j_el_tl});

        // TODO this shouldn't be needed: probably w is computed outside of needed bounds
        const T alpha = 1;
        const T beta = 0;

        gpublas::internal::Gemv<T>::call(handle, util::blasToCublas(blas::Op::ConjTrans),
                                         to_int(panel_height), to_int(j_el_tl),
                                         util::blasToCublasCast(&alpha), util::blasToCublasCast(w),
                                         to_int(tile_w.ld()), util::blasToCublasCast(v_col), 1,
                                         util::blasToCublasCast(&beta), util::blasToCublasCast(w_up), 1);
      }

      // w = w - V . w_up
      {
        const T alpha = -1;
        const T beta = 1;

        auto&& tile_v = v_tiles[i_first].get();
        auto&& tile_w = w_tiles[i_first];

        auto&& v = tile_v.ptr({i_el_tl, 0});
        auto&& w_col = tile_w.ptr({i_el_tl, j_el_tl});

        gpublas::internal::Gemv<T>::call(handle, util::blasToCublas(blas::Op::NoTrans),
                                         to_int(panel_height), to_int(j_el_tl),
                                         util::blasToCublasCast(&alpha), util::blasToCublasCast(v),
                                         to_int(tile_v.ld()), util::blasToCublasCast(w_up), 1,
                                         util::blasToCublasCast(&beta), util::blasToCublasCast(w_col),
                                         1);
      }

      // w_up = V* . v
      {
        auto&& tile_v = v_tiles[i_first].get();

        auto&& v = tile_v.ptr({i_el_tl, 0});
        auto&& v_col = tile_v.ptr({i_el_tl, j_el_tl});

        const T alpha = 1;
        const T beta = 0;
        gpublas::internal::Gemv<T>::call(handle, util::blasToCublas(blas::Op::ConjTrans),
                                         to_int(panel_height), to_int(j_el_tl),
                                         util::blasToCublasCast(&alpha), util::blasToCublasCast(v),
                                         to_int(tile_v.ld()), util::blasToCublasCast(v_col), 1,
                                         util::blasToCublasCast(&beta), util::blasToCublasCast(w_up), 1);
      }

      // w = w - W . w_up
      {
        const T alpha = -1;
        const T beta = 1;

        auto&& tile_w = w_tiles[i_first];

        auto&& w = tile_w.ptr({i_el_tl, 0});
        auto&& w_col = tile_w.ptr({i_el_tl, j_el_tl});

        gpublas::internal::Gemv<T>::call(handle, util::blasToCublas(blas::Op::NoTrans),
                                         to_int(panel_height), to_int(j_el_tl),
                                         util::blasToCublasCast(&alpha), util::blasToCublasCast(w),
                                         to_int(tile_w.ld()), util::blasToCublasCast(w_up), 1,
                                         util::blasToCublasCast(&beta), util::blasToCublasCast(w_col),
                                         1);
      }
    }

    for (std::size_t i = i_first; i < w_tiles.size(); ++i) {
      const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;
      auto&& tile_w = w_tiles[i];
      gpublas::internal::Scal<T>::call(handle, to_int(tile_w.size().rows() - i_first_tl),
                                       util::blasToCublasCast(&tau),
                                       util::blasToCublasCast(tile_w.ptr({i_first_tl, j_el_tl})), 1);
    }

    T alpha = 0;
    for (std::size_t i = i_first; i < w_tiles.size(); ++i) {
      const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;
      auto&& tile_w = w_tiles[i];
      auto&& tile_v = v_tiles[i].get();
      T partial_result;
      gpublas::internal::Dot<T>::call(handle, to_int(tile_w.size().rows() - i_first_tl),
                                      util::blasToCublasCast(tile_w.ptr({i_first_tl, j_el_tl})), 1,
                                      util::blasToCublasCast(tile_v.ptr({i_first_tl, j_el_tl})), 1,
                                      util::blasToCublasCast(&partial_result));
      alpha += partial_result;
    }
    alpha *= T(-0.5) * tau;

    for (std::size_t i = i_first; i < w_tiles.size(); ++i) {
      const SizeType i_first_tl = (i == i_first) ? i_el_tl : 0;
      auto&& tile_w = w_tiles[i];
      auto&& tile_v = v_tiles[i].get();
      gpublas::internal::Axpy<T>::call(handle, to_int(tile_w.size().rows() - i_first_tl),
                                       util::blasToCublasCast(&alpha),
                                       util::blasToCublasCast(tile_v.ptr({i_first_tl, j_el_tl})), 1,
                                       util::blasToCublasCast(tile_w.ptr({i_first_tl, j_el_tl})), 1);
    }
  }
};

template <class T>
struct batch_t {
  batch_t(const SizeType size) : pointers_({size, 1}, {size, 1}) {}

  matrix::ReadWriteTileSender<T*, Device::GPU> readwrite() {
    return pointers_.readwrite(LocalTileIndex{0, 0});
  }

  matrix::ReadOnlyTileSender<T*, Device::GPU> read() {
    return pointers_.read(LocalTileIndex{0, 0});
  }

private:
  Matrix<T*, Device::GPU> pointers_;
};

template <class T>
struct Helper<Backend::GPU, Device::GPU, T> : Helper<Backend::MC, Device::CPU, T> {
  static constexpr Backend B = Backend::GPU;
  static constexpr Device D = Device::GPU;

  Helper(const std::size_t n_workspaces, const matrix::Distribution& panel_dist)
      : Helper<Backend::MC, Device::CPU, T>(n_workspaces, panel_dist),
        panels_v(n_workspaces, panel_dist), batch_v_panel(panel_dist.nrTiles().rows()),
        batch_w_panel(panel_dist.nrTiles().rows()), batch_w_col(panel_dist.nrTiles().rows()),
        batch_w_up(panel_dist.nrTiles().rows()) {}

  void align(const GlobalTileIndex& ij) {
    Helper<Backend::MC, Device::CPU, T>::align(ij);
    auto& Vh = panels_v.nextResource();

    Vh.setRangeStart(ij);
  }

  void init(const SizeType, Matrix<T, D>&, matrix::Panel<Coord::Col, T, D>&) {}

  void step() {}

  // TODO we can think about having just align there probably
  void reset() {
    Helper<Backend::MC, Device::CPU, T>::reset();
    auto& Vh = panels_v.currentResource();
    Vh.reset();
  }

  template <class MatrixLike>
  void updateV(const matrix::Distribution& dist_a, const GlobalTileIndex& ij, const SizeType i_el,
               const TileElementIndex ij_el_tl, const matrix::SubPanelView& panel_uptonow,
               matrix::Panel<Coord::Col, const T, D>& W, MatrixLike& mat_a) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    const SizeType j = ij.col();

    const bool has_left_panel = ij_el_tl.col() > 0;

    if (!has_left_panel)
      return;

    const std::size_t ntiles = to_sizet(std::distance(panel_uptonow.iteratorLocal().begin(),
                                                      panel_uptonow.iteratorLocal().end()));

    std::vector<matrix::ReadWriteTileSender<T, D>> v_panel_rw;
    v_panel_rw.reserve(ntiles);
    for (const auto& it : panel_uptonow.iteratorLocal())
      v_panel_rw.emplace_back(splitTile(mat_a.readwrite(it), panel_uptonow(it)));

    std::vector<matrix::ReadOnlyTileSender<T, D>> w_panel_ro;
    w_panel_ro.reserve(ntiles);
    for (const auto& it : panel_uptonow.iteratorLocal())
      w_panel_ro.emplace_back(splitTile(W.read(it), panel_uptonow(it)));

    const std::size_t i_first =
        to_sizet(dist_a.template global_tile_from_global_element<Coord::Row>(i_el - 1) - j);

    auto kernelUpdateV = [dist_a, i_first, i_el, ij_el_tl](cublasHandle_t handle, auto&& w_tiles,
                                                           auto&& v_tiles) {
      const SizeType i_el_tl = dist_a.template tile_element_from_global_element<Coord::Row>(i_el - 1);
      const SizeType j_el_tl = ij_el_tl.col();

      SizeType m = 0;
      for (std::size_t i = i_first; i < v_tiles.size(); ++i)
        m += v_tiles[i].size().rows();

      const T alpha = -1;
      const T beta = 1;

      {
        auto&& tile_v = v_tiles[i_first];
        auto&& tile_wt = w_tiles[0].get();
        auto&& col_out = v_tiles[i_first];

        gpublas::internal::Gemm<T>::call(
            handle, util::blasToCublas(blas::Op::NoTrans), util::blasToCublas(blas::Op::ConjTrans),
            to_int(m), 1, to_int(j_el_tl), util::blasToCublasCast(&alpha),
            util::blasToCublasCast(tile_v.ptr({i_el_tl, 0})), to_int(tile_v.ld()),
            util::blasToCublasCast(tile_wt.ptr({j_el_tl, 0})), to_int(tile_wt.ld()),
            util::blasToCublasCast(&beta), util::blasToCublasCast(col_out.ptr({i_el_tl, j_el_tl})),
            to_int(col_out.ld()));
      }
      {
        auto&& tile_w = w_tiles[i_first].get();
        auto&& tile_vt = v_tiles[0];
        auto&& col_out = v_tiles[i_first];

        gpublas::internal::Gemm<T>::call(
            handle, util::blasToCublas(blas::Op::NoTrans), util::blasToCublas(blas::Op::ConjTrans),
            to_int(m), 1, to_int(j_el_tl), util::blasToCublasCast(&alpha),
            util::blasToCublasCast(tile_w.ptr({i_el_tl, 0})), to_int(tile_w.ld()),
            util::blasToCublasCast(tile_vt.ptr({j_el_tl, 0})), to_int(tile_vt.ld()),
            util::blasToCublasCast(&beta), util::blasToCublasCast(col_out.ptr({i_el_tl, j_el_tl})),
            to_int(col_out.ld()));
      }
    };

    ex::start_detached(di::whenAllLift(ex::when_all_vector(std::move(w_panel_ro)),
                                       ex::when_all_vector(std::move(v_panel_rw))) |
                       di::transform<di::TransformDispatchType::Blas>(di::Policy<B>(), kernelUpdateV));
  }

  template <class SenderTau, class SenderTrid>
  auto computeReflector(const GlobalTileIndex& ij, const TileElementIndex ij_el_tl, SenderTau&& taus,
                        SenderTrid&& trid, const matrix::SubPanelView& panel_uptonow,
                        Matrix<T, D>& mat_a) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    using dlaf::matrix::internal::CopyBackend_v;
    using pika::execution::thread_priority;
    using pika::execution::thread_stacksize;

    auto& Vh = panels_v.currentResource();

    std::vector<matrix::ReadOnlyTileSender<T, Device::GPU>> all_tiles_in;
    std::vector<matrix::ReadWriteTileSender<T, Device::CPU>> all_tiles_out;

    for (const auto& it : panel_uptonow.iteratorLocal()) {
      const dlaf::matrix::SubTileSpec spec = panel_uptonow(it);
      const dlaf::matrix::SubTileSpec spec_col{{spec.origin.row(),
                                                spec.origin.col() + spec.size.cols() - 1},
                                               {spec.size.rows(), 1}};

      all_tiles_in.emplace_back(splitTile(mat_a.read(it), spec_col));
      all_tiles_out.emplace_back(splitTile(Vh.readwrite(it), spec_col));
    }

    // Copy GPU to CPU
    ex::start_detached(
        di::whenAllLift(ex::when_all_vector(std::move(all_tiles_in)),
                        ex::when_all_vector(std::move(all_tiles_out)), panel_uptonow.rows()) |
        di::transform(di::Policy<Backend::GPU>(),
                      [](auto&& tile_ins, auto&& tile_out, SizeType rows, whip::stream_t stream) {
                        cudaMemcpyAsync(tile_out[0].ptr({0, 0}), tile_ins[0].get().ptr({0, 0}),
                                        sizeof(T) * to_sizet(rows), cudaMemcpyDeviceToHost, stream);
                      }));

    // Compute on CPU
    auto&& tau = Helper<Backend::MC, Device::CPU, T>::computeReflector(
        ij, ij_el_tl, std::forward<SenderTau>(taus), std::forward<SenderTrid>(trid), panel_uptonow, Vh);

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

    const SizeType i = ij.row();
    const SizeType j = ij.col();

    const std::size_t ntiles = to_sizet(std::distance(panel_uptonow.iteratorLocal().begin(),
                                                      panel_uptonow.iteratorLocal().end()));

    std::vector<matrix::ReadOnlyTileSender<T, Device::GPU>> v_panel_ro;
    v_panel_ro.reserve(ntiles);

    std::vector<matrix::ReadWriteTileSender<T, Device::GPU>> w_panel_rw;
    w_panel_rw.reserve(ntiles);

    for (const auto& it : panel_uptonow.iteratorLocal()) {
      const auto spec = panel_uptonow(it);
      v_panel_ro.emplace_back(splitTile(mat_a.read(it), spec));
      w_panel_rw.emplace_back(splitTile(W.readwrite(it), spec));
    }

    const std::size_t offset = to_sizet(i - j);
    const SizeType m =
        panel_uptonow.rows() - ij_el_tl.row() - to_SizeType(offset) * mat_a.tile_size().rows();

    ex::start_detached(
        di::whenAllLift(offset, ij_el_tl, m, ex::when_all_vector(std::move(w_panel_rw)),
                        ex::when_all_vector(std::move(v_panel_ro)), std::forward<SenderTau>(tau)) |
        di::transform<di::TransformDispatchType::Blas>(di::Policy<B>(), kernelSetupW<B, T>{}));
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
  std::size_t ntiles = 0;
  pika::execution::experimental::any_sender<std::tuple<SizeType, SizeType>> lds_snd;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> panels_v;

  // no update
  batch_t<T> batch_v_panel;
  batch_t<T> batch_w_panel;

  // to be updated
  batch_t<T> batch_w_col;
  batch_t<T> batch_w_up;
};

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
}
#endif
