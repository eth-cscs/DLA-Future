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

#include <type_traits>

#include <hpx/include/util.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/eigensolver/backtransformation/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/traits.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
static constexpr bool is_complex_v = std::is_same_v<T, ComplexType<T>>;

template <class T>
static SizeType nrSweeps(const SizeType m) {
  return std::max<SizeType>(0, is_complex_v<T> ? m - 1 : m - 2);
}

static SizeType nrStepsPerSweep(SizeType sweep, SizeType m, SizeType mb) {
  return std::max<SizeType>(0, sweep == m - 2 ? 1 : dlaf::util::ceilDiv(m - sweep - 2, mb));
}

template <class T>
struct BackTransformationT2B<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_e, Matrix<const T, Device::CPU>& mat_i);
};

template <class T>
auto setupVWellFormed(SizeType k, hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v_compact,
                      hpx::future<matrix::Tile<T, Device::CPU>> tile_v) -> decltype(tile_v_compact) {
  auto unzipV_func = [k](const auto& tile_v_compact, auto tile_v) {
    using lapack::MatrixType;

    for (SizeType j = 0; j < k; ++j) {
      const auto size =
          std::min<SizeType>(tile_v.size().rows() - (1 + j), tile_v_compact.size().rows() - 1);
      // TODO this is needed because of complex last reflector (i.e. just 1 element long)
      if (size == 0)
        continue;

      lacpy(MatrixType::General, size, 1, tile_v_compact.ptr({1, j}), tile_v_compact.ld(),
            tile_v.ptr({1 + j, j}), tile_v.ld());
    }

    // TODO is it needed because W = V . T? or is it enough just setting ones?
    laset(MatrixType::Upper, tile_v.size().rows(), k, T(0), T(1), tile_v.ptr({0, 0}), tile_v.ld());

    const SizeType mb = tile_v_compact.size().cols();
    if (tile_v.size().rows() > mb)
      laset(MatrixType::Lower, tile_v.size().rows() - mb, k - 1, T(0), T(0), tile_v.ptr({mb, 0}),
            tile_v.ld());

    return matrix::Tile<const T, Device::CPU>(std::move(tile_v));
  };
  return hpx::dataflow(hpx::unwrapping(unzipV_func), std::move(tile_v_compact), std::move(tile_v));
}

template <class T>
auto computeTFactor(hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_taus,
                    hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
                    hpx::future<matrix::Tile<T, Device::CPU>> mat_t) -> decltype(tile_v) {
  auto tfactor_task = [](const auto& tile_taus, const auto& tile_v, auto tile_t) {
    using namespace lapack;

    const auto k = tile_v.size().cols();
    const auto n = tile_v.size().rows();
    // DLAF_ASSERT_HEAVY((tile_t.size() == TileElementSize(k, k)), tile_t.size());

    std::vector<T> taus;
    taus.resize(to_sizet(tile_v.size().cols()));
    for (SizeType i = 0; i < to_SizeType(taus.size()); ++i)
      taus[to_sizet(i)] = *tile_taus.ptr({0, i});

    larft(Direction::Forward, StoreV::Columnwise, n, k, tile_v.ptr(), tile_v.ld(), taus.data(),
          tile_t.ptr(), tile_t.ld());

    return matrix::Tile<const T, Device::CPU>(std::move(tile_t));
  };
  return hpx::dataflow(hpx::unwrapping(tfactor_task), std::move(tile_taus), std::move(tile_v),
                       std::move(mat_t));
}

template <Backend backend, class VSender, class TSender>
auto computeW(hpx::threads::thread_priority priority, VSender&& tile_v, TSender&& tile_t) {
  using namespace blas;

  using T = dlaf::internal::SenderElementType<VSender>;
  dlaf::internal::whenAllLift(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1),
                              std::forward<TSender>(tile_t), std::forward<VSender>(tile_v)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class VSender, class ESender, class T, class W2Sender>
auto computeW2(hpx::threads::thread_priority priority, VSender&& tile_v, ESender&& tile_e, T beta,
               W2Sender&& tile_w2) {
  using blas::Op;
  dlaf::internal::whenAllLift(Op::ConjTrans, Op::NoTrans, T(1), std::forward<VSender>(tile_v),
                              std::forward<ESender>(tile_e), beta, std::forward<W2Sender>(tile_w2)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <Backend backend, class WSender, class W2Sender, class ESender>
auto updateE(hpx::threads::thread_priority priority, WSender&& tile_w, W2Sender&& tile_w2,
             ESender&& tile_e) {
  using blas::Op;
  using T = dlaf::internal::SenderElementType<ESender>;
  dlaf::internal::whenAllLift(Op::NoTrans, Op::NoTrans, T(-1), std::forward<WSender>(tile_w),
                              std::forward<W2Sender>(tile_w2), T(1), std::forward<ESender>(tile_e)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) | hpx::execution::experimental::detach();
}

template <class T>
void BackTransformationT2B<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_e,
                                                              Matrix<const T, Device::CPU>& mat_i) {
  static constexpr Backend backend = Backend::MC;
  using hpx::threads::thread_priority;
  using hpx::execution::experimental::keep_future;

  using matrix::Panel;
  using common::RoundRobin;
  using common::iterate_range2d;

  if (mat_i.size().isEmpty() || mat_e.size().isEmpty())
    return;

  const SizeType m = mat_e.nrTiles().rows();
  const SizeType n = mat_e.nrTiles().cols();

  const SizeType mb = mat_e.blockSize().rows();
  const SizeType b = mb;

  const auto& dist_i = mat_i.distribution();
  const TileElementSize w_blocksize(2 * b - 1, b);

  const SizeType nsweeps = nrSweeps<T>(mat_i.size().cols());

  // TODO w last tile is complete anyway, because of setup V well formed
  const matrix::Distribution dist_w({m * w_blocksize.rows(), w_blocksize.cols()}, w_blocksize,
                                    dist_i.commGridSize(), dist_i.rankIndex(), dist_i.sourceRankIndex());

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> t_panels(n_workspaces, mat_i.distribution());
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> w_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Row, T, Device::CPU>> w2_panels(n_workspaces, mat_e.distribution());

  const SizeType last_sweep_tile = (nsweeps - 1) / mb;
  for (SizeType sweep_tile = last_sweep_tile; sweep_tile >= 0; --sweep_tile) {
    const SizeType steps = nrStepsPerSweep(sweep_tile * mb, mat_i.size().cols(), mb);

    auto& mat_w = w_panels.nextResource();
    auto& mat_t = t_panels.nextResource();
    auto& mat_w2 = w2_panels.nextResource();

    for (SizeType step = 0; step < steps; ++step) {
      const SizeType i_v = sweep_tile + step;

      const LocalTileIndex ij_refls(i_v, sweep_tile);

      const bool affectsTwoRows = i_v < (m - 1);

      std::array<SizeType, 2> sizes{
          mat_i.tileSize({i_v, sweep_tile}).rows() - 1,
          affectsTwoRows ? mat_i.tileSize({i_v + 1, sweep_tile}).rows() : 0,
      };
      const SizeType w_rows = sizes[0] + sizes[1];

      // TODO fix this
      const SizeType nrefls = std::min(mb, nsweeps - sweep_tile * mb);
      const SizeType k = sweep_tile == last_sweep_tile ? nrefls : std::min(w_rows - 1, mb);

      mat_w.setWidth(k);
      mat_t.setWidth(k);
      mat_w2.setHeight(k);

      // TODO setRange?

      auto tile_w_full = mat_w(ij_refls);
      auto tile_w_rw = splitTile(tile_w_full, {{0, 0}, {w_rows, k}});

      const auto& tile_i = mat_i.read(ij_refls);
      auto tile_v = setupVWellFormed(k, tile_i, std::move(tile_w_rw));

      auto tile_t_full = mat_t(LocalTileIndex(i_v, 0));
      auto tile_t = computeTFactor(tile_i, tile_v, splitTile(tile_t_full, {{0, 0}, {k, k}}));

      // Note:
      // W2 is computed before W, because W is used as temporary storage for V.
      for (SizeType j_e = 0; j_e < n; ++j_e) {
        const auto sz_e = mat_e.tileSize({i_v, j_e});

        auto tile_v_up = keep_future(splitTile(tile_v, {{0, 0}, {sizes[0], k}}));
        auto tile_e = mat_e.read(LocalTileIndex(i_v, j_e));
        auto tile_e_up = keep_future(splitTile(tile_e, {{1, 0}, {sizes[0], sz_e.cols()}}));

        computeW2<backend>(thread_priority::normal, std::move(tile_v_up), std::move(tile_e_up), T(0),
                           mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));

        if (affectsTwoRows) {
          auto tile_v_down = keep_future(splitTile(tile_v, {{sizes[0], 0}, {sizes[1], k}}));
          auto tile_e_down = mat_e.read_sender(LocalTileIndex(i_v + 1, j_e));

          computeW2<backend>(thread_priority::normal, std::move(tile_v_down), std::move(tile_e_down),
                             T(1), mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));
        }
      }

      tile_w_rw = splitTile(tile_w_full, {{0, 0}, {w_rows, k}});
      computeW<backend>(thread_priority::normal, std::move(tile_w_rw), keep_future(tile_t));
      auto tile_w = mat_w.read(ij_refls);

      for (SizeType j_e = 0; j_e < n; ++j_e) {
        const LocalTileIndex idx_e(i_v, j_e);
        const auto sz_e = mat_e.tileSize({i_v, j_e});

        auto tile_e = mat_e(idx_e);
        auto tile_e_up = splitTile(tile_e, {{1, 0}, {sizes[0], sz_e.cols()}});
        auto tile_w_up = keep_future(splitTile(tile_w, {{0, 0}, {sizes[0], k}}));
        const auto& tile_w2 = mat_w2.read_sender(idx_e);

        updateE<backend>(hpx::threads::thread_priority::normal, tile_w_up, tile_w2, std::move(tile_e_up));

        if (affectsTwoRows) {
          auto tile_e_dn = mat_e.readwrite_sender(LocalTileIndex{i_v + 1, j_e});
          auto tile_w_dn = keep_future(splitTile(tile_w, {{sizes[0], 0}, {sizes[1], k}}));

          updateE<backend>(hpx::threads::thread_priority::normal, tile_w_dn, tile_w2, std::move(tile_e_dn));
        }
      }

      mat_t.reset();
      mat_w.reset();
      mat_w2.reset();
    }
  }
}

/// ---- ETI
#define DLAF_EIGENSOLVER_BACKTRANSFORMATION_T2B_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct BackTransformationT2B<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_BACKTRANSFORMATION_T2B_MC_ETI(extern, float)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_T2B_MC_ETI(extern, double)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_T2B_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_T2B_MC_ETI(extern, std::complex<double>)

}
}
}
