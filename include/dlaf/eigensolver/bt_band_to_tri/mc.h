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
#include "dlaf/eigensolver/bt_band_to_tri/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/traits.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver::internal {

template <class T>
inline constexpr bool is_complex_v = std::is_same_v<T, ComplexType<T>>;

template <class T>
SizeType nrSweeps(const SizeType m) {
  return std::max<SizeType>(0, is_complex_v<T> ? m - 1 : m - 2);
}

inline SizeType nrStepsPerSweep(SizeType sweep, SizeType m, SizeType mb) {
  return std::max<SizeType>(0, sweep == m - 2 ? 1 : dlaf::util::ceilDiv(m - sweep - 2, mb));
}

template <class T>
struct BackTransformationT2B<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_e, Matrix<const T, Device::CPU>& mat_i);
};

template <class T>
hpx::shared_future<matrix::Tile<const T, Device::CPU>> setupVWellFormed(
    SizeType k, hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v_compact,
    hpx::future<matrix::Tile<T, Device::CPU>> tile_v) {
  auto unzipV_func = [k](const auto& tile_v_compact, auto tile_v) {
    using lapack::MatrixType;

    // copy from compact representation reflector values (the first component set to 1 is not there)
    for (SizeType j = 0; j < k; ++j) {
      const auto compact_refl_size =
          std::min<SizeType>(tile_v.size().rows() - (1 + j), tile_v_compact.size().rows() - 1);
      // TODO this is needed because of complex last reflector (i.e. just 1 element long)
      if (compact_refl_size == 0)
        continue;

      lacpy(MatrixType::General, compact_refl_size, 1, tile_v_compact.ptr({1, j}), tile_v_compact.ld(),
            tile_v.ptr({1 + j, j}), tile_v.ld());
    }

    // add the ones as first component for each reflector
    // TODO is it needed because W = V . T? or is it enough just setting ones?
    laset(MatrixType::Upper, tile_v.size().rows(), k, T(0), T(1), tile_v.ptr({0, 0}), tile_v.ld());

    // due to the skewed shape, reflectors do not occupy the tile till the last row. this step
    // zeros out the lower triangular "padding" part under reflectors
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
      taus[to_sizet(i)] = tile_taus({0, i});

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

  const SizeType mb = mat_e.blockSize().rows();
  const SizeType b = mb;
  const SizeType nsweeps = nrSweeps<T>(mat_i.size().cols());

  const auto& dist_i = mat_i.distribution();

  // Note:
  // w_tile_sz can store reflectors are they are actually applied, opposed to how they are stored
  // in compact form.
  //
  // e.g. Given b = 4
  //
  // compact       w_tile_sz
  // 1 1 1 1       1 0 0 0
  // a b c d       a 1 0 0
  // a b c d       a b 1 0
  // a b c d       a b c 1
  //               0 b c d
  //               0 0 c d
  //               0 0 0 d

  const TileElementSize w_tile_sz(2 * b - 1, b);
  // TODO w last tile is complete anyway, because of setup V well formed
  const matrix::Distribution dist_w({mat_e.nrTiles().rows() * w_tile_sz.rows(), w_tile_sz.cols()},
                                    w_tile_sz, dist_i.commGridSize(), dist_i.rankIndex(),
                                    dist_i.sourceRankIndex());

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> t_panels(n_workspaces, mat_i.distribution());
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> w_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Row, T, Device::CPU>> w2_panels(n_workspaces, mat_e.distribution());

  // Note: sweep are on diagonals, steps are on verticals
  const SizeType j_last_sweep = (nsweeps - 1) / mb;
  for (SizeType j = j_last_sweep; j >= 0; --j) {
    auto& mat_w = w_panels.nextResource();
    auto& mat_t = t_panels.nextResource();
    auto& mat_w2 = w2_panels.nextResource();

    // Note: apply the entire column (steps)
    const SizeType steps = nrStepsPerSweep(j * mb, mat_i.size().cols(), mb);
    for (SizeType step = 0; step < steps; ++step) {
      const SizeType i = j + step;
      const LocalTileIndex ij_refls(i, j);

      const bool affectsTwoRows = i < (mat_i.nrTiles().rows() - 1);

      // Note: Since the band is of size (mb + 1), in order to tridiagonalize the matrix,
      // reflectors are of length mb and start with an offset of 1. This means that the application
      // of the reflector involves multiple tiles.
      //
      // e.g. mb = 4
      // reflectors   matrix
      //              X X X X
      // 1 0 0 0      X X X X
      // a 1 0 0      X X X X
      // a b 1 0      X X X X
      //              -------
      // a b c 1      Y Y Y Y
      // 0 b c d      Y Y Y Y
      // 0 0 c d      Y Y Y Y
      // 0 0 0 d      Y Y Y Y
      //
      // From the drawing above, it is possible to see the dashed tile separation between X and Y,
      // and how the reflectors on the left are going to be applied. In particular, the first row of
      // the upper tile is not afftected.
      //
      // For this reason, we pre-compute the size of the upper and lower (in case there is one) tiles,
      // together with the size of the tile of the refelctors (v_rows), that as depicted above it
      // has a different blocksize (no dashed line, it's a single tile)
      const std::array<SizeType, 2> sizes{
          mat_i.tileSize({i, j}).rows() - 1,
          affectsTwoRows ? mat_i.tileSize({i + 1, j}).rows() : 0,
      };
      const SizeType v_rows = sizes[0] + sizes[1];

      // Note: In general a tile contains a reflector per column, but in case there aren't enough
      // rows for storing a reflector whose length is greater than 1, then the number is reduced
      // accordingly. An exception is represented by the last bottom right tile, where the refelctors
      // are all the remaining ones (i.e. there may be a length 1 reflector in complex cases)
      const SizeType k_refls = (j != j_last_sweep) ? std::min(v_rows - 1, mb) : (nsweeps - j * mb);

      const matrix::SubTileSpec v_spec{{0, 0}, {v_rows, k_refls}};
      const matrix::SubTileSpec v_up{{0, 0}, {sizes[0], k_refls}};
      const matrix::SubTileSpec v_dn{{sizes[0], 0}, {sizes[1], k_refls}};

      if (k_refls < mat_i.tileSize({i, j}).cols()) {
        mat_w.setWidth(k_refls);
        mat_t.setWidth(k_refls);
        mat_w2.setHeight(k_refls);
      }

      // TODO setRange? it would mean setting the range to a specific tile for each step, and resetting at the end

      // Note:
      // Let's use W as a temporary storage for the well formed version of V, which is needed
      // for computing W2 = V . E, and it is also useful for computing T
      auto tile_w_full = mat_w(ij_refls);
      auto tile_w_rw = splitTile(tile_w_full, v_spec);

      const auto& tile_i = mat_i.read(ij_refls);
      auto tile_v = setupVWellFormed(k_refls, tile_i, std::move(tile_w_rw));

      // W2 = V* . E
      // Note:
      // Since the well-formed V is stored in W, we have to use it before W will get overwritten.
      // For this reason W2 is computed before W.
      for (SizeType j_e = 0; j_e < mat_e.nrTiles().cols(); ++j_e) {
        const auto sz_e = mat_e.tileSize({i, j_e});
        auto tile_e = mat_e.read(LocalTileIndex(i, j_e));

        computeW2<backend>(thread_priority::normal, keep_future(splitTile(tile_v, v_up)),
                           keep_future(splitTile(tile_e, {{1, 0}, {sizes[0], sz_e.cols()}})), T(0),
                           mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));

        if (affectsTwoRows)
          computeW2<backend>(thread_priority::normal, keep_future(splitTile(tile_v, v_dn)),
                             mat_e.read_sender(LocalTileIndex(i + 1, j_e)), T(1),
                             mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));
      }

      // Note:
      // And we use it also for computing the T factor
      auto tile_t_full = mat_t(LocalTileIndex(i, 0));
      auto tile_t = computeTFactor(tile_i, tile_v, splitTile(tile_t_full, {{0, 0}, {k_refls, k_refls}}));

      // W = V . T
      // Note:
      // At this point W can be overwritten, but this will happen just after W2 and T computations
      // finished. T was already a dependency, but W2 wasn't, meaning it won't run in parallel,
      // but it could.
      tile_w_rw = splitTile(tile_w_full, v_spec);
      computeW<backend>(thread_priority::normal, std::move(tile_w_rw), keep_future(tile_t));
      auto tile_w = mat_w.read(ij_refls);

      // E -= W . W2
      for (SizeType j_e = 0; j_e < mat_e.nrTiles().cols(); ++j_e) {
        const LocalTileIndex idx_e(i, j_e);
        const auto sz_e = mat_e.tileSize({i, j_e});

        auto tile_e = mat_e(idx_e);
        const auto& tile_w2 = mat_w2.read_sender(idx_e);

        updateE<backend>(hpx::threads::thread_priority::normal, keep_future(splitTile(tile_w, v_up)),
                         tile_w2, splitTile(tile_e, {{1, 0}, {sizes[0], sz_e.cols()}}));

        if (affectsTwoRows)
          updateE<backend>(hpx::threads::thread_priority::normal, keep_future(splitTile(tile_w, v_dn)),
                           tile_w2, mat_e.readwrite_sender(LocalTileIndex{i + 1, j_e}));
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
