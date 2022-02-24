//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <type_traits>

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/thread.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/eigensolver/band_to_tridiag/api.h"
#include "dlaf/eigensolver/bt_band_to_tridiag/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/traits.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver::internal {

template <class T>
struct BackTransformationT2B<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_e, Matrix<const T, Device::CPU>& mat_hh);
};

template <class T>
pika::shared_future<matrix::Tile<const T, Device::CPU>> setupVWellFormed(
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_i,
    pika::future<matrix::Tile<T, Device::CPU>> tile_v) {
  namespace ex = pika::execution::experimental;
  using lapack::lacpy;
  using lapack::laset;

  auto unzipV_func = [](const auto& tile_i, auto tile_v) {
    // Note: the size of of tile_i and tile_v embeds a relevant information about the number of
    // reflecotrs and their max size. This will be exploited to correctly setup the well formed
    // tile with reflectors in place as they will be applied.
    const auto k = tile_v.size().cols();

    // copy from compact representation reflector values (the first component set to 1 is not there)
    for (SizeType j = 0; j < k; ++j) {
      const auto compact_refl_size =
          std::min<SizeType>(tile_v.size().rows() - (1 + j), tile_i.size().rows() - 1);

      // this is needed because of complex last reflector (i.e. just 1 element long)
      if (compact_refl_size == 0)
        continue;

      lacpy(blas::Uplo::General, compact_refl_size, 1, tile_i.ptr({1, j}), tile_i.ld(),
            tile_v.ptr({1 + j, j}), tile_v.ld());
    }

    // Note:
    // In addition to setting the diagonal to 1 for missing first components, here it zeros out
    // both the upper and the lower part. Indeed due to the skewed shape, reflectors do not occupy
    // the full tile height, and V should be fully well-formed because the next triangular
    // multiplication, i.e. `V . T`, and the gemm `V* . E`, will use V as a general matrix.
    laset(blas::Uplo::Upper, tile_v.size().rows(), k, T(0), T(1), tile_v.ptr({0, 0}), tile_v.ld());

    const SizeType mb = tile_i.size().cols();
    if (tile_v.size().rows() > mb)
      laset(blas::Uplo::Lower, tile_v.size().rows() - mb, k - 1, T(0), T(0), tile_v.ptr({mb, 0}),
            tile_v.ld());

    return matrix::Tile<const T, Device::CPU>(std::move(tile_v));
  };
  return ex::make_future(
      dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), unzipV_func,
                                ex::when_all(ex::keep_future(std::move(tile_i)), std::move(tile_v))));
}

template <class T>
pika::shared_future<matrix::Tile<const T, Device::CPU>> computeTFactor(
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_taus,
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
    pika::future<matrix::Tile<T, Device::CPU>> mat_t) {
  namespace ex = pika::execution::experimental;

  auto tfactor_task = [](const auto& tile_taus, const auto& tile_v, auto tile_t) {
    using namespace lapack;

    // taus have to be extracted from the compact form (i.e. first row of the input tile)
    std::vector<T> taus;
    taus.resize(to_sizet(tile_v.size().cols()));
    for (SizeType i = 0; i < to_SizeType(taus.size()); ++i)
      taus[to_sizet(i)] = tile_taus({0, i});

    const auto n = tile_v.size().rows();
    const auto k = tile_v.size().cols();
    larft(Direction::Forward, StoreV::Columnwise, n, k, tile_v.ptr(), tile_v.ld(), taus.data(),
          tile_t.ptr(), tile_t.ld());

    return matrix::Tile<const T, Device::CPU>(std::move(tile_t));
  };
  return ex::make_future(
      dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), tfactor_task,
                                ex::when_all(ex::keep_future(std::move(tile_taus)),
                                             ex::keep_future(std::move(tile_v)), std::move(mat_t))));
}

template <Backend backend, class VSender, class TSender>
auto computeW(pika::threads::thread_priority priority, VSender&& tile_v, TSender&& tile_t) {
  using namespace blas;
  using T = dlaf::internal::SenderElementType<VSender>;
  dlaf::internal::whenAllLift(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1),
                              std::forward<TSender>(tile_t), std::forward<VSender>(tile_v)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class VSender, class ESender, class T, class W2Sender>
auto computeW2(pika::threads::thread_priority priority, VSender&& tile_v, ESender&& tile_e, T beta,
               W2Sender&& tile_w2) {
  using blas::Op;
  dlaf::internal::whenAllLift(Op::ConjTrans, Op::NoTrans, T(1), std::forward<VSender>(tile_v),
                              std::forward<ESender>(tile_e), beta, std::forward<W2Sender>(tile_w2)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <Backend backend, class WSender, class W2Sender, class ESender>
auto updateE(pika::threads::thread_priority priority, WSender&& tile_w, W2Sender&& tile_w2,
             ESender&& tile_e) {
  using blas::Op;
  using T = dlaf::internal::SenderElementType<ESender>;
  dlaf::internal::whenAllLift(Op::NoTrans, Op::NoTrans, T(-1), std::forward<WSender>(tile_w),
                              std::forward<W2Sender>(tile_w2), T(1), std::forward<ESender>(tile_e)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <class T>
void BackTransformationT2B<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_e,
                                                              Matrix<const T, Device::CPU>& mat_hh) {
  static constexpr Backend backend = Backend::MC;
  using pika::threads::thread_priority;
  using pika::execution::experimental::keep_future;

  using matrix::Panel;
  using common::RoundRobin;
  using common::iterate_range2d;

  if (mat_hh.size().isEmpty() || mat_e.size().isEmpty())
    return;

  const SizeType mb = mat_e.blockSize().rows();
  const SizeType b = mb;
  const SizeType nsweeps = nrSweeps<T>(mat_hh.size().cols());

  const auto& dist_i = mat_hh.distribution();

  // Note: w_tile_sz can store reflectors are they are actually applied, opposed to how they are
  // stored in compact form.
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

  // Note: The above formula is valid until it meets the constraints imposed by the matrix which
  // reflectors are applied to. Indeed, `2b-1` means `b-1` rows refers to the current row tile and
  // `b` elements to the next one.
  //
  // For sure last tile will not have room at all for the second part of the formula, because there is no
  // next tile since it is the last. Moreover, it may not have space for the first part of the formula,
  // e.g. when matrix size is not a multiple of its blocksize, so its dimension can be tailored according
  // to the number of reflectors that fits there (considering also the last reflector of size 1 for
  // complex martrices).
  //
  // Additionally, the constraints may affect also the before last tile of W. Indeed, it will have
  // for sure room for the first part of the formula, but the second part may be limited by the size
  // of the last tile of E.
  //
  // Considering all the cases, it may end up:
  // - last tile is full (matrix size is a multiple of blocksize), so just this last one is reduced;
  // - last tile is incomplete, last two tiles are both reduced in size (not the same)
  //
  // In this latter case, since the blocksize is fixed for a matrix, we can just limit the size of
  // the last tile by constraining the size of the matrix containing it. This means that the before
  // last tile will have a full size even if it will not be fully used, while the last one can be
  // reduced as needed.
  //
  // But, there is one last case: if the last tile of E does not have enough room for applying
  // reflectors, it will result that in W the last tile will be the one linked with the before last tile
  // in E, so we can actually reduce its size.
  const auto last_tile_size =
      mat_hh.tileSize(indexFromOrigin(mat_hh.nrTiles() - GlobalTileSize{1, 1})).rows();
  const SizeType last_w_tile_sz = [&]() {
    const auto last_refl_size = last_tile_size - 1;
    if (last_refl_size == 1)
      return isComplex_v<T> ? SizeType(1) : SizeType(0);
    return std::max<SizeType>(0, last_refl_size);
  }();

  const SizeType dist_w_rows = [&]() {
    if (last_w_tile_sz != 0)
      return (mat_e.nrTiles().rows() - 1) * w_tile_sz.rows() + last_w_tile_sz;
    return (mat_e.nrTiles().rows() - 2) * w_tile_sz.rows() + (b - 1) + last_tile_size;
  }();

  const matrix::Distribution dist_w({dist_w_rows, w_tile_sz.cols()}, w_tile_sz, dist_i.commGridSize(),
                                    dist_i.rankIndex(), dist_i.sourceRankIndex());

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> t_panels(n_workspaces, mat_hh.distribution());
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> w_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Row, T, Device::CPU>> w2_panels(n_workspaces, mat_e.distribution());

  // Note: sweep are on diagonals, steps are on verticals
  const SizeType j_last_sweep = (nsweeps - 1) / mb;
  for (SizeType j = j_last_sweep; j >= 0; --j) {
    auto& mat_w = w_panels.nextResource();
    auto& mat_t = t_panels.nextResource();
    auto& mat_w2 = w2_panels.nextResource();

    // Note: apply the entire column (steps)
    const SizeType steps = nrStepsForSweep(j * mb, mat_hh.size().cols(), mb);
    for (SizeType step = 0; step < steps; ++step) {
      const SizeType i = j + step;
      const LocalTileIndex ij_refls(i, j);

      const bool affectsTwoRows = i < (mat_hh.nrTiles().rows() - 1);

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
          mat_hh.tileSize({i, j}).rows() - 1,
          affectsTwoRows ? mat_hh.tileSize({i + 1, j}).rows() : 0,
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

      if (k_refls < mat_hh.tileSize({i, j}).cols()) {
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

      const auto& tile_i = mat_hh.read(ij_refls);
      auto tile_v = setupVWellFormed(tile_i, std::move(tile_w_rw));

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

        updateE<backend>(pika::threads::thread_priority::normal, keep_future(splitTile(tile_w, v_up)),
                         tile_w2, splitTile(tile_e, {{1, 0}, {sizes[0], sz_e.cols()}}));

        if (affectsTwoRows)
          updateE<backend>(pika::threads::thread_priority::normal, keep_future(splitTile(tile_w, v_dn)),
                           tile_w2, mat_e.readwrite_sender(LocalTileIndex{i + 1, j_e}));
      }

      mat_t.reset();
      mat_w.reset();
      mat_w2.reset();
    }
  }
}

/// ---- ETI
#define DLAF_EIGENSOLVER_BACKTRANSFORMATION_B2T_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct BackTransformationT2B<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_BACKTRANSFORMATION_B2T_MC_ETI(extern, float)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_B2T_MC_ETI(extern, double)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_B2T_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_BACKTRANSFORMATION_B2T_MC_ETI(extern, std::complex<double>)

}
