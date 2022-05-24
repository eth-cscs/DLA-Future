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

#include <tuple>
#include <type_traits>

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/thread.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/assert.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/eigensolver/band_to_tridiag/api.h"
#include "dlaf/eigensolver/bt_band_to_tridiag/api.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/traits.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_math.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver::internal {

namespace bt_tridiag {

template <class T>
matrix::Tile<const T, Device::CPU> task_setupVWellFormed(
    const SizeType b, const matrix::Tile<const T, Device::CPU>& tile_hh,
    matrix::Tile<T, Device::CPU> tile_v) {
  using lapack::lacpy;
  using lapack::laset;

  // Note: the size of of tile_hh and tile_v embeds a relevant information about the number of
  // reflecotrs and their max size. This will be exploited to correctly setup the well formed
  // tile with reflectors in place as they will be applied.
  const auto k = tile_v.size().cols();

  // copy from compact representation reflector values (the first component set to 1 is not there)
  for (SizeType j = 0; j < k; ++j) {
    const auto compact_refl_size =
        std::min<SizeType>(tile_v.size().rows() - (1 + j), tile_hh.size().rows() - 1);

    // this is needed because of complex last reflector (i.e. just 1 element long)
    if (compact_refl_size == 0)
      continue;

    lacpy(blas::Uplo::General, compact_refl_size, 1, tile_hh.ptr({1, j}), tile_hh.ld(),
          tile_v.ptr({1 + j, j}), tile_v.ld());
  }

  // Note:
  // In addition to setting the diagonal to 1 for missing first components, here it zeros out
  // both the upper and the lower part. Indeed due to the skewed shape, reflectors do not occupy
  // the full tile height, and V should be fully well-formed because the next triangular
  // multiplication, i.e. `V . T`, and the gemm `V* . E`, will use V as a general matrix.
  laset(blas::Uplo::Upper, tile_v.size().rows(), k, T(0), T(1), tile_v.ptr({0, 0}), tile_v.ld());

  if (tile_v.size().rows() > b)
    laset(blas::Uplo::Lower, tile_v.size().rows() - b, k - 1, T(0), T(0), tile_v.ptr({b, 0}),
          tile_v.ld());

  return matrix::Tile<const T, Device::CPU>(std::move(tile_v));
}

template <class T>
pika::shared_future<matrix::Tile<const T, Device::CPU>> setupVWellFormed(
    const SizeType b, pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_hh,
    pika::future<matrix::Tile<T, Device::CPU>> tile_v) {
  namespace ex = pika::execution::experimental;

  return dlaf::internal::whenAllLift(b, ex::keep_future(std::move(tile_hh)), std::move(tile_v)) |
         dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), task_setupVWellFormed<T>) |
         ex::make_future();
}

template <class T>
matrix::Tile<const T, Device::CPU> task_computeTFactor(
    const matrix::Tile<const T, Device::CPU>& tile_taus,
    const matrix::Tile<const T, Device::CPU>& tile_v, matrix::Tile<T, Device::CPU> tile_t) {
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
}

template <class T>
pika::shared_future<matrix::Tile<const T, Device::CPU>> computeTFactor(
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_taus,
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
    pika::future<matrix::Tile<T, Device::CPU>> mat_t) {
  namespace ex = pika::execution::experimental;

  return dlaf::internal::whenAllLift(ex::keep_future(std::move(tile_taus)),
                                     ex::keep_future(std::move(tile_v)), std::move(mat_t)) |
         dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), task_computeTFactor<T>) |
         ex::make_future();
}

template <Backend backend, class TSender, class VSender, class WSender>
auto computeW(pika::threads::thread_priority priority, TSender&& tile_t, VSender&& tile_v,
              WSender&& tile_w) {
  using namespace blas;
  using T = dlaf::internal::SenderElementType<WSender>;
  dlaf::internal::whenAllLift(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1),
                              std::forward<TSender>(tile_t), std::forward<VSender>(tile_v),
                              std::forward<WSender>(tile_w)) |
      tile::trmm3(dlaf::internal::Policy<backend>(priority)) |
      pika::execution::experimental::start_detached();
}

template <class T>
std::tuple<matrix::Tile<const T, Device::CPU>, matrix::Tile<const T, Device::CPU>,
           matrix::Tile<const T, Device::CPU>>
computeVTW(const SizeType b, const matrix::Tile<const T, Device::CPU>& tile_hh,
           matrix::Tile<T, Device::CPU> tile_v, matrix::Tile<T, Device::CPU> tile_t,
           matrix::Tile<T, Device::CPU> tile_w) {
  auto tile_v_c = task_setupVWellFormed(b, tile_hh, std::move(tile_v));
  auto tile_t_c = task_computeTFactor(tile_hh, tile_v_c, std::move(tile_t));
  using namespace blas;
  dlaf::tile::internal::trmm3_o(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), tile_t_c,
                                tile_v_c, tile_w);
  return std::make_tuple(std::move(tile_v_c), std::move(tile_t_c), std::move(tile_w));
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

struct TileAccessHelper {
  TileAccessHelper(const SizeType b, const SizeType nrefls, matrix::Distribution dist,
                   const GlobalElementIndex offset)
      : nrefls_(nrefls), input_spec_{dist.tileElementIndex(offset),
                                     {std::min(b, dist.size().rows() - offset.row()),
                                      std::min(b, dist.size().cols() - offset.col())}} {
    const TileElementIndex& sub_offset = input_spec_.origin;

    // Note:
    // Next logic is about detecting the available application space for reflectors, which once
    // extracted they expands to a matrix with 2 * b - 1 height.
    //
    // There are two main scenarios:
    // - reflectors involves rows of a single tile;
    // - reflectors involves rows across two different tiles.
    //
    // In both scenarios, it may happen that all reflectors cannot be fully applied to to matrix
    // size constraint.

    const SizeType fullsize = 2 * b - 1;
    const SizeType mb = dist.blockSize().rows();
    const SizeType rows_below = dist.size().rows() - offset.row();

    // Note:
    // In general, keep in mind that to sub_offset.row() is applied a + 1, to take into account the
    // offset about how reflectors are actually applied.
    //
    // e.g. b = 4
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
    // the upper tile is not affected.

    acrossTiles_ = [&]() {
      // Note:
      // A single tile is involved if:
      // - it is the last row tile, so by construction reflectors will be applied to a single tile;
      // - applying reflectors falls entirely withing a single tile
      const SizeType i_tile = dist.globalTileFromGlobalElement<Coord::Row>(offset.row());
      const bool is_last_row_tile = i_tile == dist.nrTiles().rows() - 1;
      return !(is_last_row_tile || sub_offset.row() + 1 + fullsize <= mb);
    }();

    if (acrossTiles_) {
      const auto part0_nrows = b - 1;
      part_top_ = Part(sub_offset.row() + 1, part0_nrows, nrefls);
      part_bottom_ = Part(0, std::min(b, rows_below - b), nrefls, part0_nrows);
    }
    else {
      part_top_ = Part(sub_offset.row() + 1, std::min(fullsize, rows_below - 1), nrefls);
    }
  };

  // Return true if the application of Householder reflectors involves multiple tiles
  bool affectsMultipleTiles() const noexcept {
    return acrossTiles_;
  }

  // Return SubTileSpec to use for accessing Householder reflectors in compact form
  matrix::SubTileSpec specHHCompact() const noexcept {
    return input_spec_;
  }

  // Return SubTileSpec to use for accessing Householder reflectors in well formed form
  matrix::SubTileSpec specHH() const noexcept {
    return {{0, 0}, {part_top_.rows() + part_bottom_.rows(), nrefls_}};
  }

  const auto& topPart() const noexcept {
    return part_top_;
  }

  const auto& bottomPart() const noexcept {
    DLAF_ASSERT_MODERATE(affectsMultipleTiles(), affectsMultipleTiles());
    return part_bottom_;
  }

private:
  struct Part {
    // Create an "empty" part
    Part() = default;

    Part(const SizeType origin, const SizeType nrows, const SizeType nrefls, const SizeType offset = 0)
        : origin_(origin), nrows_(nrows), nrefls_(nrefls), offset_(offset) {}

    // Return the number of rows that this part involves
    SizeType rows() const noexcept {
      return nrows_;
    }

    // Return the spec to use on well-formed HH
    matrix::SubTileSpec specHH() const noexcept {
      return {{offset_, 0}, {nrows_, nrefls_}};
    }

    // Return the spec to use on the matrix HH are applied to
    matrix::SubTileSpec specEV(const SizeType cols) const noexcept {
      return {{origin_, 0}, {nrows_, cols}};
    }

  private:
    SizeType origin_;
    SizeType nrows_ = 0;
    SizeType nrefls_ = 0;
    SizeType offset_;
  };

  SizeType nrefls_;
  matrix::SubTileSpec input_spec_;

  bool acrossTiles_;
  Part part_top_;
  Part part_bottom_;
};

template <Backend B, Device D, class T>
struct HHManager;

template <class T>
struct HHManager<Backend::MC, Device::CPU, T> {
  static constexpr Backend B = Backend::MC;
  static constexpr Device D = Device::CPU;

  HHManager(const SizeType b, const std::size_t, matrix::Distribution, matrix::Distribution) : b(b) {}

  auto setupVAndComputeT(const LocalTileIndex ij, const SizeType nrefls, const TileAccessHelper& helper,
                         matrix::Matrix<const T, Device::CPU>& mat_hh,
                         matrix::Panel<Coord::Col, T, D>& mat_v, matrix::Panel<Coord::Col, T, D>& mat_t,
                         matrix::Panel<Coord::Col, T, D>& mat_w) {
    namespace ex = pika::execution::experimental;

    const matrix::SubTileSpec t_spec{{0, 0}, {nrefls, nrefls}};
    const LocalTileIndex ij_t(LocalTileIndex(ij.row(), 0));

    auto tup =
        dlaf::internal::whenAllLift(b,
                                    ex::keep_future(splitTile(mat_hh.read(ij), helper.specHHCompact())),
                                    splitTile(mat_v(ij), helper.specHH()),
                                    splitTile(mat_t(ij_t), t_spec),
                                    splitTile(mat_w(ij), helper.specHH())) |
        dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), computeVTW<T>) |
        ex::make_future();

    return pika::split_future(std::move(tup));
  }

protected:
  const SizeType b;
};

#ifdef DLAF_WITH_CUDA
template <class T>
struct HHManager<Backend::GPU, Device::GPU, T> {
  static constexpr Backend B = Backend::GPU;
  static constexpr Device D = Device::GPU;

  HHManager(const SizeType b, const std::size_t n_workspaces, matrix::Distribution dist_t,
            matrix::Distribution dist_w)
      : b(b), t_panels_h(n_workspaces, dist_t), w_panels_h(n_workspaces, dist_w) {}

  auto setupVAndComputeT(const LocalTileIndex ij, const SizeType nrefls, const TileAccessHelper& helper,
                         matrix::Matrix<const T, Device::CPU>& mat_hh,
                         matrix::Panel<Coord::Col, T, D>& mat_v, matrix::Panel<Coord::Col, T, D>& mat_t,
                         matrix::Panel<Coord::Col, T, D>& mat_w) {
    namespace ex = pika::execution::experimental;

    auto& mat_v_h = w_panels_h.nextResource();
    auto& mat_t_h = t_panels_h.nextResource();

    auto tile_hh = splitTile(mat_hh.read(ij), helper.specHHCompact());

    auto tile_v_h = setupVWellFormed(b, tile_hh, splitTile(mat_v_h(ij), helper.specHH()));
    auto tile_v = scheduleCopy(tile_v_h, mat_v, ij, helper.specHH());

    const LocalTileIndex ij_t(ij.row(), 0);
    const matrix::SubTileSpec t_spec = {{0, 0}, {nrefls, nrefls}};

    auto tile_t_h = computeTFactor(tile_hh, tile_v_h, splitTile(mat_t_h(ij_t), t_spec));
    auto tile_t = scheduleCopy(tile_t_h, mat_t, ij_t, t_spec);

    // W = V . T
    using pika::threads::thread_priority;
    computeW<B>(thread_priority::normal, ex::keep_future(tile_t), ex::keep_future(tile_v),
                splitTile(mat_w(ij), helper.specHH()));

    return std::make_tuple(std::move(tile_v), std::move(tile_t), mat_w.read(ij));
  }

protected:
  auto scheduleCopy(pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_src,
                    matrix::Panel<Coord::Col, T, D>& mat, const LocalTileIndex ij,
                    const matrix::SubTileSpec spec) const {
    namespace ex = pika::execution::experimental;

    constexpr auto copy_backend = dlaf::matrix::internal::CopyBackend_v<Device::CPU, D>;

    auto fulltile = mat(ij);
    auto tile_dst = splitTile(fulltile, spec);

    dlaf::internal::whenAllLift(ex::keep_future(tile_src), std::move(tile_dst)) |
        matrix::copy(dlaf::internal::Policy<copy_backend>()) | ex::start_detached();

    return splitTile(mat.read(ij), spec);
  }

  const SizeType b;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> t_panels_h;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> w_panels_h;
};
#endif
}

template <Backend B, Device D, class T>
void BackTransformationT2B<B, D, T>::call(const SizeType band_size, Matrix<T, D>& mat_e,
                                          Matrix<const T, Device::CPU>& mat_hh) {
  using pika::threads::thread_priority;
  namespace ex = pika::execution::experimental;

  using common::iterate_range2d;
  using common::RoundRobin;
  using matrix::Panel;
  using namespace bt_tridiag;

  if (mat_hh.size().isEmpty() || mat_e.size().isEmpty())
    return;

  // Note: if no householder reflectors are going to be applied (in case of trivial matrix)
  if (mat_hh.size().rows() <= (dlaf::isComplex_v<T> ? 1 : 2))
    return;

  const SizeType b = band_size;
  const SizeType nsweeps = nrSweeps<T>(mat_hh.size().cols());

  const auto& dist_hh = mat_hh.distribution();

  // Note: w_tile_sz can store reflectors as they are actually applied, opposed to how they are
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

  const SizeType dist_w_rows = mat_e.nrTiles().rows() * w_tile_sz.rows();
  const matrix::Distribution dist_w({dist_w_rows, b}, w_tile_sz);
  const matrix::Distribution dist_t({mat_hh.size().rows(), b}, {b, b});
  const matrix::Distribution dist_w2({b, mat_e.size().cols()}, {b, mat_e.blockSize().cols()});

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, D>> t_panels(n_workspaces, dist_t);
  RoundRobin<Panel<Coord::Col, T, D>> v_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Col, T, D>> w_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Row, T, D>> w2_panels(n_workspaces, dist_w2);

  HHManager<B, D, T> helperBackend(b, n_workspaces, dist_t, dist_w);

  // Note: sweep are on diagonals, steps are on verticals
  const SizeType j_last_sweep = (nsweeps - 1) / b;
  for (SizeType j = j_last_sweep; j >= 0; --j) {
    auto& mat_t = t_panels.nextResource();
    auto& mat_v = v_panels.nextResource();
    auto& mat_w = w_panels.nextResource();
    auto& mat_w2 = w2_panels.nextResource();

    // Note: apply the entire column (steps)
    const SizeType steps = nrStepsForSweep(j * b, mat_hh.size().cols(), b);
    for (SizeType step = 0; step < steps; ++step) {
      const SizeType i = j + step;

      const GlobalElementIndex ij_e(i * b, j * b);
      const LocalTileIndex ij(dist_hh.localTileIndex(dist_hh.globalTileIndex(ij_e)));

      // Note:  reflector with size = 1 must be ignored, except for the last step of the last sweep
      //        with complex type
      const SizeType nrefls = [&]() {
        const bool allowSize1 = isComplex_v<T> && j == j_last_sweep && step == steps - 1;
        const GlobalElementSize delta(dist_hh.size().rows() - ij_e.row() - 1,
                                      std::min(b, dist_hh.size().cols() - ij_e.col()));
        return std::min(b, std::min(delta.rows() - (allowSize1 ? 0 : 1), delta.cols()));
      }();

      const TileAccessHelper helper(b, nrefls, mat_hh.distribution(), ij_e);

      if (nrefls < b) {
        mat_t.setWidth(nrefls);
        mat_v.setWidth(nrefls);
        mat_w.setWidth(nrefls);
        mat_w2.setHeight(nrefls);
      }

      // TODO setRange? it would mean setting the range to a specific tile for each step, and resetting at the end

      auto [tile_v, tile_t, tile_w] =
          helperBackend.setupVAndComputeT(ij, nrefls, helper, mat_hh, mat_v, mat_t, mat_w);

      // W2 = V* . E
      for (SizeType j_e = 0; j_e < mat_e.nrTiles().cols(); ++j_e) {
        const auto sz_e = mat_e.tileSize({ij.row(), j_e});
        auto tile_e = mat_e.read(LocalTileIndex(ij.row(), j_e));

        computeW2<B>(thread_priority::normal,
                     ex::keep_future(splitTile(tile_v, helper.topPart().specHH())),
                     ex::keep_future(splitTile(tile_e, helper.topPart().specEV(sz_e.cols()))), T(0),
                     mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));

        if (helper.affectsMultipleTiles()) {
          auto tile_e = mat_e.read(LocalTileIndex(ij.row() + 1, j_e));
          computeW2<B>(thread_priority::normal,
                       ex::keep_future(splitTile(tile_v, helper.bottomPart().specHH())),
                       ex::keep_future(splitTile(tile_e, helper.bottomPart().specEV(sz_e.cols()))), T(1),
                       mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));
        }
      }

      // E -= W . W2
      for (SizeType j_e = 0; j_e < mat_e.nrTiles().cols(); ++j_e) {
        const LocalTileIndex idx_e(ij.row(), j_e);
        const auto sz_e = mat_e.tileSize({ij.row(), j_e});

        const auto& tile_w2 = mat_w2.read_sender(idx_e);

        updateE<B>(pika::threads::thread_priority::normal,
                   ex::keep_future(splitTile(tile_w, helper.topPart().specHH())), tile_w2,
                   splitTile(mat_e(idx_e), helper.topPart().specEV(sz_e.cols())));

        if (helper.affectsMultipleTiles()) {
          updateE<B>(pika::threads::thread_priority::normal,
                     ex::keep_future(splitTile(tile_w, helper.bottomPart().specHH())), tile_w2,
                     splitTile(mat_e(LocalTileIndex{ij.row() + 1, j_e}),
                               helper.bottomPart().specEV(sz_e.cols())));
        }
      }

      mat_t.reset();
      mat_v.reset();
      mat_w.reset();
      mat_w2.reset();
    }
  }
}

}
