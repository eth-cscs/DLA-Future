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

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include "dlaf/blas/tile.h"
#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels/broadcast.h"
#include "dlaf/communication/kernels/p2p.h"
#include "dlaf/communication/kernels/p2p_allsum.h"
#include "dlaf/eigensolver/band_to_tridiag/api.h"
#include "dlaf/eigensolver/bt_band_to_tridiag/api.h"
#include "dlaf/matrix/copy_tile.h"
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
matrix::Tile<const T, Device::CPU> setupVWellFormed(const SizeType b,
                                                    const matrix::Tile<const T, Device::CPU>& tile_hh,
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
matrix::Tile<const T, Device::CPU> computeTFactor(const matrix::Tile<const T, Device::CPU>& tile_taus,
                                                  const matrix::Tile<const T, Device::CPU>& tile_v,
                                                  matrix::Tile<T, Device::CPU> tile_t) {
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
std::tuple<matrix::Tile<const T, Device::CPU>, matrix::Tile<const T, Device::CPU>> computeVT(
    const SizeType b, const matrix::Tile<const T, Device::CPU>& tile_hh,
    matrix::Tile<T, Device::CPU> tile_v, matrix::Tile<T, Device::CPU> tile_t) {
  auto tile_v_c = setupVWellFormed(b, tile_hh, std::move(tile_v));
  auto tile_t_c = computeTFactor(tile_hh, tile_v_c, std::move(tile_t));
  return std::make_tuple(std::move(tile_v_c), std::move(tile_t_c));
}

template <class T>
std::tuple<matrix::Tile<const T, Device::CPU>, matrix::Tile<const T, Device::CPU>> computeVW(
    const SizeType b, const matrix::Tile<const T, Device::CPU>& tile_hh,
    matrix::Tile<T, Device::CPU> tile_v, matrix::Tile<T, Device::CPU> tile_t,
    matrix::Tile<T, Device::CPU> tile_w) {
  using namespace blas;

  auto [tile_v_c, tile_t_c] = computeVT(b, tile_hh, std::move(tile_v), std::move(tile_t));

  dlaf::tile::internal::trmm3(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), tile_t_c,
                              tile_v_c, tile_w);
  return std::make_tuple(std::move(tile_v_c), std::move(tile_w));
}

template <Backend B, class T>
struct ApplyHHToSingleTileRow;

template <class T>
struct ApplyHHToSingleTileRow<Backend::MC, T> {
  void operator()(const matrix::Tile<const T, Device::CPU>& tile_v,
                  const matrix::Tile<const T, Device::CPU>& tile_w, matrix::Tile<T, Device::CPU> tile_w2,
                  const matrix::Tile<T, Device::CPU>& tile_e) {
    using namespace blas;
    using tile::internal::gemm;
    // W2 = V* . E
    gemm(Op::ConjTrans, Op::NoTrans, T(1), tile_v, tile_e, T(0), tile_w2);
    // E -= W . W2
    gemm(Op::NoTrans, Op::NoTrans, T(-1), tile_w, tile_w2, T(1), tile_e);
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct ApplyHHToSingleTileRow<Backend::GPU, T> {
  void operator()(cublasHandle_t handle, const matrix::Tile<const T, Device::GPU>& tile_v,
                  const matrix::Tile<const T, Device::GPU>& tile_w,
                  const matrix::Tile<T, Device::GPU>& tile_w2,
                  const matrix::Tile<T, Device::GPU>& tile_e) {
    using namespace blas;
    using tile::internal::gemm;
    // W2 = V* . E
    gemm(handle, Op::ConjTrans, Op::NoTrans, T(1), tile_v, tile_e, T(0), tile_w2);
    // E -= W . W2
    gemm(handle, Op::NoTrans, Op::NoTrans, T(-1), tile_w, tile_w2, T(1), tile_e);
  }
};
#endif

template <Backend B, class T>
struct ApplyHHToDoubleTileRow;

template <class T>
struct ApplyHHToDoubleTileRow<Backend::MC, T> {
  void operator()(const matrix::Tile<const T, Device::CPU>& tile_v0,
                  const matrix::Tile<const T, Device::CPU>& tile_v1,
                  const matrix::Tile<const T, Device::CPU>& tile_w0,
                  const matrix::Tile<const T, Device::CPU>& tile_w1,
                  const matrix::Tile<T, Device::CPU>& tile_w2,
                  const matrix::Tile<T, Device::CPU>& tile_e_top,
                  const matrix::Tile<T, Device::CPU>& tile_e_bottom) {
    using namespace blas;
    using tile::internal::gemm;
    // W2 = V* . E
    gemm(Op::ConjTrans, Op::NoTrans, T(1), tile_v0, tile_e_top, T(0), tile_w2);
    gemm(Op::ConjTrans, Op::NoTrans, T(1), tile_v1, tile_e_bottom, T(1), tile_w2);
    // E -= W . W2
    gemm(Op::NoTrans, Op::NoTrans, T(-1), tile_w0, tile_w2, T(1), tile_e_top);
    gemm(Op::NoTrans, Op::NoTrans, T(-1), tile_w1, tile_w2, T(1), tile_e_bottom);
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct ApplyHHToDoubleTileRow<Backend::GPU, T> {
  void operator()(cublasHandle_t handle, const matrix::Tile<const T, Device::GPU>& tile_v0,
                  const matrix::Tile<const T, Device::GPU>& tile_v1,
                  const matrix::Tile<const T, Device::GPU>& tile_w0,
                  const matrix::Tile<const T, Device::GPU>& tile_w1,
                  const matrix::Tile<T, Device::GPU>& tile_w2,
                  const matrix::Tile<T, Device::GPU>& tile_e_top,
                  const matrix::Tile<T, Device::GPU>& tile_e_bottom) {
    using namespace blas;
    using tile::internal::gemm;
    // W2 = V* . E
    gemm(handle, Op::ConjTrans, Op::NoTrans, T(1), tile_v0, tile_e_top, T(0), tile_w2);
    gemm(handle, Op::ConjTrans, Op::NoTrans, T(1), tile_v1, tile_e_bottom, T(1), tile_w2);
    // E -= W . W2
    gemm(handle, Op::NoTrans, Op::NoTrans, T(-1), tile_w0, tile_w2, T(1), tile_e_top);
    gemm(handle, Op::NoTrans, Op::NoTrans, T(-1), tile_w1, tile_w2, T(1), tile_e_bottom);
  }
};
#endif

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
  //
  // SubTileSpec returned points to the sub-block in the full tile containing the HH data in compact
  // form. If @p reset_origin is true, then the origin component of the SubTileSpec is resetted and
  // it will just describe the size of the sub-block containing the Householder reflectors (useful
  // for panel access which might not have full-tiles).
  matrix::SubTileSpec specHHCompact(const bool reset_origin = false) const noexcept {
    if (reset_origin)
      return {{0, 0}, input_spec_.size};
    return input_spec_;
  }

  // Return SubTileSpec to use for accessing Householder reflectors in well formed form
  matrix::SubTileSpec specHH() const noexcept {
    return {{0, 0}, {part_top_.rows() + part_bottom_.rows(), nrefls_}};
  }

  // Return SubTileSpec to use for accessing T factor
  matrix::SubTileSpec specT() const noexcept {
    return {{0, 0}, {nrefls_, nrefls_}};
  }

  matrix::SubTileSpec specW2(const SizeType cols) const noexcept {
    return {{0, 0}, {nrefls_, cols}};
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

/// Note:
/// This is a helper specific for the distributed version of the algorithm. It provides:
/// - helper for checking if a rank is involved, as main or partner, in a given step of the algorithm
/// - helper for computing rank of the row partner for P2P communications
/// - helper for indexing panel tile
///
/// In particular, about the latter one, the panel has an extra workspace used when a row is involved
/// in the computation as partner of the row above it (e.g. P2P communication).
/// This is needed because applying a block of HH affects (excluding some edge-cases) two rows of tiles,
/// that might reside on different ranks.
/// wsIndexHH() job is to return the right index to use in the panel, according to the row this rank
/// is working on, and its role as a main or partner row.
struct DistIndexing {
  DistIndexing(const TileAccessHelper& helper, const matrix::Distribution& dist_hh, const SizeType b,
               const GlobalTileIndex& ij, const SizeType& ij_b_row)
      : dist_hh(dist_hh), b(b), mb(dist_hh.blockSize().rows()), helper(helper), ij(ij),
        ij_b_row(ij_b_row) {
    rank = dist_hh.rankIndex();
    rankHH = dist_hh.rankGlobalTile(ij);
    n_ws_per_block = to_SizeType(static_cast<size_t>(std::ceil(mb / b / 2.0f)) + 1);
  }

  comm::IndexT_MPI rankRowPartner() const {
    return (rankHH.row() + 1) % dist_hh.commGridSize().rows();
  }

  bool isInvolved() const {
    const bool isSameRow = rank.row() == rankHH.row();
    const bool isPartnerRow = rank.row() == rankRowPartner();
    return isSameRow ||
           (dist_hh.commGridSize().rows() > 1 && isPartnerRow && helper.affectsMultipleTiles());
  }

  LocalTileIndex wsIndexHH() const {
    const SizeType row = [&]() -> SizeType {
      if (rank.row() == rankHH.row()) {
        // Note: index starts at 1 (0 is the extra workspace), moreover max half blocks will run in parallel
        const SizeType intra_idx = 1 + (ij_b_row % (mb / b)) / 2;
        DLAF_ASSERT_HEAVY(intra_idx < n_ws_per_block, intra_idx, n_ws_per_block);
        return dist_hh.localTileFromGlobalTile<Coord::Row>(ij.row()) * n_ws_per_block + intra_idx;
      }
      else {
        DLAF_ASSERT_HEAVY(helper.affectsMultipleTiles() && (rank.row() == rankRowPartner()),
                          helper.affectsMultipleTiles(), rank.row(), rankRowPartner());
        return dist_hh.localNrTiles().isEmpty()
                   ? 0
                   : dist_hh.localTileFromGlobalTile<Coord::Row>(ij.row() + 1) * n_ws_per_block;
      }
    }();
    return {row, 0};
  }

protected:
  matrix::Distribution dist_hh;
  SizeType b;
  SizeType mb;
  SizeType n_ws_per_block;

  TileAccessHelper helper;

  comm::Index2D rank;
  comm::Index2D rankHH;

  GlobalTileIndex ij;
  SizeType ij_b_row;
};

template <Backend B, Device D, class T>
struct HHManager;

template <class T>
struct HHManager<Backend::MC, Device::CPU, T> {
  static constexpr Backend B = Backend::MC;
  static constexpr Device D = Device::CPU;

  HHManager(const SizeType b, const std::size_t, matrix::Distribution, matrix::Distribution) : b(b) {}

  template <class SenderHH>
  std::tuple<pika::shared_future<matrix::Tile<const T, D>>, pika::shared_future<matrix::Tile<const T, D>>>
  computeVW(const LocalTileIndex ij, const TileAccessHelper& helper, SenderHH&& tile_hh,
            matrix::Panel<Coord::Col, T, D>& mat_v, matrix::Panel<Coord::Col, T, D>& mat_t,
            matrix::Panel<Coord::Col, T, D>& mat_w) {
    namespace ex = pika::execution::experimental;

    auto tup =
        dlaf::internal::whenAllLift(b, std::forward<SenderHH>(tile_hh),
                                    splitTile(mat_v(ij), helper.specHH()),
                                    splitTile(mat_t(ij), helper.specT()),
                                    splitTile(mat_w(ij), helper.specHH())) |
        dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), bt_tridiag::computeVW<T>) |
        ex::make_future();

    return pika::split_future(std::move(tup));
  }

protected:
  const SizeType b;
};

#ifdef DLAF_WITH_GPU
template <class T>
struct HHManager<Backend::GPU, Device::GPU, T> {
  static constexpr Backend B = Backend::GPU;
  static constexpr Device D = Device::GPU;

  HHManager(const SizeType b, const std::size_t n_workspaces, matrix::Distribution dist_t,
            matrix::Distribution dist_w)
      : b(b), t_panels_h(n_workspaces, dist_t), w_panels_h(n_workspaces, dist_w) {}

  template <class SenderHH>
  std::tuple<pika::shared_future<matrix::Tile<const T, D>>, pika::shared_future<matrix::Tile<const T, D>>>
  computeVW(const LocalTileIndex ij, const TileAccessHelper& helper, SenderHH&& tile_hh,
            matrix::Panel<Coord::Col, T, D>& mat_v, matrix::Panel<Coord::Col, T, D>& mat_t,
            matrix::Panel<Coord::Col, T, D>& mat_w) {
    namespace ex = pika::execution::experimental;

    auto& mat_v_h = w_panels_h.nextResource();
    auto& mat_t_h = t_panels_h.nextResource();

    const LocalTileIndex ij_t = ij;
    const matrix::SubTileSpec t_spec = helper.specT();

    auto tup = dlaf::internal::whenAllLift(b, std::forward<SenderHH>(tile_hh),
                                           splitTile(mat_v_h(ij), helper.specHH()),
                                           splitTile(mat_t_h(ij_t), t_spec)) |
               dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), computeVT<T>) |
               ex::make_future();

    auto [tile_v_h, tile_t_h] = pika::split_future(std::move(tup));

    auto copyVTandComputeW =
        [](cublasHandle_t handle, const matrix::Tile<const T, Device::CPU>& tile_v_h,
           const matrix::Tile<const T, Device::CPU>& tile_t_h, matrix::Tile<T, Device::GPU>& tile_v,
           matrix::Tile<T, Device::GPU>& tile_t, matrix::Tile<T, Device::GPU>& tile_w) {
          whip::stream_t stream;
          DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));

          matrix::internal::copy(tile_v_h, tile_v, stream);
          matrix::internal::copy(tile_t_h, tile_t, stream);

          // W = V . T
          using namespace blas;
          tile::internal::trmm3(handle, Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1),
                                tile_t, tile_v, tile_w);

          return std::make_tuple(matrix::Tile<const T, D>(std::move(tile_v)),
                                 matrix::Tile<const T, D>(std::move(tile_w)));
        };

    auto tup2 = ex::when_all(ex::keep_future(std::move(tile_v_h)), ex::keep_future(std::move(tile_t_h)),
                             splitTile(mat_v(ij), helper.specHH()), splitTile(mat_t(ij_t), t_spec),
                             splitTile(mat_w(ij), helper.specHH())) |
                dlaf::internal::transform<
                    dlaf::internal::TransformDispatchType::Blas>(dlaf::internal::Policy<Backend::GPU>(),
                                                                 copyVTandComputeW) |
                ex::make_future();

    return pika::split_future(std::move(tup2));
  }

protected:
  const SizeType b;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> t_panels_h;
  common::RoundRobin<matrix::Panel<Coord::Col, T, Device::CPU>> w_panels_h;
};
#endif
}

template <Backend B, Device D, class T>
void BackTransformationT2B<B, D, T>::call(const SizeType band_size, Matrix<T, D>& mat_e,
                                          Matrix<const T, Device::CPU>& mat_hh) {
  using pika::execution::thread_priority;
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

      auto [tile_v, tile_w] =
          helperBackend.computeVW(ij, helper,
                                  ex::keep_future(splitTile(mat_hh.read(ij), helper.specHHCompact())),
                                  mat_v, mat_t, mat_w);

      for (SizeType j_e = 0; j_e < mat_e.nrTiles().cols(); ++j_e) {
        const LocalTileIndex idx_e(ij.row(), j_e);
        const auto sz_e = mat_e.tileSize({ij.row(), j_e});
        auto tile_e = mat_e.read(idx_e);

        if (not helper.affectsMultipleTiles()) {
          ex::start_detached(
              ex::when_all(ex::keep_future(splitTile(tile_v, helper.topPart().specHH())),
                           ex::keep_future(splitTile(tile_w, helper.topPart().specHH())),
                           mat_w2.readwrite_sender(LocalTileIndex(0, j_e)),
                           splitTile(mat_e(idx_e), helper.topPart().specEV(sz_e.cols()))) |
              dlaf::internal::transform<
                  dlaf::internal::TransformDispatchType::Blas>(dlaf::internal::Policy<B>(
                                                                   thread_priority::normal),
                                                               ApplyHHToSingleTileRow<B, T>{}));
        }
        else {
          auto tile_vs = splitTile(tile_v, {helper.topPart().specHH(), helper.bottomPart().specHH()});
          auto tile_ws = splitTile(tile_w, {helper.topPart().specHH(), helper.bottomPart().specHH()});
          ex::start_detached(
              ex::when_all(ex::keep_future(tile_vs[0]), ex::keep_future(tile_vs[1]),
                           ex::keep_future(tile_ws[0]), ex::keep_future(tile_ws[1]),
                           mat_w2.readwrite_sender(LocalTileIndex(0, j_e)),
                           splitTile(mat_e(idx_e), helper.topPart().specEV(sz_e.cols())),
                           splitTile(mat_e(LocalTileIndex{ij.row() + 1, j_e}),
                                     helper.bottomPart().specEV(sz_e.cols()))) |
              dlaf::internal::transform<
                  dlaf::internal::TransformDispatchType::Blas>(dlaf::internal::Policy<B>(
                                                                   thread_priority::normal),
                                                               ApplyHHToDoubleTileRow<B, T>{}));
        }
      }

      mat_t.reset();
      mat_v.reset();
      mat_w.reset();
      mat_w2.reset();
    }
  }
}

template <Backend B, Device D, class T>
void BackTransformationT2B<B, D, T>::call(comm::CommunicatorGrid grid, const SizeType band_size,
                                          Matrix<T, D>& mat_e, Matrix<const T, Device::CPU>& mat_hh) {
  using pika::execution::thread_priority;
  namespace ex = pika::execution::experimental;

  using common::iterate_range2d;
  using common::RoundRobin;
  using matrix::Panel;
  using namespace bt_tridiag;

  if (mat_hh.size().isEmpty() || mat_e.size().isEmpty())
    return;

  // Note: if no householder reflectors are going to be applied (in case of trivial matrix)
  if (nrSweeps<T>(mat_hh.size().rows()) == 0)
    return;

  const SizeType b = band_size;
  const SizeType mb = mat_hh.blockSize().rows();

  const auto& dist_hh = mat_hh.distribution();
  const auto& dist_e = mat_e.distribution();

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

  const SizeType nlocal_ws =
      std::max<SizeType>(1, dist_hh.localNrTiles().rows() *
                                to_SizeType(static_cast<SizeType>(std::ceil(mb / b / 2.0f)) + 1));
  const matrix::Distribution dist_ws_hh({nlocal_ws * b, b}, {b, b});
  const matrix::Distribution dist_ws_v({nlocal_ws * w_tile_sz.rows(), w_tile_sz.cols()}, w_tile_sz);
  const matrix::Distribution dist_ws_w2({nlocal_ws * b, mat_e.size().cols()},
                                        {b, mat_e.blockSize().cols()});

  constexpr std::size_t n_workspaces = 2;

  RoundRobin<Panel<Coord::Col, T, D>> t_panels(n_workspaces, dist_ws_hh);
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> hh_panels(n_workspaces, dist_ws_hh);

  RoundRobin<Panel<Coord::Col, T, D>> v_panels(n_workspaces, dist_ws_v);
  RoundRobin<Panel<Coord::Col, T, D>> w_panels(n_workspaces, dist_ws_v);

  RoundRobin<Panel<Coord::Row, T, D>> w2_panels(n_workspaces, dist_ws_w2);
  RoundRobin<Panel<Coord::Row, T, D>> w2tmp_panels(n_workspaces, dist_ws_w2);

  HHManager<B, D, T> helperBackend(b, n_workspaces, dist_ws_hh, dist_ws_v);

  // Note: This distributed algorithm encompass two communication categories:
  // 1. exchange of HH: broadcast + send p2p
  // 2. reduction for computing W2: all reduce p2p
  // P2P communication can happen out of order since they can be matched via tags, but this is not
  // possible for collective operations such as the broadcast.
  //
  // For this reason, communications of the phase 1 will be ordered with a pipeline. Instead, for the
  // second part, with the aim to not over constrain execution of the update, no order will be
  // enforced by relying solely on tags.
  common::Pipeline<comm::Communicator> mpi_chain_row(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_chain_col(grid.colCommunicator().clone());
  const auto mpi_col_comm = ex::just(grid.colCommunicator().clone());

  const SizeType idx_last_sweep_b = (nrSweeps<T>(mat_hh.size().cols()) - 1) / b;
  const SizeType maxsteps_b = nrStepsForSweep(0, mat_hh.size().rows(), b);

  // Note: Next two nested `for`s describe a special order loop over the matrix, which allow to
  // better schedule communications considering the structure of the algorithm.
  //
  // Each element depends on:
  // - top
  // - bottom-right
  // - right
  //
  // This basic rule for dependencies can be described collectively as a mechanism where elements are
  // "unlocked" in different epochs, which forms a pattern like if the matrix get scanned not
  // perpendicularly to their main axis, but instead it gets scanned by a slightly skewed line that goes
  // from top right to bottom left.
  //
  //  5 x x x x
  //  6 4 x x x
  //  7 5 3 x x
  //  8 6 4 2 x
  //  9 7 5 3 1
  //
  // Elements of the same epoch are somehow "independent" and so they can potentially run in parallel,
  // given that previous epoch has been completed. Since scheduling happens sequentially, elements
  // of the same epoch will be ordered starting from top-most one, resulting in
  //
  //  7  x x x x
  // 10  5 x x x
  // 12  8 3 x x
  // 14 11 6 2 x
  // 15 13 9 4 1
  for (SizeType k = idx_last_sweep_b; k > -maxsteps_b; --k) {
    auto& mat_t = t_panels.nextResource();
    auto& panel_hh = hh_panels.nextResource();
    auto& mat_v = v_panels.nextResource();
    auto& mat_w = w_panels.nextResource();
    auto& mat_w2 = w2_panels.nextResource();
    auto& mat_w2tmp = w2tmp_panels.nextResource();

    for (SizeType i_b = std::abs<SizeType>(k), j_b = std::max<SizeType>(0, k);
         i_b < j_b + nrStepsForSweep(j_b * b, mat_hh.size().cols(), b); i_b += 2, ++j_b) {
      const SizeType step_b = i_b - j_b;
      const GlobalElementIndex ij_el(i_b * b, j_b * b);
      const GlobalTileIndex ij_g(dist_hh.globalTileIndex(ij_el));

      const comm::Index2D rank = dist_hh.rankIndex();
      const comm::Index2D rankHH = dist_hh.rankGlobalTile(ij_g);

      // Note:  reflector with size = 1 must be ignored, except for the last step of the last sweep
      //        with complex type
      const SizeType nrefls = [&]() {
        const bool allowSize1 = isComplex_v<T> && j_b == idx_last_sweep_b &&
                                step_b == nrStepsForSweep(j_b * b, mat_hh.size().cols(), b) - 1;
        const GlobalElementSize delta(dist_hh.size().rows() - ij_el.row() - 1,
                                      std::min(b, dist_hh.size().cols() - ij_el.col()));
        return std::min(b, std::min(delta.rows() - (allowSize1 ? 0 : 1), delta.cols()));
      }();

      const TileAccessHelper helper(b, nrefls, dist_hh, ij_el);
      const DistIndexing indexing_helper(helper, dist_hh, b, ij_g, i_b);

      if (!indexing_helper.isInvolved())
        continue;

      if (nrefls < b) {
        mat_t.setWidth(nrefls);
        mat_v.setWidth(nrefls);
        mat_w.setWidth(nrefls);
        mat_w2.setHeight(nrefls);
        mat_w2tmp.setHeight(nrefls);
      }

      // Note:
      // From HH it is possible to extract V that is needed for computing W and W2, both required
      // for updating E.

      // Send HH to all involved ranks: broadcast on row + send p2p on col
      const LocalTileIndex ij_hh_panel = indexing_helper.wsIndexHH();

      // Broadcast on ROW
      if (grid.size().cols() > 1 && rank.row() == rankHH.row()) {
        if (rank.col() == rankHH.col()) {
          ex::start_detached(
              comm::scheduleSendBcast(mpi_chain_row(),
                                      ex::keep_future(
                                          splitTile(mat_hh.read(ij_g), helper.specHHCompact()))));
        }
        else {
          ex::start_detached(comm::scheduleRecvBcast(mpi_chain_row(), rankHH.col(),
                                                     ex::make_unique_any_sender(
                                                         splitTile(panel_hh(ij_hh_panel),
                                                                   helper.specHHCompact(true)))));
        }
      }

      // Send P2P on col
      if (helper.affectsMultipleTiles() && grid.size().rows() > 1) {
        const comm::IndexT_MPI rank_src = rankHH.row();
        const comm::IndexT_MPI rank_dst = indexing_helper.rankRowPartner();

        if (rank.row() == rank_src) {
          auto tile_hh = rank.col() == rankHH.col()
                             ? splitTile(mat_hh.read(ij_g), helper.specHHCompact())
                             : splitTile(panel_hh.read(ij_hh_panel), helper.specHHCompact(true));
          ex::start_detached(comm::scheduleSend(mpi_chain_col(), rank_dst, 0, ex::keep_future(tile_hh)));
        }
        else if (rank.row() == rank_dst) {
          ex::start_detached(
              comm::scheduleRecv(mpi_chain_col(), rank_src, 0, panel_hh.readwrite_sender(ij_hh_panel)));
        }
      }

      // COMPUTE V and W from HH and T
      auto tile_hh = (rankHH == rank)
                         ? splitTile(mat_hh.read(ij_g), helper.specHHCompact())
                         : splitTile(panel_hh.read(ij_hh_panel), helper.specHHCompact(true));
      auto [tile_v, tile_w] =
          helperBackend.computeVW(indexing_helper.wsIndexHH(), helper,
                                  ex::keep_future(std::move(tile_hh)), mat_v, mat_t, mat_w);

      // UPDATE E
      const SizeType ncols_local = dist_e.localNrTiles().cols();
      for (SizeType j_e = 0; j_e < ncols_local; ++j_e) {
        const SizeType j_e_g = dist_e.template globalTileFromLocalTile<Coord::Col>(j_e);
        const LocalTileIndex idx_w2(indexing_helper.wsIndexHH().row(), j_e);

        // SINGLE RANK UPDATE
        if (!helper.affectsMultipleTiles() || grid.size().rows() == 1) {
          const SizeType i_e_g = ij_g.row();
          const SizeType i_e = dist_e.template localTileFromGlobalTile<Coord::Row>(ij_g.row());
          const LocalTileIndex idx_e(i_e, j_e);
          const auto nb = mat_e.tileSize({i_e_g, j_e_g}).cols();

          // TWO ROWs (same RANK)
          if (helper.affectsMultipleTiles()) {
            const auto partsSpecs = {helper.topPart().specHH(), helper.bottomPart().specHH()};
            const auto tile_vs = splitTile(tile_v, partsSpecs);
            const auto tile_ws = splitTile(tile_w, partsSpecs);

            ex::start_detached(
                ex::when_all(ex::keep_future(tile_vs[0]), ex::keep_future(tile_vs[1]),
                             ex::keep_future(tile_ws[0]), ex::keep_future(tile_ws[1]),
                             splitTile(mat_w2(idx_w2), helper.specW2(nb)),
                             splitTile(mat_e(idx_e), helper.topPart().specEV(nb)),
                             splitTile(mat_e(LocalTileIndex{idx_e.row() + 1, j_e}),
                                       helper.bottomPart().specEV(nb))) |
                dlaf::internal::transform<
                    dlaf::internal::TransformDispatchType::Blas>(dlaf::internal::Policy<B>(
                                                                     thread_priority::normal),
                                                                 ApplyHHToDoubleTileRow<B, T>{}));
          }
          // SINGLE ROW (edge-case)
          else {
            ex::start_detached(
                ex::when_all(ex::keep_future(splitTile(tile_v, helper.topPart().specHH())),
                             ex::keep_future(splitTile(tile_w, helper.topPart().specHH())),
                             splitTile(mat_w2(idx_w2), helper.specW2(nb)),
                             splitTile(mat_e(idx_e), helper.topPart().specEV(nb))) |
                dlaf::internal::transform<
                    dlaf::internal::TransformDispatchType::Blas>(dlaf::internal::Policy<B>(
                                                                     thread_priority::normal),
                                                                 ApplyHHToSingleTileRow<B, T>{}));
          }
        }
        // TWO RANKs UPDATE (MAIN + PARTNER) on TWO ROWS
        else {
          const bool isTopRank = rank.row() == rankHH.row();
          const comm::IndexT_MPI rankPartner =
              isTopRank ? indexing_helper.rankRowPartner() : rankHH.row();

          const comm::IndexT_MPI tag = to_int(j_e + i_b * ncols_local);

          const auto part = isTopRank ? helper.topPart() : helper.bottomPart();
          const SizeType i_e_g = isTopRank ? ij_g.row() : (ij_g.row() + 1);
          const SizeType i_e = dist_e.template localTileFromGlobalTile<Coord::Row>(i_e_g);
          const LocalTileIndex idx_e(i_e, j_e);

          const auto nb = mat_e.tileSize({i_e_g, j_e_g}).cols();

          // W2 = V* . E
          ex::start_detached(
              dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, T(1),
                                          ex::keep_future(splitTile(tile_v, part.specHH())),
                                          splitTile(mat_e(idx_e), part.specEV(nb)), T(0),
                                          splitTile(mat_w2tmp(idx_w2), helper.specW2(nb))) |
              dlaf::tile::gemm(dlaf::internal::Policy<B>(thread_priority::normal)));

          // Compute final W2 by adding the contribution from the partner rank
          ex::start_detached(comm::scheduleAllSumP2P<B>(mpi_col_comm, rankPartner, tag,
                                                        ex::keep_future(splitTile(mat_w2tmp.read(idx_w2),
                                                                                  helper.specW2(nb))),
                                                        splitTile(mat_w2(idx_w2), helper.specW2(nb))));

          // E -= W . W2
          ex::start_detached(
              dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, T(-1),
                                          ex::keep_future(splitTile(tile_w, part.specHH())),
                                          ex::keep_future(
                                              splitTile(mat_w2.read(idx_w2), helper.specW2(nb))),
                                          T(1), splitTile(mat_e(idx_e), part.specEV(nb))) |
              dlaf::tile::gemm(dlaf::internal::Policy<B>(thread_priority::normal)));
        }
      }

      mat_t.reset();
      panel_hh.reset();
      mat_v.reset();
      mat_w.reset();
      mat_w2tmp.reset();
      mat_w2.reset();
    }
  }
}
}
