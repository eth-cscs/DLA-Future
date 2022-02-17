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
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
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
  static void call(Matrix<T, Device::CPU>& mat_e, Matrix<const T, Device::CPU>& mat_hh,
                   const SizeType band_size);
};

template <class T>
pika::shared_future<matrix::Tile<const T, Device::CPU>> setupVWellFormed(
    const SizeType b, pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_i,
    pika::future<matrix::Tile<T, Device::CPU>> tile_v) {
  auto unzipV_func = [b](const auto& tile_i, auto tile_v) {
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

    if (tile_v.size().rows() > b)
      laset(blas::Uplo::Lower, tile_v.size().rows() - b, k - 1, T(0), T(0), tile_v.ptr({b, 0}),
            tile_v.ld());

    return matrix::Tile<const T, Device::CPU>(std::move(tile_v));
  };
  return pika::dataflow(pika::unwrapping(unzipV_func), std::move(tile_i), std::move(tile_v));
}

template <class T>
pika::shared_future<matrix::Tile<const T, Device::CPU>> computeTFactor(
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_taus,
    pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
    pika::future<matrix::Tile<T, Device::CPU>> mat_t) {
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
  return pika::dataflow(pika::unwrapping(tfactor_task), std::move(tile_taus), std::move(tile_v),
                        std::move(mat_t));
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

class Helper {
  struct part_t {
    SizeType rows() const noexcept {
      return nrows_;
    }

    TileElementIndex origin() const noexcept {
      return {origin_, 0};
    }

    TileElementIndex origin_full() const noexcept {
      return {offset_, 0};
    }

    SizeType nrows_ = 0;
    SizeType origin_;
    SizeType offset_ = 0;
  };

public:
  Helper(const SizeType b, matrix::Distribution dist, const GlobalElementIndex offset)
      : b_(b), dist_(dist), offset_(offset) {
    const TileElementIndex sub_offset = dist_.tileElementIndex(offset_);

    const SizeType tile_row = dist_.globalTileFromGlobalElement<Coord::Row>(offset_.row());
    const bool isLastRow = tile_row == dist.nrTiles().rows() - 1;

    const SizeType max_size = dist_.size().rows() - offset_.row() - 1;
    if (isLastRow && max_size < 2 * b_ - 1) {
      mode_ = Mode::SINGLE_PART;
      parts_[0] = part_t{max_size, sub_offset.row() + 1};
    }
    else {
      const SizeType mb = dist.blockSize().rows();
      if (mb - sub_offset.row() - 1 >= 2 * b_ - 1) {
        mode_ = Mode::SINGLE_FULL;
        parts_[0] = part_t{2 * b_ - 1, sub_offset.row() + 1};
      }
      else {
        mode_ = Mode::DOUBLE_FULL;
        const SizeType mb2 = dist_.size().rows() - offset_.row() - b_;
        parts_[0] = part_t{b_ - 1, sub_offset.row() + 1};
        parts_[1] = part_t{std::min(b_, mb2), 0, b_ - 1};
      }
    }

    input_spec_ = {sub_offset,
                   {std::min(b_, dist_.size().rows() - offset_.row()),
                    std::min(b_, dist_.size().cols() - offset_.col())}};
  };

  bool isMultiPart() const noexcept {
    return mode_ == Mode::DOUBLE_FULL;
  }

  matrix::SubTileSpec inputSpec() const noexcept {
    return input_spec_;
  }

  const part_t& operator[](const SizeType index) const noexcept {
    return parts_[to_sizet(index)];
  }

private:
  SizeType b_;
  matrix::Distribution dist_;
  GlobalElementIndex offset_;

  matrix::SubTileSpec input_spec_ = {{}, {0, 0}};

  enum class Mode { SINGLE_FULL, DOUBLE_FULL, SINGLE_PART } mode_;
  std::array<part_t, 2> parts_;
};

template <class T>
void BackTransformationT2B<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_e,
                                                              Matrix<const T, Device::CPU>& mat_hh,
                                                              const SizeType band_size) {
  static constexpr Backend backend = Backend::MC;
  using pika::threads::thread_priority;
  using pika::execution::experimental::keep_future;

  using matrix::Panel;
  using common::RoundRobin;
  using common::iterate_range2d;

  if (mat_hh.size().isEmpty() || mat_e.size().isEmpty())
    return;

  const SizeType b = band_size;
  const SizeType nsweeps = nrSweeps<T>(mat_hh.size().cols());

  const auto& dist_i = mat_hh.distribution();

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

  // TODO optimize allocated space for w
  const SizeType dist_w_rows = [&]() { return mat_e.nrTiles().rows() * w_tile_sz.rows(); }();

  const matrix::Distribution dist_w({dist_w_rows, b}, w_tile_sz);
  const matrix::Distribution dist_t({mat_hh.size().rows(), b}, {b, b});
  const matrix::Distribution dist_w2({b, mat_e.size().cols()}, {b, mat_e.blockSize().cols()});

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> t_panels(n_workspaces, dist_t);
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> w_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Row, T, Device::CPU>> w2_panels(n_workspaces, dist_w2);

  // Note: sweep are on diagonals, steps are on verticals
  const SizeType j_last_sweep = (nsweeps - 1) / b;
  for (SizeType j = j_last_sweep; j >= 0; --j) {
    auto& mat_w = w_panels.nextResource();
    auto& mat_t = t_panels.nextResource();
    auto& mat_w2 = w2_panels.nextResource();

    // Note: apply the entire column (steps)
    const SizeType steps = nrStepsForSweep(j * b, mat_hh.size().cols(), b);
    for (SizeType step = 0; step < steps; ++step) {
      const SizeType i = j + step;

      const GlobalElementIndex ij_e(i * b, j * b);
      const LocalTileIndex ij(dist_i.localTileIndex(dist_i.globalTileIndex(ij_e)));

      const Helper helper(b, mat_hh.distribution(), ij_e);
      const SizeType v_rows = helper[0].rows() + helper[1].rows();

      // Note: In general a tile contains a reflector per column, but in case there aren't enough
      // rows for storing a reflector whose length is greater than 1, then the number is reduced
      // accordingly. An exception is represented by the last bottom right tile, where the refelctors
      // are all the remaining ones (i.e. there may be a length 1 reflector in complex cases)
      const SizeType nrefls = (j != j_last_sweep) ? std::min(v_rows - 1, b) : (nsweeps - j * b);

      if (nrefls < b) {
        mat_w.setWidth(nrefls);
        mat_t.setWidth(nrefls);
        mat_w2.setHeight(nrefls);
      }

      const matrix::SubTileSpec v_spec{helper[0].origin_full(), {v_rows, nrefls}};
      const matrix::SubTileSpec v_up{helper[0].origin_full(), {helper[0].rows(), nrefls}};
      const matrix::SubTileSpec v_dn{helper[1].origin_full(), {helper[1].rows(), nrefls}};

      // TODO setRange? it would mean setting the range to a specific tile for each step, and resetting at the end

      // Note:
      // Let's use W as a temporary storage for the well formed version of V, which is needed
      // for computing W2 = V . E, and it is also useful for computing T
      auto tile_w_full = mat_w(ij);
      auto tile_w_rw = splitTile(tile_w_full, v_spec);

      auto tile_i = splitTile(mat_hh.read(ij), helper.inputSpec());
      auto tile_v = setupVWellFormed(b, tile_i, std::move(tile_w_rw));

      // W2 = V* . E
      // Note:
      // Since the well-formed V is stored in W, we have to use it before W will get overwritten.
      // For this reason W2 is computed before W.
      for (SizeType j_e = 0; j_e < mat_e.nrTiles().cols(); ++j_e) {
        const auto sz_e = mat_e.tileSize({ij.row(), j_e});
        auto tile_e = mat_e.read(LocalTileIndex(ij.row(), j_e));

        computeW2<backend>(thread_priority::normal, keep_future(splitTile(tile_v, v_up)),
                           keep_future(
                               splitTile(tile_e, {helper[0].origin(), {helper[0].rows(), sz_e.cols()}})),
                           T(0), mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));

        if (helper.isMultiPart()) {
          auto tile_e = mat_e.read(LocalTileIndex(ij.row() + 1, j_e));
          computeW2<backend>(thread_priority::normal, keep_future(splitTile(tile_v, v_dn)),
                             keep_future(splitTile(tile_e, {helper[1].origin(),
                                                            {helper[1].rows(), sz_e.cols()}})),
                             T(1), mat_w2.readwrite_sender(LocalTileIndex(0, j_e)));
        }
      }

      // Note:
      // And we use it also for computing the T factor
      auto tile_t_full = mat_t(LocalTileIndex(ij.row(), 0));
      auto tile_t = computeTFactor(tile_i, tile_v, splitTile(tile_t_full, {{0, 0}, {nrefls, nrefls}}));

      // W = V . T
      // Note:
      // At this point W can be overwritten, but this will happen just after W2 and T computations
      // finished. T was already a dependency, but W2 wasn't, meaning it won't run in parallel,
      // but it could.
      tile_w_rw = splitTile(tile_w_full, v_spec);
      computeW<backend>(thread_priority::normal, std::move(tile_w_rw), keep_future(tile_t));
      auto tile_w = mat_w.read(ij);

      // E -= W . W2
      for (SizeType j_e = 0; j_e < mat_e.nrTiles().cols(); ++j_e) {
        const LocalTileIndex idx_e(ij.row(), j_e);
        const auto sz_e = mat_e.tileSize({ij.row(), j_e});

        auto tile_e = mat_e(idx_e);
        const auto& tile_w2 = mat_w2.read_sender(idx_e);

        updateE<backend>(pika::threads::thread_priority::normal, keep_future(splitTile(tile_w, v_up)),
                         tile_w2,
                         splitTile(tile_e, {helper[0].origin(), {helper[0].rows(), sz_e.cols()}}));

        if (helper.isMultiPart()) {
          auto tile_e = mat_e.readwrite_sender(LocalTileIndex{ij.row() + 1, j_e});
          updateE<backend>(pika::threads::thread_priority::normal, keep_future(splitTile(tile_w, v_dn)),
                           tile_w2,
                           keep_future(splitTile(tile_e, {helper[1].origin(),
                                                          {helper[1].rows(), sz_e.cols()}})));
        }
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
