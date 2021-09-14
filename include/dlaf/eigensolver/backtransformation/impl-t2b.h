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

#include <hpx/include/util.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/eigensolver/backtransformation/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix/print_numpy.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
struct BackTransformationT2B<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_e, Matrix<const T, Device::CPU>& mat_i);
private:
  static constexpr bool is_complex = std::is_same<T, ComplexType<T>>::value;

  static SizeType nrSweeps(const SizeType m) {
    return std::max<SizeType>(0, is_complex ? m - 1 : m - 2);
  }

  static SizeType nrStepsPerSweep(SizeType sweep, SizeType m, SizeType mb) {
    return std::max<SizeType>(0, sweep == m - 2 ? m - 1 : dlaf::util::ceilDiv(m - sweep - 2, mb));
  }
};

template <class T>
auto setupVWellFormed(SizeType k, hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v_compact,
                      hpx::future<matrix::Tile<T, Device::CPU>> tile_v) -> decltype(tile_v_compact) {
  auto unzipV_func = [k](const auto& tile_v_compact, auto tile_v) {
    constexpr auto General = lapack::MatrixType::General;
    constexpr auto Upper = lapack::MatrixType::Upper;
    constexpr auto Lower = lapack::MatrixType::Lower;

    std::cout << "Vnrefls = " << k << "\n";

    std::cout << "Vc = ";
    print(format::numpy{}, tile_v_compact);

    std::cout << "Vp = ";
    print(format::numpy{}, tile_v);

    // TODO this requires W complete (even the bottom one)
    //lacpy(General, b - 1, b,
    //    tile_v_compact.ptr({1, 0}), tile_v_compact.ld(),
    //    tile_v.ptr({1, 0}), tile_v.ld() + 1);

    for (SizeType j = 0; j < k; ++j) {
      const SizeType size = std::min<SizeType>(
          tile_v.size().rows() - (1 + j),
          tile_v_compact.size().rows() - 1);
      lacpy(General, size, 1,
          tile_v_compact.ptr({1, j}), tile_v_compact.ld(),
          tile_v.ptr({1 + j, j}), tile_v.ld());
    }

    // TODO is it needed because W = V . T? or is it enough just setting ones?
    laset(Upper, tile_v.size().rows(), k, T(0), T(1), tile_v.ptr({0, 0}), tile_v.ld());

    const SizeType mb = tile_v_compact.size().cols();
    if (tile_v.size().rows() > mb) {
      laset(Lower, tile_v.size().rows() - mb, k - 1, T(0), T(0), tile_v.ptr({mb, 0}), tile_v.ld());
    }

    std::cout << "V = ";
    print(format::numpy{}, tile_v);

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

    //std::cout << "V = ";
    //print(format::numpy{}, tile_v);

    //std::cout << "taus = ";
    //print(format::numpy{}, tile_taus);

    const auto k = tile_v.size().cols();
    const auto n = tile_v.size().rows();
    //DLAF_ASSERT_HEAVY((tile_t.size() == TileElementSize(k, k)), tile_t.size());

    std::vector<T> taus;
    taus.resize(to_sizet(tile_v.size().cols()));
    for (SizeType i = 0; i < to_SizeType(taus.size()); ++i)
      taus[to_sizet(i)] = *tile_taus.ptr({0, i});

    larft(Direction::Forward, StoreV::Columnwise, n, k, tile_v.ptr(), tile_v.ld(), taus.data(),
          tile_t.ptr(), tile_t.ld());

    //std::cout << "T = ";
    //print(format::numpy{}, tile_t);
    return matrix::Tile<const T, Device::CPU>(std::move(tile_t));
  };
  return hpx::dataflow(hpx::unwrapping(tfactor_task), std::move(tile_taus), std::move(tile_v),
                       std::move(mat_t));
}

template <class T>
auto computeW(hpx::future<matrix::Tile<T, Device::CPU>> tile_v,
              hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t) {
  using namespace blas;

  //hpx::dataflow(hpx::unwrapping([](auto tile_w, const auto& tile_t) {
  //      std::cout << "\tV = ";
  //      print(format::numpy{}, tile_w);
  //      std::cout << "\tT = ";
  //      print(format::numpy{}, tile_t);
  //      tile::trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), tile_t,
  //          tile_w); std::cout << "W = "; print(format::numpy{}, tile_w);
  //      }),
  //    std::move(tile_v), std::move(tile_t));

  hpx::dataflow(matrix::unwrapExtendTiles(tile::trmm_o), Side::Right, Uplo::Upper, Op::NoTrans,
                Diag::NonUnit, T(1), std::move(tile_t), std::move(tile_v));
}

template <class T>
auto computeW2(hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_e, T beta,
               hpx::future<matrix::Tile<T, Device::CPU>> tile_w2) {
  using namespace blas;

  // hpx::dataflow(hpx::unwrapping(
  //                  [](const auto& tile_v, const auto& tile_e, const auto beta, auto tile_w2) {
  //                    std::cout << "\t_V = ";
  //                    print(format::numpy{}, tile_v);
  //                    std::cout << "\t_E = ";
  //                    print(format::numpy{}, tile_e);
  //                    std::cout << "\tbeta = " << beta << "\n";
  //                    std::cout << "\t_W2pre = ";
  //                    print(format::numpy{}, tile_w2);
  //                    tile::gemm(Op::ConjTrans, Op::NoTrans, T(1), tile_v, tile_e, beta, tile_w2);
  //                    std::cout << "_W2 = ";
  //                    print(format::numpy{}, tile_w2);
  //                  }),
  //              std::move(tile_v), std::move(tile_e), beta, std::move(tile_w2));
  hpx::dataflow(matrix::unwrapExtendTiles(tile::gemm_o), Op::ConjTrans, Op::NoTrans, T(1),
                std::move(tile_v), std::move(tile_e), beta, std::move(tile_w2));
}

template <class T>
auto updateE(hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_w,
             hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_w2,
             hpx::future<matrix::Tile<T, Device::CPU>> tile_e) {
  using blas::Op;

  //auto func = hpx::unwrapping([](auto&& tile_w, auto&& tile_w2, auto tile_e) {
  //  std::cout << "\tW = ";
  //  print(format::numpy{}, tile_w);
  //  std::cout << "\tW2 = ";
  //  print(format::numpy{}, tile_w2);
  //  std::cout << "\tEpre = ";
  //  print(format::numpy{}, tile_e);
  //  tile::gemm(Op::NoTrans, Op::NoTrans, T(-1), tile_w, tile_w2, T(1), tile_e);
  //  std::cout << "E = ";
  //  print(format::numpy{}, tile_e);
  //});
  //hpx::dataflow(func, std::move(tile_w), std::move(tile_w2), std::move(tile_e));

  hpx::dataflow(hpx::unwrapping(tile::gemm_o), Op::NoTrans, Op::NoTrans, T(-1), std::move(tile_w),
                std::move(tile_w2), T(1), std::move(tile_e));
}

template <class T>
void BackTransformationT2B<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_e,
                                                              Matrix<const T, Device::CPU>& mat_i) {
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

  const SizeType nsweeps = nrSweeps(mat_i.size().cols());

  // TODO w last tile is complete anyway, because of setup V well formed
  const SizeType w_nrows = [=, last_rows=mat_i.tileSize({m - 1, 0}).rows()]() {
    if (last_rows == mb)
      return (m - 1) * w_blocksize.rows() + mb - 1;
    else
      return (mb - 1) + 2 * w_blocksize.rows() + (m > 2 ? (m - 2) * w_blocksize.rows() - 1 : 0);
  }();
  const matrix::Distribution dist_w(
      {w_nrows, w_blocksize.cols()},
      w_blocksize,
      dist_i.commGridSize(), dist_i.rankIndex(), dist_i.sourceRankIndex());

  std::cout << "W: " << dist_w.size() << "\n";
  std::cout << "infoW:" << nsweeps << " " << mat_e.tileSize({m - 1, 0}).rows() << "\n";

  print(format::numpy{}, "I", mat_i);

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> t_panels(n_workspaces, mat_i.distribution());
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> w_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Row, T, Device::CPU>> w2_panels(n_workspaces, mat_e.distribution());

  const SizeType last_sweep_tile = (nsweeps - 1) / mb;
  for (SizeType sweep_tile = last_sweep_tile; sweep_tile >= 0; --sweep_tile) {
    std::cout << "sweep_tile=" << sweep_tile << "/" << last_sweep_tile << "\n";

    const SizeType steps = nrStepsPerSweep(sweep_tile * mb, mat_i.size().cols(), mb);

    std::cout << "steps=" << steps << "\n";

    auto& mat_w = w_panels.nextResource();
    auto& mat_t = t_panels.nextResource();
    auto& mat_w2 = w2_panels.nextResource();

    for (SizeType step = 0; step < steps; ++step) {
      const SizeType i_v = sweep_tile + step;

      const LocalTileIndex ij_refls(i_v, sweep_tile);

      const bool affectsTwoRows = i_v < (m - 1);

      std::array<SizeType, 2> sizes;
      sizes[0] = mat_i.tileSize({i_v, sweep_tile}).rows() - 1;
      if (affectsTwoRows)
        sizes[1] = mat_i.tileSize({i_v + 1, sweep_tile}).rows();
      const SizeType w_rows = sizes[0] + (affectsTwoRows ? sizes[1] : 0);

      // TODO fix this
      const SizeType nrefls = std::min(mb, nsweeps - sweep_tile * mb);

      const SizeType k = (nrefls < mb) ? nrefls : std::min(w_rows - 1, mb);
      std::cout << "REFL: nrefls=" << nrefls << " mb=" << mb << " w_rows=" << w_rows << " k=" << k << "\n";

      mat_w.setWidth(k);
      mat_t.setWidth(k);
      mat_w2.setHeight(k);

      // TODO setRange?

      const auto& tile_i = mat_i.read(ij_refls);
      auto tile_v = setupVWellFormed(k, tile_i, mat_w(ij_refls));

      auto tile_t_full = mat_t(LocalTileIndex(i_v, 0));
      auto tile_t = computeTFactor(tile_i, tile_v, splitTile(tile_t_full, {{0, 0}, {k, k}}));

      // Note:
      // W2 is computed before W, because W is used as temporary storage for V.
      for (SizeType j_e = 0; j_e < n; ++j_e) {
        const auto sz_e = mat_e.tileSize({i_v, j_e});

        auto tile_v_up = splitTile(tile_v, {{0, 0}, {sizes[0], k}});
        auto tile_e = mat_e.read(LocalTileIndex(i_v, j_e));
        const auto& tile_e_up = splitTile(tile_e, {{1, 0}, {sizes[0], sz_e.cols()}});

        computeW2(tile_v_up, tile_e_up, T(0), mat_w2(LocalTileIndex(0, j_e)));

        if (affectsTwoRows) {
          auto tile_v_down = splitTile(tile_v, {{sizes[0], 0}, {sizes[1], k}});
          auto tile_e_down = mat_e.read(LocalTileIndex(i_v + 1, j_e));

          computeW2(tile_v_down, tile_e_down, T(1), mat_w2(LocalTileIndex(0, j_e)));
        }
      }

      computeW(mat_w(ij_refls), tile_t);
      auto tile_w = mat_w.read(ij_refls);

      for (SizeType j_e = 0; j_e < n; ++j_e) {
        const LocalTileIndex idx_e(i_v, j_e);
        const auto sz_e = mat_e.tileSize({i_v, j_e});

        auto tile_e = mat_e(idx_e);
        auto tile_e_up = splitTile(tile_e, {{1, 0}, {sizes[0], sz_e.cols()}});
        auto tile_w_up = splitTile(tile_w, {{0, 0}, {sizes[0], k}});
        const auto& tile_w2 = mat_w2.read(idx_e);

        updateE(tile_w_up, tile_w2, std::move(tile_e_up));

        if (affectsTwoRows) {
          auto tile_e_dn = mat_e(LocalTileIndex{i_v + 1, j_e});
          auto tile_w_dn = splitTile(tile_w, {{sizes[0], 0}, {sizes[1], k}});

          updateE(tile_w_dn, tile_w2, std::move(tile_e_dn));
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
