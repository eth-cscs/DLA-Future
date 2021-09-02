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
};

template <class T>
auto setupVWellFormed(const SizeType k,
                      hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v_compact,
                      hpx::future<matrix::Tile<T, Device::CPU>> tile_v) -> decltype(tile_v_compact) {
  auto unzipV_func = [k](const auto& tile_v_orig, auto tile_v) {
    using namespace lapack;

    const auto b = tile_v_orig.size().cols();

    lacpy(MatrixType::General, b - 1, b, tile_v_orig.ptr({1, 0}), tile_v_orig.ld(), tile_v.ptr({1, 0}),
          tile_v.ld() + 1);
    // TODO is it needed because W = V . T? or is it enough just setting ones?
    laset(MatrixType::Upper, b, k, T(0), T(1), tile_v.ptr({0, 0}), tile_v.ld());
    laset(MatrixType::Lower, b - 1, b - 1, T(0), T(0), tile_v.ptr({b, 0}), tile_v.ld());

    // TODO this is just a workaround...it must be fixed in W?
    if (k < b)
      laset(MatrixType::General, 2 * b - 1, b - k, T(0), T(0), tile_v.ptr({0, k}), tile_v.ld());

    //std::cout << "V = ";
    //print(format::numpy{}, tile_v);

    return matrix::Tile<const T, Device::CPU>(std::move(tile_v));
  };
  return hpx::dataflow(hpx::unwrapping(unzipV_func), std::move(tile_v_compact), std::move(tile_v));
}

template <class T>
auto computeTFactor(const SizeType n, const SizeType k,
                    hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_taus,
                    hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
                    hpx::future<matrix::Tile<T, Device::CPU>> mat_t) -> decltype(tile_v) {
  auto tfactor_task = [n, k](const auto& tile_taus, const auto& tile_v, auto tile_t) {
    using namespace lapack;

    // std::cout << "V = ";
    // print(format::numpy{}, tile_v);

    std::vector<T> taus;
    taus.resize(to_sizet(tile_v.size().cols()));
    for (SizeType i = 0; i < to_SizeType(taus.size()); ++i)
      taus[to_sizet(i)] = *tile_taus.ptr({0, i});

    larft(Direction::Forward, StoreV::Columnwise, n, k, tile_v.ptr(), tile_v.ld(), taus.data(),
          tile_t.ptr(), tile_t.ld());

    // std::cout << "T = ";
    // print(format::numpy{}, tile_t);
    return matrix::Tile<const T, Device::CPU>(std::move(tile_t));
  };
  return hpx::dataflow(hpx::unwrapping(tfactor_task), std::move(tile_taus), std::move(tile_v),
                       std::move(mat_t));
}

template <class T>
auto computeW(hpx::future<matrix::Tile<T, Device::CPU>> tile_v,
              hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t) {
  using namespace blas;

  // hpx::dataflow(hpx::unwrapping([](auto tile_w, const auto& tile_t) {
  //                // std::cout << "\tV = ";
  //                // print(format::numpy{}, tile_w);
  //                // std::cout << "\tT = ";
  //                // print(format::numpy{}, tile_t);
  //                tile::trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit, T(1), tile_t,
  //                tile_w); std::cout << "W = "; print(format::numpy{}, tile_w);
  //              }),
  //              std::move(tile_v), std::move(tile_t));

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
  //                    // std::cout << "\t_V = ";
  //                    // print(format::numpy{}, tile_v);
  //                    // std::cout << "\t_E = ";
  //                    // print(format::numpy{}, tile_e);
  //                    // std::cout << "\tbeta = " << beta << "\n";
  //                    // std::cout << "\t_W2pre = ";
  //                    // print(format::numpy{}, tile_w2);
  //                    tile::gemm(Op::ConjTrans, Op::NoTrans, T(1), tile_v, tile_e, beta, tile_w2);
  //                    // std::cout << "_W2 = ";
  //                    // print(format::numpy{}, tile_w2);
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

  // auto func = hpx::unwrapping([](auto&& tile_w, auto&& tile_w2, auto tile_e) {
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
  // hpx::dataflow(func, std::move(tile_w), std::move(tile_w2), std::move(tile_e));

  hpx::dataflow(hpx::unwrapping(tile::gemm_o), Op::NoTrans, Op::NoTrans, T(-1), std::move(tile_w),
                std::move(tile_w2), T(1), std::move(tile_e));
}

template <class T>
void BackTransformationT2B<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_e,
                                                              Matrix<const T, Device::CPU>& mat_i) {
  using matrix::Panel;
  using common::RoundRobin;
  using common::iterate_range2d;

  const SizeType band_size = mat_i.blockSize().rows();
  const SizeType m = mat_e.nrTiles().rows();
  const SizeType n = mat_e.nrTiles().cols();

  const auto& dist_i = mat_i.distribution();
  const TileElementSize w_blocksize(dist_i.blockSize().rows() * 2 - 1, dist_i.blockSize().cols());
  const matrix::Distribution dist_w({w_blocksize.rows() * dist_i.nrTiles().rows(), w_blocksize.cols()},
                                    w_blocksize, dist_i.commGridSize(), dist_i.rankIndex(),
                                    dist_i.sourceRankIndex());

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> t_panels(n_workspaces, mat_i.distribution());
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> w_panels(n_workspaces, dist_w);
  RoundRobin<Panel<Coord::Row, T, Device::CPU>> w2_panels(n_workspaces, mat_e.distribution());

  for (SizeType j_v = mat_i.nrTiles().cols() - 1; j_v >= 0; --j_v) {
    for (SizeType i_v = j_v; i_v < mat_i.nrTiles().rows(); ++i_v) {
      const LocalTileIndex ij_refls(i_v, j_v);

      const bool affectsTwoRows = i_v < mat_i.nrTiles().rows() - 1;

      auto& mat_t = t_panels.nextResource();
      mat_t.setRange(GlobalTileIndex(i_v, 0), GlobalTileIndex(i_v + 1, 0));

      auto& mat_w = w_panels.nextResource();
      // TODO mat_w.setRange(GlobalTileIndex(i_v, 0), GlobalTileIndex(i_v + (affectsTwoRows ? 2 : 1), 0));

      // TODO check and fix this
      const SizeType refl_size = affectsTwoRows ? 2 * band_size - 1 : band_size - 1;
      const SizeType k = affectsTwoRows ? band_size : band_size - 2;

      const auto& tile_i = mat_i.read(ij_refls);
      auto tile_v = setupVWellFormed(k, tile_i, mat_w(ij_refls));

      auto tile_t = computeTFactor(refl_size, k, tile_i, tile_v, mat_t(LocalTileIndex(i_v, 0)));

      auto& mat_w2 = w2_panels.nextResource();
      for (SizeType j = 0; j < n; ++j) {
        auto tile_vs = splitTile(tile_v, {
                                             {{0, 0}, {band_size - 1, band_size}},
                                             {{band_size - 1, 0}, {band_size, band_size}},
                                         });

        auto tile_e_up = mat_e.read(LocalTileIndex(i_v, j));
        const auto& tile_e = splitTile(tile_e_up, {{1, 0}, {band_size - 1, mat_e.blockSize().cols()}});
        computeW2(tile_vs[0], tile_e, T(0), mat_w2(LocalTileIndex(0, j)));

        if (affectsTwoRows) {
          const auto& tile_e_down = mat_e.read(LocalTileIndex(i_v + 1, j));
          computeW2(tile_vs[1], tile_e_down, T(1), mat_w2(LocalTileIndex(0, j)));
        }
      }

      computeW(mat_w(ij_refls), tile_t);

      const auto affected_rows =
          iterate_range2d(LocalTileIndex(i_v, 0), LocalTileIndex(std::min<SizeType>(i_v + 2, m), n));
      for (const auto& ij : affected_rows) {
        auto tile_e = mat_e(ij);
        auto tile_w = mat_w.read(ij_refls);
        if (ij.row() == i_v) {
          tile_e = matrix::splitTile(tile_e, {{1, 0}, {band_size - 1, mat_e.blockSize().cols()}});
          tile_w = matrix::splitTile(tile_w, {{0, 0}, {band_size - 1, band_size}});
        }
        else {
          tile_w = matrix::splitTile(tile_w, {{band_size - 1, 0}, {band_size, band_size}});
        }
        auto tile_w2 = mat_w2.read(ij);
        updateE(tile_w, tile_w2, std::move(tile_e));
      }

      // PRINT_MATRIX("Eb", mat_e);

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
