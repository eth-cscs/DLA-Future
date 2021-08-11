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

constexpr bool PRINT_DEBUG = false;

#define PRINT_MATRIX(symbol, mat)        \
  if (PRINT_DEBUG) {                     \
    mat.waitLocalTiles();                \
    print(format::numpy{}, symbol, mat); \
  }

#define PRINT_TILE(symbol, tile)  \
  if (PRINT_DEBUG) {              \
    std::cout << symbol " = ";    \
    print(format::numpy{}, tile); \
  }

template <class T>
auto computeTFactor(SizeType b, hpx::future<matrix::Tile<T, Device::CPU>> tile_t,
                    hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
                    hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_taus)
    -> hpx::shared_future<matrix::Tile<const T, Device::CPU>> {
  auto task = hpx::unwrapping([b](auto tile_t, const auto& tile_v, const auto& tile_taus) {
    // Note: well formed upper-triangular T factor (handy for next calculations, e.g. W)
    tile::set0(tile_t);

    for (SizeType k = 0; k < tile_taus.size().cols(); ++k) {
      const T tau = tile_taus({0, k});

      tile_t({k, k}) = tau;

      for (SizeType j = 0; j < k; ++j) {
        const SizeType size = b - (k - j);
        blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, size, 1, -tau, tile_v.ptr({k - j, j}),
                   tile_v.ld(), tile_v.ptr({0, k}), 1, T(1), tile_t.ptr({j, k}), 1);
      }

      blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit, k,
                 tile_t.ptr(), tile_t.ld(), tile_t.ptr({0, k}), 1);
    }

    PRINT_TILE("T", tile_t);

    return matrix::Tile<const T, Device::CPU>(std::move(tile_t));
  });

  return hpx::dataflow(task, std::move(tile_t), std::move(tile_v), std::move(tile_taus));
}

template <class T>
auto computeW(SizeType b, hpx::future<matrix::Tile<T, Device::CPU>> tile_w,
              hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
              hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t) {
  auto task = [b](auto tile_w, const auto& tile_v, const auto& tile_t) {
    using namespace ::blas;

    tile::set0(tile_w);

    for (SizeType i = 1; i < tile_v.size().rows(); ++i) {
      const auto size = i;
      gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 1, b, size, T(1), tile_v.ptr({i - 1, 0}),
           tile_v.ld() - 1, tile_t.ptr(), tile_t.ld(), T(1), tile_w.ptr({i, 0}), tile_w.ld());
    }
  };

  return hpx::dataflow(hpx::unwrapping(task), std::move(tile_w), tile_v, tile_t);
}

template <class T>
auto computeW(SizeType b, hpx::future<matrix::Tile<T, Device::CPU>> tile_w_up,
              hpx::future<matrix::Tile<T, Device::CPU>> tile_w_down,
              hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
              hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_t) {
  computeW(b, std::move(tile_w_up), tile_v, tile_t);

  auto task = [b](auto tile_w, const auto& tile_v, const auto& tile_t) {
    using namespace ::blas;

    tile::set0(tile_w);

    for (SizeType j = 0; j < tile_v.size().cols(); ++j) {
      const auto size = b - j;
      gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans, 1, b, size, T(1), tile_v.ptr({b - 1, j}),
           tile_v.ld() - 1, tile_t.ptr({j, 0}), tile_t.ld(), T(1), tile_w.ptr({j, 0}), tile_w.ld());
    }
  };

  hpx::dataflow(hpx::unwrapping(task), std::move(tile_w_down), tile_v, tile_t);
}

template <class T>
auto computeW2(SizeType b, hpx::future<matrix::Tile<T, Device::CPU>> tile_w2,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_e) {
  auto task = [b](auto tile_w2, const auto& tile_v, const auto& tile_e) {
    using namespace ::blas;

    tile::set0(tile_w2);

    for (SizeType j = 0; j < b - 1; ++j) {
      const SizeType size = (b - 1) - j;

      gemm(Layout::ColMajor, Op::ConjTrans, Op::NoTrans, 1, tile_e.size().cols(), size, T(1),
           tile_v.ptr({0, 0}), tile_v.ld(), tile_e.ptr({j + 1, 0}), tile_e.ld(), T(1),
           tile_w2.ptr({0, 0}), tile_w2.ld());
    }

    return std::move(tile_w2);
  };

  return hpx::dataflow(hpx::unwrapping(task), std::move(tile_w2), tile_v, tile_e);
}

template <class T>
auto computeW2(SizeType b, hpx::future<matrix::Tile<T, Device::CPU>> tile_w2,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_v,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_e_up,
               hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_e_down) {
  tile_w2 = computeW2(b, std::move(tile_w2), tile_v, tile_e_up);

  auto task_dn = [b](auto tile_w2, const auto& tile_v, const auto& tile_e) {
    using namespace ::blas;

    for (SizeType j = 0; j < b; ++j) {
      const SizeType i = (b - 1) - j;
      const auto size = j + 1;
      gemm(Layout::ColMajor, Op::ConjTrans, Op::NoTrans, 1, tile_e.size().cols(), size, T(1),
           tile_v.ptr({i, j}), tile_v.ld(), tile_e.ptr({0, 0}), tile_e.ld(), T(1), tile_w2.ptr({j, 0}),
           tile_w2.ld());
    }
  };

  return hpx::dataflow(hpx::unwrapping(task_dn), std::move(tile_w2), tile_v, tile_e_down);
}

template <class T>
auto updateE(hpx::future<matrix::Tile<T, Device::CPU>> tile_e,
             hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_w,
             hpx::shared_future<matrix::Tile<const T, Device::CPU>> tile_w2) {
  using blas::Op;
  hpx::dataflow(hpx::unwrapping(tile::gemm_o), Op::NoTrans, Op::NoTrans, T(-1), tile_w, tile_w2, T(1),
                std::move(tile_e));
}

auto debugBarrier = [](const auto& v, const auto& t, const auto& w, const auto& w2) {
  // std::cout << "\n\n\nSTEP     " << message << "\n";

  PRINT_TILE("v", v);
  PRINT_TILE("t", t);

  for (const auto& tile_w : w)
    PRINT_TILE("w", tile_w.get());

  for (const auto& tile_w2 : w2)
    PRINT_TILE("w2", tile_w2.get());

  // std::cout << "STEP-END " << message << "\n\n\n";
};

template <class T>
struct BackTransformationT2B<Backend::MC, Device::CPU, T> {
  static void call(SizeType band_size, Matrix<T, Device::CPU>& mat_e,
                   Matrix<const T, Device::CPU>& mat_v, Matrix<const T, Device::CPU>& mat_taus);
};

template <class T>
void BackTransformationT2B<Backend::MC, Device::CPU, T>::call(SizeType band_size,
                                                              Matrix<T, Device::CPU>& mat_e,
                                                              Matrix<const T, Device::CPU>& mat_v,
                                                              Matrix<const T, Device::CPU>& mat_taus) {
  using matrix::Panel;
  using common::RoundRobin;
  using common::iterate_range2d;

  const SizeType mb = mat_v.blockSize().rows();
  const SizeType n = mat_e.nrTiles().cols();

  constexpr std::size_t n_workspaces = 2;
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> t_panels(n_workspaces, mat_v.distribution());
  RoundRobin<Panel<Coord::Col, T, Device::CPU>> w_panels(n_workspaces, mat_v.distribution());
  RoundRobin<Panel<Coord::Row, T, Device::CPU>> w2_panels(n_workspaces, mat_e.distribution());

  for (SizeType j_v = mat_v.nrTiles().cols() - 1; j_v >= 0; --j_v) {
    for (SizeType i_v = j_v; i_v < mat_v.nrTiles().rows(); ++i_v) {
      const LocalTileIndex ij_refls(i_v, j_v);

      const bool affectsTwoRows = i_v < mat_v.nrTiles().rows() - 1;

      auto& mat_t = t_panels.nextResource();
      mat_t.setRange(GlobalTileIndex(i_v, 0), GlobalTileIndex(i_v + 1, 0));

      auto& mat_w = w_panels.nextResource();
      mat_w.setRange(GlobalTileIndex(i_v, 0), GlobalTileIndex(i_v + (affectsTwoRows ? 2 : 1), 0));

      auto& mat_w2 = w2_panels.nextResource();

      auto tile_t = mat_t(LocalTileIndex(i_v, 0));
      auto tile_v = mat_v.read(ij_refls);
      auto tile_taus = mat_taus.read(ij_refls);

      const auto i_tile_edge = mb - band_size;
      for (SizeType j_subv = mb - band_size; j_subv >= 0; j_subv -= band_size) {
        for (SizeType i_subv = 0; i_subv < mb; i_subv += band_size) {
          const auto i_skewed = i_subv + j_subv;
          const bool onTileEdge = i_skewed == i_tile_edge;

          if (!affectsTwoRows && i_skewed >= mb)  // TODO
            continue;

          // COMPUTE Tfactor
          auto subtile_v = splitTile(tile_v, {{i_subv, j_subv}, {band_size, band_size}});
          auto subtile_taus = splitTile(tile_taus, {{i_subv, j_subv}, {band_size, band_size}});
          auto subtile_t_rw = splitTile(tile_t, {{i_subv, j_subv}, {band_size, band_size}});

          auto subtile_t = computeTFactor(band_size, std::move(subtile_t_rw), subtile_v, subtile_taus);

          // COMPUTE W = V T
          std::vector<hpx::future<matrix::Tile<T, Device::CPU>>> tiles_w_rw;
          tiles_w_rw.emplace_back(mat_w(ij_refls));
          if (affectsTwoRows)
            tiles_w_rw.emplace_back(mat_w(ij_refls + LocalTileSize(1, 0)));

          if (onTileEdge) {
            if (affectsTwoRows) {
              auto subtile_w_up =
                  splitTile(tiles_w_rw[0], {{mb - band_size, j_subv}, {band_size, band_size}});
              auto subtile_w_dn = splitTile(tiles_w_rw[1], {{0, j_subv}, {band_size, band_size}});
              computeW(band_size, std::move(subtile_w_up), std::move(subtile_w_dn), subtile_v,
                       subtile_t);
            }
            else {
              auto subtile_w_up = splitTile(tiles_w_rw[0], {{i_subv, j_subv}, {band_size, band_size}});
              computeW(band_size, std::move(subtile_w_up), subtile_v, subtile_t);
            }
          }
          else {
            auto tile_w = std::move(tiles_w_rw[i_skewed < mb ? 0 : 1]);
            auto subtiles_w =
                splitTileDisjoint(tile_w,
                                  {{{i_skewed % mb, j_subv}, {band_size, band_size}},
                                   {{(i_skewed + band_size) % mb, j_subv}, {band_size, band_size}}});
            computeW(band_size, std::move(subtiles_w[0]), std::move(subtiles_w[1]), subtile_v,
                     subtile_t);
          }

          // COMPUTE W2 = V* E
          const TileElementSize w2_subtile_sz(band_size, mat_e.blockSize().cols());
          for (SizeType j_e = 0; j_e < n; ++j_e) {
            auto tile_w2 = mat_w2(LocalTileIndex(j_v, j_e));
            auto subtile_w2 = splitTile(tile_w2, {{j_subv, 0}, w2_subtile_sz});

            auto tile_e_up = mat_e.read(LocalTileIndex(i_v, j_e));

            if (onTileEdge) {
              if (affectsTwoRows) {
                auto tile_e_dn = mat_e.read(LocalTileIndex(i_v + 1, j_e));
                auto subtile_e_up = splitTile(tile_e_up, {{mb - band_size, 0}, w2_subtile_sz});
                auto subtile_e_dn = splitTile(tile_e_dn, {{0, 0}, w2_subtile_sz});
                computeW2(band_size, std::move(subtile_w2), subtile_v, subtile_e_up, subtile_e_dn);
              }
              else {
                auto subtile_e = splitTile(tile_e_up, {{i_skewed % mb, 0}, w2_subtile_sz});
                computeW2(band_size, std::move(subtile_w2), subtile_v, subtile_e);
              }
            }
            else {
              decltype(tile_e_up) tile_e;

              if (affectsTwoRows)
                tile_e = (i_skewed < mb) ? tile_e_up : mat_e.read(LocalTileIndex(i_v + 1, j_e));
              else
                tile_e = tile_e_up;

              auto subtiles_e = splitTile(tile_e, {
                                                      {{i_skewed % mb, 0}, w2_subtile_sz},
                                                      {{(i_skewed + band_size) % mb, 0}, w2_subtile_sz},
                                                  });
              computeW2(band_size, std::move(subtile_w2), subtile_v, subtiles_e[0], subtiles_e[1]);
            }
          }

          // COMPUTE E -= W @ W2
          for (SizeType j_e = 0; j_e < n; ++j_e) {
            auto tile_e_up = mat_e(LocalTileIndex(i_v, j_e));
            auto tile_w_up = mat_w.read(LocalTileIndex(i_v, j_e));
            auto tile_w2 = mat_w2.read(LocalTileIndex(i_v, j_e));

            auto subtile_w2 = splitTile(tile_w2, {{j_subv, 0}, {w2_subtile_sz}});

            if (onTileEdge) {
              if (affectsTwoRows) {
                auto tile_e_dn = mat_e(LocalTileIndex(i_v + 1, j_e));
                auto tile_w_dn = mat_w.read(LocalTileIndex(i_v + 1, j_e));

                auto subtile_e_up = splitTile(tile_e_up, {{mb - band_size + 1, 0},
                                                          {band_size - 1, w2_subtile_sz.cols()}});
                auto subtile_e_dn = splitTile(tile_e_dn, {{0, 0}, w2_subtile_sz});

                auto subtile_w_up =
                    splitTile(tile_w_up, {{mb - band_size + 1, j_subv}, {band_size - 1, band_size}});
                auto subtile_w_dn = splitTile(tile_w_dn, {{0, j_subv}, {band_size, band_size}});

                updateE(std::move(subtile_e_up), subtile_w_up, subtile_w2);
                updateE(std::move(subtile_e_dn), subtile_w_dn, subtile_w2);
              }
              else {
                auto subtile_e = splitTile(tile_e_up, {{(i_skewed + 1) % mb, 0},
                                                       {band_size - 1, w2_subtile_sz.cols()}});
                auto subtile_w =
                    splitTile(tile_w_up, {{(i_skewed + 1) % mb, j_subv}, {band_size - 1, band_size}});

                updateE(std::move(subtile_e), subtile_w, subtile_w2);
              }
            }
            else {
              decltype(tile_e_up) tile_e;
              decltype(tile_w_up) tile_w;
              if (affectsTwoRows) {
                tile_e = i_skewed < mb ? std::move(tile_e_up) : mat_e(LocalTileIndex(i_v + 1, j_e));
                tile_w = i_skewed < mb ? tile_w_up : mat_w.read(LocalTileIndex(i_v + 1, j_e));
              }
              else {
                tile_e = std::move(tile_e_up);
                tile_w = std::move(tile_w_up);
              }

              auto subtiles_e =
                  splitTileDisjoint(tile_e, {
                                                {{(i_skewed + 1) % mb, 0},
                                                 {band_size - 1, w2_subtile_sz.cols()}},
                                                {{(i_skewed + +band_size) % mb, 0}, w2_subtile_sz},
                                            });

              auto subtiles_w =
                  splitTile(tile_w, {
                                        {{(i_skewed + 1) % mb, j_subv}, {band_size - 1, band_size}},
                                        {{(i_skewed + band_size) % mb, j_subv}, {band_size, band_size}},
                                    });

              updateE(std::move(subtiles_e[0]), subtiles_w[0], subtile_w2);
              updateE(std::move(subtiles_e[1]), subtiles_w[1], subtile_w2);
            }
          }

          hpx::dataflow(hpx::unwrapping(debugBarrier), subtile_v, subtile_t,
                        hpx::when_all(selectRead(mat_w, mat_w.iteratorLocal())),
                        hpx::when_all(selectRead(mat_w2, mat_w2.iteratorLocal())));
        }
      }

      mat_t.reset();
      mat_w.reset();
      mat_w2.reset();

      PRINT_MATRIX("Eb", mat_e);
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
