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

#include <algorithm>

#include <pika/datastructures/tuple.hpp>
#include <pika/future.hpp>
#include <pika/modules/iterator_support.hpp>
#include <pika/parallel/algorithms/partition.hpp>
#include <pika/parallel/algorithms/sort.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/callable_object.h"
#include "dlaf/eigensolver/tridiag_solver/api.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
struct TridiagSolver<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_a, SizeType i_begin, SizeType i_end,
                   Matrix<T, Device::CPU>& d, Matrix<T, Device::CPU>& z,
                   Matrix<SizeType, Device::CPU>& index, Matrix<T, Device::CPU>& mat_ws,
                   Matrix<T, Device::CPU>& mat_ev);
  static void call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a,
                   Matrix<T, Device::CPU>& mat_ev);
};

template <class T>
void cuppensDecomposition(const matrix::Tile<T, Device::CPU>& top,
                          const matrix::Tile<T, Device::CPU>& bottom) {
  (void) top;
  (void) bottom;

  T offdiag_val = top(TileElementIndex{top.size().rows() - 1, top.size().cols() - 1});
  T& top_diag_val = top(TileElementIndex{top.size().rows() - 1, top.size().cols() - 2});
  T& bottom_diag_val = bottom(TileElementIndex{0, 0});

  top_diag_val -= offdiag_val;
  bottom_diag_val -= offdiag_val;
}

DLAF_MAKE_CALLABLE_OBJECT(cuppensDecomposition);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(cuppensDecomposition, cuppensDecomposition_o)

// Copies and normalizes a row of the `tile` into the column vector tile `col`
//
template <class T>
void copyTileRowAndNormalize(SizeType row, const matrix::Tile<const T, Device::CPU>& tile,
                             const matrix::Tile<T, Device::CPU>& col) {
  for (SizeType i = 0; i < tile.size().rows(); ++i) {
    col(TileElementIndex(i, 0)) = tile(TileElementIndex(row, i)) / std::sqrt(2);
  }
}

DLAF_MAKE_CALLABLE_OBJECT(copyTileRowAndNormalize);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(copyTileRowAndNormalize, copyTileRowAndNormalize_o)

// The bottom row of Q1 and the top row of Q2
//
// Note that the norm of `z` is sqrt(2) because it is a concatination of two normalized vectors. Hence to
// normalize `z` we have to divide by sqrt(2). That is handled in `copyTileRowAndNormalize()`
template <class T>
void assembleZVec(SizeType i_begin, SizeType i_middle, SizeType i_end,
                  Matrix<const T, Device::CPU>& mat_ev, Matrix<T, Device::CPU>& z) {
  using pika::threads::thread_priority;
  using pika::execution::experimental::start_detached;
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;

  // Iterate over tiles of Q1 and Q2 around the split row `i_middle`.
  for (SizeType i = i_begin; i <= i_end; ++i) {
    // Move to the row below `i_middle` for `Q2`
    SizeType mat_ev_row = i_middle + ((i > i_middle) ? 1 : 0);
    GlobalTileIndex mat_ev_idx(mat_ev_row, i);
    // Take the last row of a `Q1` tile or the first row of a `Q2` tile
    SizeType tile_row = (i > i_middle) ? 0 : mat_ev.distribution().tileSize(mat_ev_idx).rows() - 1;
    GlobalTileIndex z_idx(i, 0);
    // Copy the row into the column vector `z`
    whenAllLift(tile_row, mat_ev.read_sender(mat_ev_idx), z.readwrite_sender(z_idx)) |
        copyTileRowAndNormalize(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

template <class T>
void copyDiagTile(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                  const matrix::Tile<T, Device::CPU>& diag_tile) {
  for (SizeType i = 0; i < tridiag_tile.size().rows(); ++i) {
    diag_tile(TileElementIndex(i, 0)) = tridiag_tile(TileElementIndex(i, 0));
  }
}

DLAF_MAKE_CALLABLE_OBJECT(copyDiagTile);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(copyDiagTile, copyDiagTile_o)

template <class T>
void assembleDiag(SizeType i_begin, SizeType i_end, Matrix<const T, Device::CPU>& mat_a,
                  Matrix<T, Device::CPU>& d) {
  using pika::threads::thread_priority;
  using pika::execution::experimental::start_detached;
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;

  for (SizeType i = i_begin; i <= i_end; ++i) {
    whenAllLift(mat_a.read_sender(GlobalTileIndex(i, 0)), d.readwrite_sender(GlobalTileIndex(i, 0))) |
        copyDiagTile(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

template <class T>
T extractRho(const matrix::Tile<const T, Device::CPU>& mat_a_tile) {
  // Get the bottom-right element of the tile
  // Multiply by factor 2 to account for the normalization of `z`
  return 2 * mat_a_tile(TileElementIndex(mat_a_tile.size().rows() - 1, 1));
}

DLAF_MAKE_CALLABLE_OBJECT(extractRho);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(extractRho, extractRho_o)

// Returns the maximum element of a portion of a column vector from tile indices `i_begin` to `i_end` including.
//
template <class T>
pika::future<T> maxVectorElement(SizeType i_begin, SizeType i_end, Matrix<const T, Device::CPU>& vec) {
  std::vector<pika::future<T>> tiles_max;
  tiles_max.reserve(to_sizet(i_end - i_begin + 1));
  for (SizeType i = i_begin; i <= i_end; ++i) {
    tiles_max.push_back(pika::dataflow(pika::unwrapping(tile::internal::lange_o), lapack::Norm::Max,
                                       vec.read(LocalTileIndex(i, 0))));
  }

  auto tol_calc_fn = [](const std::vector<T>& maxvals) {
    return *std::max_element(maxvals.begin(), maxvals.end());
  };
  return pika::dataflow(pika::unwrapping(std::move(tol_calc_fn)), std::move(tiles_max));
}

// The tolerance calculation is the same as the one used in LAPACK's stedc implementation [1].
//
// [1] LAPACK 3.10.0, file dlaed2.f, line 315, variable TOL
template <class T>
T calcTolerance(T dmax, T zmax) {
  return 8 * std::numeric_limits<T>::epsilon() * std::max(dmax, zmax);
}

template <class T>
std::vector<pika::future<matrix::Tile<T, Device::CPU>>> collectVectorTileFutures(
    SizeType i_begin, SizeType i_end, Matrix<T, Device::CPU>& vec) {
  std::size_t num_tiles = to_sizet(i_end - i_begin + 1);
  std::vector<pika::future<matrix::Tile<T, Device::CPU>>> tiles;
  tiles.reserve(num_tiles);

  for (SizeType i = i_begin; i <= i_end; ++i) {
    GlobalTileIndex tile_idx(i, 0);
    tiles.push_back(vec(tile_idx));
  }

  return tiles;
}

template <class T>
SizeType permutateZVecZeroesToBottom(
    T rho, T tol, const std::vector<matrix::Tile<T, Device::CPU>>& d_tiles,
    const std::vector<matrix::Tile<T, Device::CPU>>& z_tiles,
    const std::vector<matrix::Tile<SizeType, Device::CPU>>& index_tiles) {
  SizeType len =
      (to_SizeType(d_tiles.size()) - 1) * d_tiles.front().size().rows() + d_tiles.back().size().rows();
  TileElementIndex zero_idx(0, 0);
  T* d_ptr = d_tiles[0].ptr(zero_idx);
  T* z_ptr = z_tiles[0].ptr(zero_idx);
  SizeType* index_ptr = index_tiles[0].ptr(zero_idx);

  // false values go to the bottom
  auto smallz_fn = [rho, tol](const pika::tuple<T, T, SizeType>& t) {
    return rho * std::abs(std::get<1>(t)) > tol;
  };

  auto it_begin = pika::util::make_zip_iterator(d_ptr, z_ptr, index_ptr);
  auto it_end = pika::util::make_zip_iterator(d_ptr + len, z_ptr + len, index_ptr + len);
  // Note that stable_partition preserves the relative order of elements in the two groups
  // The iterator points to the first zero of the z vector after all zeroes were moved to the bottom.
  auto it_split = pika::stable_partition(pika::execution::par, it_begin, it_end, smallz_fn);
  // Returns the number of non-zero entries in `z`
  return std::distance(it_begin, it_split);
}

// Returns true if `d1` is close to `d2`.
//
// Given's deflation condition is the same as the one used in LAPACK's stedc implementation [1].
//
// [1] LAPACK 3.10.0, file dlaed2.f, line 393
template <class T>
bool givensDeflationCondition(T tol, T d1, T d2, T z1, T z2) {
  // Note that this is similar to calling `rotg()` but we want to make sure that the condition is
  // satisfied before modifying z1, z2 that is why the function is not used here.
  return std::abs(z1 * z2 * (d1 - d2) / (z1 * z1 + z2 * z2)) < tol;
}

template <class T>
void sortAscendingBasedOnDiagonal(const std::vector<matrix::Tile<T, Device::CPU>>& d_tiles,
                                  const std::vector<matrix::Tile<T, Device::CPU>>& z_tiles,
                                  const std::vector<matrix::Tile<SizeType, Device::CPU>>& index_tiles) {
  SizeType len =
      (to_SizeType(d_tiles.size()) - 1) * d_tiles.front().size().rows() + d_tiles.back().size().rows();
  TileElementIndex zero_idx(0, 0);
  T* d_ptr = d_tiles[0].ptr(zero_idx);
  T* z_ptr = z_tiles[0].ptr(zero_idx);
  SizeType* index_ptr = index_tiles[0].ptr(zero_idx);

  // The entries of the tuples are elements of the vectors `d`, `z` and `index` in that order.
  auto sortComp = [](const pika::tuple<T, T, SizeType>& t1, const pika::tuple<T, T, SizeType>& t2) {
    return std::get<0>(t1) < std::get<0>(t2);
  };

  // Note that pika's zip iterator may not exactly model the requirements of `std::sort` and it can't be
  // used here. [1]
  //
  // [1]: https://stackoverflow.com/questions/13840998/sorting-zipped-locked-containers-in-c-using-boost-or-the-stl
  pika::sort(pika::execution::par, pika::util::make_zip_iterator(d_ptr, z_ptr, index_ptr),
             pika::util::make_zip_iterator(d_ptr + len, z_ptr + len, index_ptr + len), sortComp);
}

inline void initIndexTile(SizeType tile_row, const matrix::Tile<SizeType, Device::CPU>& index) {
  for (SizeType i = 0; i < index.size().rows(); ++i) {
    index(TileElementIndex(i, 0)) = tile_row + i;
  }
}

DLAF_MAKE_CALLABLE_OBJECT(initIndexTile);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(initIndexTile, initIndexTile_o)

inline void initIndex(SizeType i_begin, SizeType i_end, Matrix<SizeType, Device::CPU>& index) {
  using dlaf::internal::whenAllLift;
  using pika::threads::thread_priority;
  using dlaf::internal::Policy;
  using pika::execution::experimental::start_detached;

  for (SizeType i = i_begin; i <= i_end; ++i) {
    GlobalTileIndex tile_idx(i, 0);
    SizeType tile_row = index.distribution().globalElementIndex(tile_idx, TileElementIndex(0, 0)).row();
    whenAllLift(tile_row, index.readwrite_sender(tile_idx)) |
        initIndexTile(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_a, SizeType i_begin,
                                                      SizeType i_end, Matrix<T, Device::CPU>& d,
                                                      Matrix<T, Device::CPU>& z,
                                                      Matrix<SizeType, Device::CPU>& index,
                                                      Matrix<T, Device::CPU>& mat_ws,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  using dlaf::internal::whenAllLift;
  using pika::threads::thread_priority;
  using dlaf::internal::Policy;
  using pika::execution::experimental::start_detached;
  using Solver = TridiagSolver<Backend::MC, Device::CPU, T>;

  if (i_begin == i_end) {
    // Solve leaf eigensystem with stedc
    whenAllLift(mat_a.readwrite_sender(LocalTileIndex(i_begin, 0)),
                mat_ev.readwrite_sender(LocalTileIndex(i_begin, i_begin))) |
        tile::stedc(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
    return;
  }
  SizeType i_midpoint = (i_begin + i_end) / 2;

  // Cuppen's tridiagonal decomposition
  dlaf::internal::whenAllLift(mat_a.readwrite_sender(LocalTileIndex(i_midpoint, 0)),
                              mat_a.readwrite_sender(LocalTileIndex(i_midpoint + 1, 0))) |
      cuppensDecomposition(Policy<Backend::MC>(thread_priority::normal)) | start_detached();

  // Left leaf
  Solver::call(mat_a, i_begin, i_midpoint, d, z, index, mat_ws, mat_ev);
  // Right leaf
  Solver::call(mat_a, i_midpoint + 1, i_end, d, z, index, mat_ws, mat_ev);

  // Form D + rzz^T from `mat_a` and `mat_ev`
  assembleZVec(i_begin, i_midpoint, i_end, mat_ev, z);
  assembleDiag(i_begin, i_end, mat_a, d);
  pika::shared_future<T> rho_fut =
      pika::dataflow(pika::unwrapping(extractRho<T>), mat_a.read(LocalTileIndex(i_midpoint, 0)));

  // Calculate the tolerance used for deflation
  pika::future<T> dmax_fut = maxVectorElement(i_begin, i_end, d);
  pika::future<T> zmax_fut = maxVectorElement(i_begin, i_end, z);
  pika::shared_future<T> tol_fut =
      pika::dataflow(pika::unwrapping(calcTolerance<T>), std::move(dmax_fut), std::move(zmax_fut));

  // Initialize the index
  initIndex(i_begin, i_end, index);

  {
    // Sort the diagonal in ascending order
    auto d_tiles = collectVectorTileFutures(i_begin, i_end, d);
    auto z_tiles = collectVectorTileFutures(i_begin, i_end, z);
    auto index_tiles = collectVectorTileFutures(i_begin, i_end, index);
    pika::dataflow(pika::unwrapping(sortAscendingBasedOnDiagonal<T>), std::move(d_tiles),
                   std::move(z_tiles), std::move(index_tiles));
  }

  // The number of non-zero entries in `z`
  pika::future<SizeType> k_fut;
  {
    // Deflate based on `z` entries close to zero
    auto d_tiles = collectVectorTileFutures(i_begin, i_end, d);
    auto z_tiles = collectVectorTileFutures(i_begin, i_end, z);
    auto index_tiles = collectVectorTileFutures(i_begin, i_end, index);
    k_fut = pika::dataflow(pika::unwrapping(permutateZVecZeroesToBottom<T>), tol_fut, rho_fut,
                           std::move(d_tiles), std::move(z_tiles), std::move(index_tiles));
  }

  // Find evals of D + rzz^T with laed4 (root solver)
  // Form evecs
  // Gemm
}

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(comm::CommunicatorGrid grid,
                                                      Matrix<T, Device::CPU>& mat_a,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  (void) grid;
  (void) mat_a;
  (void) mat_ev;
}

/// ---- ETI
#define DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct TridiagSolver<BACKEND, DEVICE, DATATYPE>;

DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, double)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_CUDA
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, float)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, double)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif

}
}
}
