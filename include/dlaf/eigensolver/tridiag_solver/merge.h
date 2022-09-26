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

#include <pika/algorithm.hpp>
#include <pika/future.hpp>
#include <pika/parallel/algorithms/for_each.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/eigensolver/tridiag_solver/coltype.h"
#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/multiplication/general.h"
#include "dlaf/permutations/general.h"
#include "dlaf/permutations/general/impl.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix/print_csv.h"

namespace dlaf::eigensolver::internal {

template <class T>
struct GivensRotation {
  SizeType i;  // the first column index
  SizeType j;  // the second column index
  T c;         // cosine
  T s;         // sine
};

// Auxiliary matrix and vectors used for the D&C algorithm
template <class T, Device device>
struct WorkSpace {
  // Extra workspace for Q1', Q2' and U1' if n2 > n1 or U2' if n1 >= n2 which are packed as follows:
  //
  //   ┌──────┬──────┬───┐
  //   │  Q1' │  Q2' │   │
  //   │      │      │   │
  //   ├──────┤      │   │
  //   │      └──────┘   │
  //   │                 │
  //   ├────────────┐    │
  //   │     U1'    │    │
  //   │            │    │
  //   └────────────┴────┘
  //
  Matrix<T, device> mat1;
  Matrix<T, device> mat2;

  // Holds the values of the deflated diagonal sorted in ascending order
  Matrix<T, device> dtmp;
  // Holds the values of Cuppen's rank-1 vector
  Matrix<T, device> z;
  // Holds the values of the rank-1 update vector sorted corresponding to `d_defl`
  Matrix<T, device> ztmp;

  // Steps:
  //
  // 1. Sort index based on diagonal values in ascending order. This creates a map sorted to inital indices
  //
  //        initial <--- pre_sorted
  //
  // 2. Sort index based on column types such that all deflated entries are at the end
  //
  //        pre_sorted <--- deflated
  //
  // 3. Sort index based on updated diagonal values in ascending order. The diagonal conatins eigenvalues
  //    of the deflated problem and deflated entreis from the initial diagonal
  //
  //        deflated <--- post_sorted
  //
  // 4. Sort index based on column types such that matrices `Q` and `U` are in matrix multiplication form.
  //
  //        post_sorted <--- matmul
  //
  Matrix<SizeType, device> i1;
  Matrix<SizeType, device> i2;
  Matrix<SizeType, device> i3;

  // Assigns a type to each column of Q which is used to calculate the permutation indices for Q and U
  // that bring them in matrix multiplication form.
  Matrix<ColType, device> c;
};

// Device::GPU
template <class T, Device D>
struct WorkSpaceHostMirror {
  matrix::Matrix<T, Device::CPU> evals;
  matrix::Matrix<T, Device::CPU> mat1;
  matrix::Matrix<T, Device::CPU> dtmp;
  matrix::Matrix<T, Device::CPU> z;
  matrix::Matrix<T, Device::CPU> ztmp;
  matrix::Matrix<SizeType, Device::CPU> i2;
  matrix::Matrix<ColType, Device::CPU> c;
};

template <class T>
struct WorkSpaceHostMirror<T, Device::CPU> {
  matrix::Matrix<T, Device::CPU>& evals;
  matrix::Matrix<T, Device::CPU>& mat1;
  matrix::Matrix<T, Device::CPU>& dtmp;
  matrix::Matrix<T, Device::CPU>& z;
  matrix::Matrix<T, Device::CPU>& ztmp;
  matrix::Matrix<SizeType, Device::CPU>& i2;
  matrix::Matrix<ColType, Device::CPU>& c;
};

template <class T>
Matrix<T, Device::CPU> initMirrorMatrix(Matrix<T, Device::GPU>& mat) {
  return Matrix<T, Device::CPU>(mat.distribution());
}

template <class T>
Matrix<T, Device::CPU>& initMirrorMatrix(Matrix<T, Device::CPU>& mat) {
  return mat;
}

// Calculates the problem size in the tile range [i_begin, i_end]
inline SizeType problemSize(SizeType i_begin, SizeType i_end, const matrix::Distribution& distr) {
  SizeType nb = distr.blockSize().rows();
  SizeType nbr = distr.tileSize(GlobalTileIndex(i_end, 0)).rows();
  return (i_end - i_begin) * nb + nbr;
}

// The index starts at `0` for tiles in the range [i_begin, i_end].
template <Device D>
inline void initIndex(SizeType i_begin, SizeType i_end, Matrix<SizeType, D>& index) {
  SizeType nb = index.distribution().blockSize().rows();
  for (SizeType i = i_begin; i <= i_end; ++i) {
    GlobalTileIndex tile_idx(i, 0);
    SizeType tile_row = (i - i_begin) * nb;
    initIndexTileAsync<D>(tile_row, index.readwrite_sender(tile_idx));
  }
}

// The bottom row of Q1 and the top row of Q2. The bottom row of Q1 is negated if `rho < 0`.
//
// Note that the norm of `z` is sqrt(2) because it is a concatination of two normalized vectors. Hence
// to normalize `z` we have to divide by sqrt(2).
template <class T, Device D>
void assembleZVec(SizeType i_begin, SizeType i_split, SizeType i_end, pika::shared_future<T> rho_fut,
                  Matrix<const T, D>& mat_ev, Matrix<T, D>& z) {
  // Iterate over tiles of Q1 and Q2 around the split row `i_middle`.
  for (SizeType i = i_begin; i <= i_end; ++i) {
    // True if tile is in Q1
    bool top_tile = i <= i_split;
    // Move to the row below `i_middle` for `Q2`
    SizeType mat_ev_row = i_split + ((top_tile) ? 0 : 1);
    GlobalTileIndex mat_ev_idx(mat_ev_row, i);
    // Take the last row of a `Q1` tile or the first row of a `Q2` tile
    GlobalTileIndex z_idx(i, 0);

    // Copy the row into the column vector `z`
    assembleRank1UpdateVectorTileAsync<T, D>(top_tile, rho_fut, mat_ev.read_sender(mat_ev_idx),
                                             z.readwrite_sender(z_idx));
  }
}

// Multiply by factor 2 to account for the normalization of `z` vector and make sure rho > 0 f
//
template <class T>
pika::future<T> scaleRho(pika::shared_future<T> rho_fut) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  return std::move(rho_fut) |
         di::transform(di::Policy<Backend::MC>(), [](T rho) { return 2 * std::abs(rho); }) |
         ex::make_future();
}

// Returns the maximum element of a portion of a column vector from tile indices `i_begin` to `i_end`
// including.
//
template <class T, Device D>
auto maxVectorElement(SizeType i_begin, SizeType i_end, Matrix<const T, D>& vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<ex::unique_any_sender<T>> tiles_max;
  tiles_max.reserve(to_sizet(i_end - i_begin + 1));
  for (SizeType i = i_begin; i <= i_end; ++i) {
    tiles_max.push_back(maxElementInColumnTileAsync<T, D>(vec.read_sender(LocalTileIndex(i, 0))));
  }

  auto tol_calc_fn = [](const std::vector<T>& maxvals) {
    return *std::max_element(maxvals.begin(), maxvals.end());
  };

  return ex::when_all_vector(std::move(tiles_max)) |
         di::transform(di::Policy<Backend::MC>(), std::move(tol_calc_fn));
}

// The tolerance calculation is the same as the one used in LAPACK's stedc implementation [1].
//
// [1] LAPACK 3.10.0, file dlaed2.f, line 315, variable TOL
template <class T, Device D>
pika::future<T> calcTolerance(SizeType i_begin, SizeType i_end, Matrix<const T, D>& d,
                              Matrix<const T, D>& z) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto dmax_fut = maxVectorElement(i_begin, i_end, d);
  auto zmax_fut = maxVectorElement(i_begin, i_end, z);

  auto tol_fn = [](T dmax, T zmax) {
    return 8 * std::numeric_limits<T>::epsilon() * std::max(dmax, zmax);
  };

  return ex::when_all(std::move(dmax_fut), std::move(zmax_fut)) |
         di::transform(di::Policy<Backend::MC>(), std::move(tol_fn)) | ex::make_future();
}

struct TileCollector {
  SizeType i_begin;
  SizeType i_end;

private:
  template <class T, Device D>
  std::pair<GlobalTileIndex, GlobalTileSize> getRange(Matrix<const T, D>& mat) {
    SizeType ntiles = i_end - i_begin + 1;
    bool is_col_matrix = mat.distribution().size().cols() == 1;
    SizeType col_begin = (is_col_matrix) ? 0 : i_begin;
    SizeType col_sz = (is_col_matrix) ? 1 : ntiles;
    return std::make_pair(GlobalTileIndex(i_begin, col_begin), GlobalTileSize(ntiles, col_sz));
  }

public:
  template <class T, Device D>
  auto read(Matrix<const T, D>& mat) {
    auto [begin, end] = getRange(mat);
    return matrix::util::collectReadTiles(begin, end, mat);
  }

  template <class T, Device D>
  auto readwrite(Matrix<T, D>& mat) {
    auto [begin, end] = getRange(mat);
    return matrix::util::collectReadWriteTiles(begin, end, mat);
  }
};

// Sorts an index `in_index_tiles` based on values in `vals_tiles` in ascending order into the index
// `out_index_tiles` where `vals_tiles` is composed of two pre-sorted ranges in ascending order that
// are merged, the first is [0, k) and the second is [k, n).
//
template <class T, Device D>
void sortIndex(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
               Matrix<const T, D>& vec, Matrix<const SizeType, D>& in_index,
               Matrix<SizeType, D>& out_index) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_end, vec.distribution());
  auto sort_fn = [n](const auto& k, const auto& vec_futs, const auto& in_index_futs,
                     const auto& out_index, [[maybe_unused]] auto&&... ts) {
    DLAF_ASSERT(k <= n, k, n);

    TileElementIndex zero_idx(0, 0);
    const T* v_ptr = vec_futs[0].get().ptr(zero_idx);
    const SizeType* in_index_ptr = in_index_futs[0].get().ptr(zero_idx);
    SizeType* out_index_ptr = out_index[0].ptr(zero_idx);

    auto begin_it = in_index_ptr;
    auto split_it = in_index_ptr + k;
    auto end_it = in_index_ptr + n;
    if constexpr (D == Device::CPU) {
      auto cmp = [v_ptr](SizeType i1, SizeType i2) { return v_ptr[i1] < v_ptr[i2]; };
      pika::merge(pika::execution::par, begin_it, split_it, split_it, end_it, out_index_ptr,
                  std::move(cmp));
    }
    else {
#ifdef DLAF_WITH_GPU
      mergeIndicesOnDevice(begin_it, split_it, end_it, out_index_ptr, v_ptr, ts...);
#endif
    }
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(std::move(k_fut), ex::when_all_vector(tc.read<T, D>(vec)),
                             ex::when_all_vector(tc.read<SizeType, D>(in_index)),
                             ex::when_all_vector(tc.readwrite<SizeType, D>(out_index)));

  ex::start_detached(
      di::transform<di::TransformDispatchType::Plain, false>(di::Policy<DefaultBackend_v<D>>(),
                                                             std::move(sort_fn), std::move(sender)));
}

// Applies `index` to `in` to get `out`
//
// Note: `pika::unwrapping()` can't be used on this function because std::vector<Tile<const >> is
// copied internally which requires that Tile<const > is copiable which it isn't. As a consequence the
// API can't be simplified unless const is dropped.
template <class T, Device D>
void applyIndex(SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& index,
                Matrix<const T, D>& in, Matrix<T, D>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_end, index.distribution());
  auto applyIndex_fn = [n](const auto& index_futs, const auto& in_futs, const auto& out,
                           [[maybe_unused]] auto&&... ts) {
    TileElementIndex zero_idx(0, 0);
    const SizeType* i_ptr = index_futs[0].get().ptr(zero_idx);
    const T* in_ptr = in_futs[0].get().ptr(zero_idx);
    T* out_ptr = out[0].ptr(zero_idx);

    if constexpr (D == Device::CPU) {
      for (SizeType i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i_ptr[i]];
      }
    }
    else {
#ifdef DLAF_WITH_GPU
      applyIndexOnDevice(n, i_ptr, in_ptr, out_ptr, ts...);
#endif
    }
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(ex::when_all_vector(tc.read(index)), ex::when_all_vector(tc.read(in)),
                             ex::when_all_vector(tc.readwrite(out)));
  ex::start_detached(
      di::transform<di::TransformDispatchType::Plain, false>(di::Policy<DefaultBackend_v<D>>(),
                                                             std::move(applyIndex_fn),
                                                             std::move(sender)));
}

template <Device D>
inline void invertIndex(SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& in,
                        Matrix<SizeType, D>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto inv_fn = [n](const auto& in_tiles_futs, const auto& out_tiles, [[maybe_unused]] auto&&... ts) {
    TileElementIndex zero(0, 0);
    const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero);
    SizeType* out_ptr = out_tiles[0].ptr(zero);

    if constexpr (D == Device::CPU) {
      for (SizeType i = 0; i < n; ++i) {
        out_ptr[in_ptr[i]] = i;
      }
    }
    else {
      invertIndexOnDevice(n, in_ptr, out_ptr, ts...);
    }
  };

  TileCollector tc{i_begin, i_end};
  auto sender = ex::when_all(ex::when_all_vector(tc.read(in)), ex::when_all_vector(tc.readwrite(out)));
  ex::start_detached(
      di::transform<di::TransformDispatchType::Plain, false>(di::Policy<DefaultBackend_v<D>>(),
                                                             std::move(inv_fn), std::move(sender)));
}

// The index array `out_ptr` holds the indices of elements of `c_ptr` that order it such that
// ColType::Deflated entries are moved to the end. The `c_ptr` array is implicitly ordered according to
// `in_ptr` on entry.
//
inline SizeType stablePartitionIndexForDeflationArrays(SizeType n, const ColType* c_ptr,
                                                       const SizeType* in_ptr, SizeType* out_ptr) {
  // Get the number of non-deflated entries
  SizeType k = 0;
  for (SizeType i = 0; i < n; ++i) {
    if (c_ptr[i] != ColType::Deflated)
      ++k;
  }

  SizeType i1 = 0;  // index of non-deflated values in out
  SizeType i2 = k;  // index of deflated values
  for (SizeType i = 0; i < n; ++i) {
    SizeType ii = in_ptr[i];
    SizeType& io = (c_ptr[ii] != ColType::Deflated) ? i1 : i2;
    out_ptr[io] = ii;
    ++io;
  }
  return k;
}

template <Device D>
inline pika::future<SizeType> stablePartitionIndexForDeflation(SizeType i_begin, SizeType i_end,
                                                               Matrix<const ColType, D>& c,
                                                               Matrix<const SizeType, D>& in,
                                                               Matrix<SizeType, D>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto part_fn = [n](const auto& c_tiles_futs, const auto& in_tiles_futs, const auto& out_tiles,
                     [[maybe_unused]] auto&&... ts) {
    TileElementIndex zero_idx(0, 0);
    const ColType* c_ptr = c_tiles_futs[0].get().ptr(zero_idx);
    const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero_idx);
    SizeType* out_ptr = out_tiles[0].ptr(zero_idx);

    if constexpr (D == Device::CPU) {
      return stablePartitionIndexForDeflationArrays(n, c_ptr, in_ptr, out_ptr);
    }
    else {
      return stablePartitionIndexOnDevice(n, c_ptr, in_ptr, out_ptr, ts...);
    }
  };

  TileCollector tc{i_begin, i_end};
  auto sender = ex::when_all(ex::when_all_vector(tc.read(c)), ex::when_all_vector(tc.read(in)),
                             ex::when_all_vector(tc.readwrite(out)));

  return ex::make_future(
      di::transform<di::TransformDispatchType::Plain, false>(di::Policy<DefaultBackend_v<D>>(),
                                                             std::move(part_fn), std::move(sender)));
}

template <Device D>
void initColTypes(SizeType i_begin, SizeType i_split, SizeType i_end, Matrix<ColType, D>& coltypes) {
  for (SizeType i = i_begin; i <= i_end; ++i) {
    ColType val = (i <= i_split) ? ColType::UpperHalf : ColType::LowerHalf;
    setColTypeTileAsync<D>(val, coltypes.readwrite_sender(LocalTileIndex(i, 0)));
  }
}

// Assumption 1: The algorithm assumes that the arrays `d_ptr`, `z_ptr` and `c_ptr` are of equal length
// `len` and are sorted in ascending order of `d_ptr` elements with `i_ptr`.
//
// Note: parallelizing this algorithm is non-trivial because the deflation regions due to Givens
// rotations can cross over tiles and are of unknown length. However such algorithm is unlikely to
// benefit much from parallelization anyway as it is quite light on flops and it appears memory bound.
//
// Returns an array of Given's rotations used to update the colunmns of the eigenvector matrix Q
template <class T>
std::vector<GivensRotation<T>> applyDeflationToArrays(T rho, T tol, SizeType len, const SizeType* i_ptr,
                                                      T* d_ptr, T* z_ptr, ColType* c_ptr) {
  std::vector<GivensRotation<T>> rots;
  rots.reserve(to_sizet(len));

  SizeType i1 = 0;  // index of 1st element in the Givens rotation
  // Iterate over the indices of the sorted elements in pair (i1, i2) where i1 < i2 for every iteration
  for (SizeType i2 = 1; i2 < len; ++i2) {
    SizeType i1s = i_ptr[i1];
    SizeType i2s = i_ptr[i2];
    T& d1 = d_ptr[i1s];
    T& d2 = d_ptr[i2s];
    T& z1 = z_ptr[i1s];
    T& z2 = z_ptr[i2s];
    ColType& c1 = c_ptr[i1s];
    ColType& c2 = c_ptr[i2s];

    // if z1 nearly zero deflate the element and move i1 forward to i2
    if (std::abs(rho * z1) <= tol) {
      c1 = ColType::Deflated;
      i1 = i2;
      continue;
    }

    // Deflate the second element if z2 nearly zero
    if (std::abs(rho * z2) <= tol) {
      c2 = ColType::Deflated;
      continue;
    }

    // Given's deflation condition is the same as the one used in LAPACK's stedc implementation [1].
    // However, here the second entry is deflated instead of the first (z2/d2 instead of z1/d1), thus
    // `s` is not negated.
    //
    // [1] LAPACK 3.10.0, file dlaed2.f, line 393
    T r = std::sqrt(z1 * z1 + z2 * z2);
    T c = z1 / r;
    T s = z2 / r;

    // If d1 is not nearly equal to d2, move i1 forward to i2
    if (std::abs(c * s * (d2 - d1)) > tol) {
      i1 = i2;
      continue;
    }

    // When d1 is nearly equal to d2 apply Givens rotation
    z1 = r;
    z2 = 0;
    T tmp = d1 * s * s + d2 * c * c;
    d1 = d1 * c * c + d2 * s * s;
    d2 = tmp;

    rots.push_back(GivensRotation<T>{i1s, i2s, c, s});
    //  Set the the `i1` column as "Dense" if the `i2` column has opposite non-zero structure (i.e if
    //  one comes from Q1 and the other from Q2 or vice-versa)
    if ((c1 == ColType::UpperHalf && c2 == ColType::LowerHalf) ||
        (c1 == ColType::LowerHalf && c2 == ColType::UpperHalf)) {
      c1 = ColType::Dense;
    }
    c2 = ColType::Deflated;
  }

  return rots;
}

template <class T>
pika::future<std::vector<GivensRotation<T>>> applyDeflation(
    SizeType i_begin, SizeType i_end, pika::shared_future<T> rho_fut, pika::shared_future<T> tol_fut,
    Matrix<const SizeType, Device::CPU>& index, Matrix<T, Device::CPU>& d, Matrix<T, Device::CPU>& z,
    Matrix<ColType, Device::CPU>& c) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_end, index.distribution());

  auto deflate_fn = [n](auto rho, auto tol, auto index_tiles_futs, auto d_tiles, auto z_tiles,
                        auto c_tiles) {
    TileElementIndex zero_idx(0, 0);
    const SizeType* i_ptr = index_tiles_futs[0].get().ptr(zero_idx);
    T* d_ptr = d_tiles[0].ptr(zero_idx);
    T* z_ptr = z_tiles[0].ptr(zero_idx);
    ColType* c_ptr = c_tiles[0].ptr(zero_idx);
    return applyDeflationToArrays(rho, tol, n, i_ptr, d_ptr, z_ptr, c_ptr);
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(std::move(rho_fut), std::move(tol_fut), ex::when_all_vector(tc.read(index)),
                             ex::when_all_vector(tc.readwrite(d)), ex::when_all_vector(tc.readwrite(z)),
                             ex::when_all_vector(tc.readwrite(c)));

  return ex::make_future(
      di::transform<di::TransformDispatchType::Plain, false>(di::Policy<Backend::MC>(),
                                                             std::move(deflate_fn), std::move(sender)));
}

// Assumption: the memory layout of the matrix from which the tiles are coming is column major.
//
// `tiles`: The tiles of the matrix between tile indices `(i_begin, i_begin)` and `(i_end, i_end)` that
// are potentially affected by the Givens rotations. `n` : column size
//
// Note: a column index may be paired to more than one other index, this may lead to a race condition if
//       parallelized trivially. Current implementation is serial.
//
template <class T, Device D>
void applyGivensRotationsToMatrixColumns(SizeType i_begin, SizeType i_end,
                                         pika::future<std::vector<GivensRotation<T>>> rots_fut,
                                         Matrix<T, D>& mat) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_end, mat.distribution());
  SizeType nb = mat.distribution().blockSize().rows();

  auto givens_rots_fn = [n, nb](const auto& rots, const auto& tiles, [[maybe_unused]] auto&&... ts) {
    // Distribution of the merged subproblems
    matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

    for (const GivensRotation<T>& rot : rots) {
      // Get the index of the tile that has column `rot.i` and the the index of the column within the tile.
      SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, rot.i));
      SizeType i_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.i);
      T* x = tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_el));

      // Get the index of the tile that has column `rot.j` and the the index of the column within the tile.
      SizeType j_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, rot.j));
      SizeType j_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.j);
      T* y = tiles[to_sizet(j_tile)].ptr(TileElementIndex(0, j_el));

      // Apply Givens rotations
      if constexpr (D == Device::CPU) {
        blas::rot(n, x, 1, y, 1, rot.c, rot.s);
      }
      else {
        givensRotationOnDevice(n, x, y, rot.c, rot.s, ts...);
      }
    }
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(std::move(rots_fut), ex::when_all_vector(tc.readwrite(mat)));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), std::move(givens_rots_fn), std::move(sender));
}

template <class T>
void solveRank1Problem(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
                       pika::shared_future<T> rho_fut, Matrix<const T, Device::CPU>& d,
                       Matrix<const T, Device::CPU>& z, Matrix<T, Device::CPU>& evals,
                       Matrix<T, Device::CPU>& evecs) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType n = problemSize(i_begin, i_end, evals.distribution());
  SizeType nb = evals.distribution().blockSize().rows();

  auto rank1_fn = [n, nb](auto k, auto rho, auto d_tiles_futs, auto z_tiles_futs, auto eval_tiles,
                          auto evec_tiles) {
    TileElementIndex zero(0, 0);
    const T* d_ptr = d_tiles_futs[0].get().ptr(zero);
    const T* z_ptr = z_tiles_futs[0].get().ptr(zero);
    T* eval_ptr = eval_tiles[0].ptr(zero);

    matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

    std::vector<SizeType> loop_arr(to_sizet(k));
    std::iota(std::begin(loop_arr), std::end(loop_arr), 0);
    pika::for_each(pika::execution::par, std::begin(loop_arr), std::end(loop_arr), [&](SizeType i) {
      T& eigenval = eval_ptr[to_sizet(i)];

      SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, i));
      SizeType i_col = distr.tileElementFromGlobalElement<Coord::Col>(i);
      T* delta = evec_tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_col));

      lapack::laed4(static_cast<int>(k), static_cast<int>(i), d_ptr, z_ptr, delta, rho, &eigenval);
    });
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(std::move(k_fut), std::move(rho_fut), ex::when_all_vector(tc.read(d)),
                             ex::when_all_vector(tc.read(z)), ex::when_all_vector(tc.readwrite(evals)),
                             ex::when_all_vector(tc.readwrite(evecs)));

  ex::start_detached(di::transform<di::TransformDispatchType::Plain, false>(di::Policy<Backend::MC>(),
                                                                            std::move(rank1_fn),
                                                                            std::move(sender)));
}

// References:
// - lapack 3.10.0, dlaed3.f, line 293
// - LAPACK Working Notes: lawn132, Parallelizing the Divide and Conquer Algorithm for the Symmetric
//   Tridiagonal Eigenvalue Problem on Distributed Memory Architectures, 4.2 Orthogonality
template <class T, Device D>
void formEvecs(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
               Matrix<const T, D>& diag, Matrix<const T, D>& z, Matrix<T, D>& ws, Matrix<T, D>& evecs) {
  const matrix::Distribution& distr = evecs.distribution();

  // Reduce by multiplication into the first column of each tile of ws
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      divideEvecsByDiagonalAsync<D>(k_fut, i_subm_el, j_subm_el,
                                    diag.read_sender(GlobalTileIndex(i_tile, 0)),
                                    diag.read_sender(GlobalTileIndex(j_tile, 0)),
                                    evecs.read_sender(GlobalTileIndex(i_tile, j_tile)),
                                    ws.readwrite_sender(GlobalTileIndex(i_tile, j_tile)));
    }
  }

  // Reduce by multiplication into the first column of ws
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin + 1; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      multiplyFirstColumnsAsync<D>(k_fut, i_subm_el, j_subm_el,
                                   ws.read_sender(GlobalTileIndex(i_tile, j_tile)),
                                   ws.readwrite_sender(GlobalTileIndex(i_tile, i_begin)));
    }
  }

  // Use the first column of `ws` matrix to compute the eigenvectors
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      calcEvecsFromWeightVecAsync<D>(k_fut, i_subm_el, j_subm_el,
                                     z.read_sender(GlobalTileIndex(i_tile, 0)),
                                     ws.read_sender(GlobalTileIndex(i_tile, i_begin)),
                                     evecs.readwrite_sender(GlobalTileIndex(i_tile, j_tile)));
    }
  }

  // Calculate the sum of square for each column in each tile
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      GlobalTileIndex mat_idx(i_tile, j_tile);
      sumsqColsAsync<D>(k_fut, i_subm_el, j_subm_el, evecs.read_sender(mat_idx),
                        ws.readwrite_sender(mat_idx));
    }
  }

  // Sum the sum of squares into the first row of `ws` submatrix
  for (SizeType i_tile = i_begin + 1; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      addFirstRowsAsync<D>(k_fut, i_subm_el, j_subm_el, ws.read_sender(GlobalTileIndex(i_tile, j_tile)),
                           ws.readwrite_sender(GlobalTileIndex(i_begin, j_tile)));
    }
  }

  // Normalize column vectors
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      divideColsByFirstRowAsync<D>(k_fut, i_subm_el, j_subm_el,
                                   ws.read_sender(GlobalTileIndex(i_begin, j_tile)),
                                   evecs.readwrite_sender(GlobalTileIndex(i_tile, j_tile)));
    }
  }
}

template <class T, Device D>
void setUnitDiag(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
                 Matrix<T, D>& mat) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  // Iterate over diagonal tiles
  const matrix::Distribution& distr = mat.distribution();
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType tile_begin = distr.globalElementFromGlobalTileAndTileElement<Coord::Row>(i_tile, 0) -
                          distr.globalElementFromGlobalTileAndTileElement<Coord::Row>(i_begin, 0);

    auto diag_fn = [tile_begin](const auto& k, const auto& tile, [[maybe_unused]] auto&&... ts) {
      // If all elements of the tile are after the `k` index reset the offset
      SizeType tile_offset = k - tile_begin;
      if (tile_offset < 0)
        tile_offset = 0;

      // Set all diagonal elements of the tile to 1.
      if constexpr (D == Device::CPU) {
        for (SizeType i = tile_offset; i < tile.size().rows(); ++i) {
          tile(TileElementIndex(i, i)) = 1;
        }
      }
      else {
        if (tile_offset < tile.size().rows()) {
          setUnitDiagTileOnDevice(tile.size().rows() - tile_offset, tile.ld(),
                                  tile.ptr(TileElementIndex(tile_offset, tile_offset)), ts...);
        }
      }
    };

    auto sender = ex::when_all(k_fut, mat.readwrite_sender(GlobalTileIndex(i_tile, i_tile)));
    di::transformDetach(di::Policy<DefaultBackend_v<D>>(), std::move(diag_fn), std::move(sender));
  }
}

// Set submatrix to zero
template <class T>
void resetSubMatrix(SizeType i_begin, SizeType i_end, Matrix<T, Device::CPU>& mat) {
  using dlaf::internal::Policy;
  using pika::execution::thread_priority;
  using pika::execution::experimental::start_detached;

  for (SizeType j = i_begin; j <= i_end; ++j) {
    for (SizeType i = i_begin; i <= i_end; ++i) {
      start_detached(mat.readwrite_sender(GlobalTileIndex(i, j)) |
                     tile::set0(Policy<Backend::MC>(thread_priority::normal)));
    }
  }
}

template <class T, Device Source, Device Destination>
void copy(SizeType i_begin, SizeType i_end, Matrix<const T, Source>& source,
          Matrix<T, Destination>& dest) {
  if constexpr (Source == Destination) {
    if (&source == &dest)
      return;
  }

  namespace ex = pika::execution::experimental;
  const auto& distribution = source.distribution();

  bool is_col_matrix = distribution.size().cols() == 1;
  SizeType j_begin = (is_col_matrix) ? 0 : i_begin;
  SizeType j_end = (is_col_matrix) ? 0 : i_end;

  for (SizeType j = j_begin; j <= j_end; ++j) {
    for (SizeType i = i_begin; i <= i_end; ++i) {
      ex::start_detached(
          ex::when_all(source.read_sender(LocalTileIndex(i, j)),
                       dest.readwrite_sender(LocalTileIndex(i, j))) |
          dlaf::matrix::copy(
              dlaf::internal::Policy<matrix::internal::CopyBackend_v<Source, Destination>>{}));
    }
  }
}

template <Backend backend, Device device, class T>
void mergeSubproblems(SizeType i_begin, SizeType i_split, SizeType i_end, pika::shared_future<T> rho_fut,
                      WorkSpace<T, device>& ws, WorkSpaceHostMirror<T, device>& ws_h,
                      Matrix<T, device>& evals, Matrix<T, device>& evecs) {
  // Calculate the size of the upper subproblem
  SizeType n1 = problemSize(i_begin, i_split, evecs.distribution());

  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleZVec(i_begin, i_split, i_end, rho_fut, evecs, ws.z);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  rho_fut = scaleRho(std::move(rho_fut));

  // Calculate the tolerance used for deflation
  pika::shared_future<T> tol_fut = calcTolerance(i_begin, i_end, evals, ws.z);

  // Initialize the column types vector `c`
  initColTypes(i_begin, i_split, i_end, ws.c);

  // Step #1
  //
  //    i1 (out) : initial <--- initial (identity map)
  //    i2 (out) : initial <--- pre_sorted
  //
  // - deflate `d`, `z` and `c`
  // - apply Givens rotations to `Q` - `mat_ev`
  //
  initIndex(i_begin, i_end, ws.i1);
  sortIndex(i_begin, i_end, pika::make_ready_future(n1), evals, ws.i1, ws.i2);

  // copy from GPU to CPU
  copy(i_begin, i_end, ws.i2, ws_h.i2);
  copy(i_begin, i_end, ws.z, ws_h.z);
  copy(i_begin, i_end, ws.c, ws_h.c);
  copy(i_begin, i_end, evals, ws_h.evals);

  pika::future<std::vector<GivensRotation<T>>> rots_fut =
      applyDeflation(i_begin, i_end, rho_fut, tol_fut, ws_h.i2, ws_h.evals, ws_h.z, ws_h.c);

  // copy from CPU to GPU
  copy(i_begin, i_end, ws_h.z, ws.z);
  copy(i_begin, i_end, ws_h.c, ws.c);
  copy(i_begin, i_end, ws_h.evals, evals);

  applyGivensRotationsToMatrixColumns(i_begin, i_end, std::move(rots_fut), evecs);

  // Step #2
  //
  //    i2 (in)  : initial <--- pre_sorted
  //    i3 (out) : initial <--- deflated
  //
  // - reorder `d -> dtmp`, `z -> ztmp`, using `i3` such that deflated entries are at the bottom.
  // - solve the rank-1 problem and save eigenvalues in `dtmp` and eigenvectors in `mat1`.
  // - set deflated diagonal entries of `U` to 1 (temporary solution until optimized GEMM is implemented)
  //
  pika::shared_future<SizeType> k_fut =
      stablePartitionIndexForDeflation(i_begin, i_end, ws.c, ws.i2, ws.i3);
  applyIndex(i_begin, i_end, ws.i3, evals, ws.dtmp);
  applyIndex(i_begin, i_end, ws.i3, ws.z, ws.ztmp);
  copy(i_begin, i_end, ws.dtmp, evals);

  copy(i_begin, i_end, evals, ws_h.evals);
  copy(i_begin, i_end, ws.dtmp, ws_h.dtmp);
  copy(i_begin, i_end, ws.ztmp, ws_h.ztmp);
  resetSubMatrix(i_begin, i_end, ws_h.mat1);
  solveRank1Problem(i_begin, i_end, k_fut, rho_fut, ws_h.evals, ws_h.ztmp, ws_h.dtmp, ws_h.mat1);
  copy(i_begin, i_end, ws_h.mat1, ws.mat1);
  copy(i_begin, i_end, ws_h.dtmp, ws.dtmp);
  formEvecs(i_begin, i_end, k_fut, evals, ws.ztmp, ws.mat2, ws.mat1);
  setUnitDiag(i_begin, i_end, k_fut, ws.mat1);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  //    i3 (in)  : initial <--- deflated
  //    i2 (out) : initial ---> deflated
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  //
  invertIndex(i_begin, i_end, ws.i3, ws.i2);
  dlaf::permutations::permute<backend, device, T, Coord::Row>(i_begin, i_end, ws.i2, ws.mat1, ws.mat2);
  dlaf::multiplication::generalSubMatrix<backend, device, T>(i_begin, i_end, blas::Op::NoTrans,
                                                             blas::Op::NoTrans, T(1), evecs, ws.mat2,
                                                             T(0), ws.mat1);

  // Step #4: Final sorting of eigenvalues and eigenvectors
  //
  //    i1 (in)  : deflated <--- deflated  (identity map)
  //    i2 (out) : deflated <--- post_sorted
  //
  // - reorder `dtmp -> d` using the `i2` such that `d` values (eigenvalues and deflated values) are in
  //   ascending order
  // - reorder columns in `mat_ev` using `i2` such that eigenvectors match eigenvalues
  //
  sortIndex(i_begin, i_end, k_fut, ws.dtmp, ws.i1, ws.i2);
  applyIndex(i_begin, i_end, ws.i2, ws.dtmp, evals);
  dlaf::permutations::permute<backend, device, T, Coord::Col>(i_begin, i_end, ws.i2, ws.mat1, evecs);
}
}
