//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/algorithm.hpp>
#include <pika/execution.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/tridiag_solver/coltype.h"
#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/eigensolver/tridiag_solver/rot.h"
#include "dlaf/eigensolver/tridiag_solver/tile_collector.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/multiplication/general.h"
#include "dlaf/permutations/general.h"
#include "dlaf/permutations/general/impl.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix/print_csv.h"

namespace dlaf::eigensolver::internal {

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

// forward declaration : Device::GPU - unused
template <class T, Device D>
struct WorkSpaceHostMirror {
  matrix::Matrix<T, Device::CPU> evals;
  matrix::Matrix<T, Device::CPU> mat1;
  matrix::Matrix<T, Device::CPU> dtmp;
  matrix::Matrix<T, Device::CPU> z;
  matrix::Matrix<T, Device::CPU> ztmp;
  matrix::Matrix<SizeType, Device::CPU> i2;
  matrix::Matrix<ColType, Device::CPU> c;

  // only needed for the distributed tridiagonal solver
  matrix::Matrix<T, Device::CPU> evecs;
  matrix::Matrix<T, Device::CPU> mat2;

  // Test
  matrix::Matrix<T, Device::CPU> tridiag;
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

  // only needed for the distributed tridiagonal solver
  matrix::Matrix<T, Device::CPU>& evecs;
  matrix::Matrix<T, Device::CPU>& mat2;

  // Test
  matrix::Matrix<T, Device::CPU>& tridiag;
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
template <class T, Device D, class RhoSender>
void assembleZVec(SizeType i_begin, SizeType i_split, SizeType i_end, RhoSender&& rho,
                  Matrix<const T, D>& evecs, Matrix<T, D>& z) {
  // Iterate over tiles of Q1 and Q2 around the split row `i_middle`.
  for (SizeType i = i_begin; i <= i_end; ++i) {
    // True if tile is in Q1
    bool top_tile = i <= i_split;
    // Move to the row below `i_middle` for `Q2`
    SizeType evecs_row = i_split + ((top_tile) ? 0 : 1);
    GlobalTileIndex idx_evecs(evecs_row, i);
    // Take the last row of a `Q1` tile or the first row of a `Q2` tile
    GlobalTileIndex z_idx(i, 0);

    // Copy the row into the column vector `z`
    assembleRank1UpdateVectorTileAsync<T, D>(top_tile, rho, evecs.read_sender(idx_evecs),
                                             z.readwrite_sender(z_idx));
  }
}

// Multiply by factor 2 to account for the normalization of `z` vector and make sure rho > 0 f
//
template <class RhoSender>
auto scaleRho(RhoSender&& rho) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  return std::forward<RhoSender>(rho) |
         di::transform(di::Policy<Backend::MC>(), [](auto rho) { return 2 * std::abs(rho); });
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
auto calcTolerance(SizeType i_begin, SizeType i_end, Matrix<const T, D>& d, Matrix<const T, D>& z) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto dmax = maxVectorElement(i_begin, i_end, d);
  auto zmax = maxVectorElement(i_begin, i_end, z);

  auto tol_fn = [](T dmax, T zmax) {
    return 8 * std::numeric_limits<T>::epsilon() * std::max(dmax, zmax);
  };

  return ex::when_all(std::move(dmax), std::move(zmax)) |
         di::transform(di::Policy<Backend::MC>(), std::move(tol_fn)) |
         // TODO: This releases the tiles that are kept in the operation state.
         // This is a temporary fix and needs to be replaced by a different
         // adaptor or different lifetime guarantees. This is tracked in
         // https://github.com/pika-org/pika/issues/479.
         ex::ensure_started();
}

// Sorts an index `in_index_tiles` based on values in `vals_tiles` in ascending order into the index
// `out_index_tiles` where `vals_tiles` is composed of two pre-sorted ranges in ascending order that
// are merged, the first is [0, k) and the second is [k, n).
//
template <class T, Device D, class KSender>
void sortIndex(SizeType i_begin, SizeType i_end, KSender&& k, Matrix<const T, D>& vec,
               Matrix<const SizeType, D>& in_index, Matrix<SizeType, D>& out_index) {
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

  auto sender = ex::when_all(std::forward<KSender>(k), ex::when_all_vector(tc.read<T, D>(vec)),
                             ex::when_all_vector(tc.read<SizeType, D>(in_index)),
                             ex::when_all_vector(tc.readwrite<SizeType, D>(out_index)));

  ex::start_detached(
      di::transform(di::Policy<DefaultBackend_v<D>>(), std::move(sort_fn), std::move(sender)));
}

// Applies `index` to `in` to get `out`
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
      di::transform(di::Policy<DefaultBackend_v<D>>(), std::move(applyIndex_fn), std::move(sender)));
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
      di::transform(di::Policy<DefaultBackend_v<D>>(), std::move(inv_fn), std::move(sender)));
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
auto stablePartitionIndexForDeflation(SizeType i_begin, SizeType i_end, Matrix<const ColType, D>& c,
                                      Matrix<const SizeType, D>& in, Matrix<SizeType, D>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  constexpr auto backend = dlaf::DefaultBackend_v<D>;

  SizeType n = problemSize(i_begin, i_end, in.distribution());
  if constexpr (D == Device::CPU) {
    auto part_fn = [n](const auto& c_tiles_futs, const auto& in_tiles_futs, const auto& out_tiles) {
      TileElementIndex zero_idx(0, 0);
      const ColType* c_ptr = c_tiles_futs[0].get().ptr(zero_idx);
      const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero_idx);
      SizeType* out_ptr = out_tiles[0].ptr(zero_idx);

      return stablePartitionIndexForDeflationArrays(n, c_ptr, in_ptr, out_ptr);
    };

    TileCollector tc{i_begin, i_end};
    return ex::when_all(ex::when_all_vector(tc.read(c)), ex::when_all_vector(tc.read(in)),
                        ex::when_all_vector(tc.readwrite(out))) |
           di::transform(di::Policy<backend>(), std::move(part_fn));
  }
  else {
#ifdef DLAF_WITH_GPU
    auto part_fn = [n](const auto& c_tiles_futs, const auto& in_tiles_futs, const auto& out_tiles,
                       auto& host_k, auto& device_k) {
      TileElementIndex zero_idx(0, 0);
      const ColType* c_ptr = c_tiles_futs[0].get().ptr(zero_idx);
      const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero_idx);
      SizeType* out_ptr = out_tiles[0].ptr(zero_idx);

      return ex::just(n, c_ptr, in_ptr, out_ptr, host_k(), device_k()) |
             di::transform(di::Policy<backend>(), stablePartitionIndexOnDevice) |
             ex::then([&host_k]() { return *host_k(); });
    };

    TileCollector tc{i_begin, i_end};
    return ex::when_all(ex::when_all_vector(tc.read(c)), ex::when_all_vector(tc.read(in)),
                        ex::when_all_vector(tc.readwrite(out)),
                        ex::just(memory::MemoryChunk<SizeType, Device::CPU>{1},
                                 memory::MemoryChunk<SizeType, Device::GPU>{1})) |
           ex::let_value(std::move(part_fn));
#endif
  }
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

template <class T, class RhoSender, class TolSender>
auto applyDeflation(SizeType i_begin, SizeType i_end, RhoSender&& rho, TolSender&& tol,
                    Matrix<const SizeType, Device::CPU>& index, Matrix<T, Device::CPU>& d,
                    Matrix<T, Device::CPU>& z, Matrix<ColType, Device::CPU>& c) {
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

  auto sender = ex::when_all(std::forward<RhoSender>(rho), std::forward<TolSender>(tol),
                             ex::when_all_vector(tc.read(index)), ex::when_all_vector(tc.readwrite(d)),
                             ex::when_all_vector(tc.readwrite(z)), ex::when_all_vector(tc.readwrite(c)));

  return di::transform(di::Policy<Backend::MC>(), std::move(deflate_fn), std::move(sender)) |
         // TODO: This releases the tiles that are kept in the operation state.
         // This is a temporary fix and needs to be replaced by a different
         // adaptor or different lifetime guarantees. This is tracked in
         // https://github.com/pika-org/pika/issues/479.
         ex::ensure_started();
}

// Apply GivenRotations to tiles of the square sub-matrix identified by tile in range [i_begin, i_last].
//
// @param i_begin global tile index for both row and column identifying the start of the sub-matrix
// @param i_last global tile index for both row and column identifying the end of the sub-matrix (inclusive)
// @param rots GivenRotations to apply (element column indices of rotations are relative to the sub-matrix)
// @param mat matrix where the sub-matrix is located
//
// @pre mat is not distributed
// @pre memory layout of @p mat is column major.
template <class T, Device D, class RotsSender>
void applyGivensRotationsToMatrixColumns(SizeType i_begin, SizeType i_end, RotsSender&& rots,
                                         Matrix<T, D>& mat) {
  // Note:
  // a column index may be paired to more than one other index, this may lead to a race
  // condition if parallelized trivially. Current implementation is serial.

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

  auto sender = ex::when_all(std::forward<RotsSender>(rots), ex::when_all_vector(tc.readwrite(mat)));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), std::move(givens_rots_fn), std::move(sender));
}

template <class T, class KSender, class RhoSender>
void solveRank1Problem(SizeType i_begin, SizeType i_end, KSender&& k, RhoSender&& rho,
                       Matrix<const T, Device::CPU>& d, Matrix<const T, Device::CPU>& z,
                       Matrix<T, Device::CPU>& evals, Matrix<T, Device::CPU>& evecs) {
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

  auto sender =
      ex::when_all(std::forward<KSender>(k), std::forward<RhoSender>(rho),
                   ex::when_all_vector(tc.read(d)), ex::when_all_vector(tc.read(z)),
                   ex::when_all_vector(tc.readwrite(evals)), ex::when_all_vector(tc.readwrite(evecs)));

  ex::start_detached(di::transform(di::Policy<Backend::MC>(), std::move(rank1_fn), std::move(sender)));
}

// Initializes a weight vector in the first local column of the local or distributed workspace matrix @p `ws`.
//
// @p diag local matrix of size (n x 1)
// @p evecs local or distributed matrix of size (n x n)
// @p ws local or distributed matrix of size (n x n)
//
// Assumption: @p evecs and @p ws have equal distributions
//
// References:
// - lapack 3.10.0, dlaed3.f, line 293
// - LAPACK Working Notes: lawn132, Parallelizing the Divide and Conquer Algorithm for the Symmetric
//   Tridiagonal Eigenvalue Problem on Distributed Memory Architectures, 4.2 Orthogonality
template <class T, Device D, class KSender>
void initWeightVector(GlobalTileIndex idx_gl_begin, LocalTileIndex idx_loc_begin,
                      LocalTileSize sz_loc_tiles, KSender&& k, Matrix<const T, D>& diag,
                      Matrix<const T, D>& evecs, Matrix<T, D>& ws) {
  const matrix::Distribution& dist = evecs.distribution();

  // Reduce by multiplication into the first local column of each tile of the workspace matrix `ws`
  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    // Divide the eigenvectors of the rank1 update problem `evecs` by it's diagonal matrix `diag` and
    // reduce multiply into the first column of each tile of the workspace matrix `ws`
    divideEvecsByDiagonalAsync<D>(k, sz_gl_el.rows(), sz_gl_el.cols(),
                                  diag.read_sender(GlobalTileIndex(idx_gl_tile.row(), 0)),
                                  diag.read_sender(GlobalTileIndex(idx_gl_tile.col(), 0)),
                                  evecs.read_sender(idx_loc_tile), ws.readwrite_sender(idx_loc_tile));

    // skip the first local column
    if (idx_loc_tile.col() == idx_loc_begin.col())
      continue;

    // reduce-multiply the first column of each local tile of the workspace matrix into the first local
    // column of the matrix
    LocalTileIndex idx_ws_first_col_tile(idx_loc_tile.row(), idx_loc_begin.col());
    multiplyFirstColumnsAsync<D>(k, sz_gl_el.rows(), sz_gl_el.cols(), ws.read_sender(idx_loc_tile),
                                 ws.readwrite_sender(idx_ws_first_col_tile));
  }
}

// References:
// - lapack 3.10.0, dlaed3.f, line 293
// - LAPACK Working Notes: lawn132, Parallelizing the Divide and Conquer Algorithm for the Symmetric
//   Tridiagonal Eigenvalue Problem on Distributed Memory Architectures, 4.2 Orthogonality
template <class T, Device D, class KSender>
void formEvecsUsingWeightVec(GlobalTileIndex idx_gl_begin, LocalTileIndex idx_loc_begin,
                             LocalTileSize sz_loc_tiles, KSender&& k, Matrix<const T, D>& z,
                             Matrix<const T, D>& ws, Matrix<T, D>& evecs) {
  const matrix::Distribution& dist = evecs.distribution();

  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    LocalTileIndex idx_ws_first_local_column(idx_loc_tile.row(), idx_loc_begin.col());

    calcEvecsFromWeightVecAsync<D>(k, sz_gl_el.rows(), sz_gl_el.cols(),
                                   z.read_sender(GlobalTileIndex(idx_gl_tile.row(), 0)),
                                   ws.read_sender(idx_ws_first_local_column),
                                   evecs.readwrite_sender(idx_loc_tile));
  }
}

// Sum of squares of columns of @p evecs into the first row of @p ws
template <class T, Device D, class KSender>
void sumsqEvecs(GlobalTileIndex idx_gl_begin, LocalTileIndex idx_loc_begin, LocalTileSize sz_loc_tiles,
                KSender&& k, Matrix<const T, D>& evecs, Matrix<T, D>& ws) {
  const matrix::Distribution& dist = evecs.distribution();

  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    sumsqColsAsync<D>(k, sz_gl_el.rows(), sz_gl_el.cols(), evecs.read_sender(idx_loc_tile),
                      ws.readwrite_sender(idx_loc_tile));

    // skip the first local row
    if (idx_loc_tile.row() == idx_loc_begin.row())
      continue;

    LocalTileIndex idx_ws_first_row_tile(idx_loc_begin.row(), idx_loc_tile.col());
    addFirstRowsAsync<D>(k, sz_gl_el.rows(), sz_gl_el.cols(), ws.read_sender(idx_loc_tile),
                         ws.readwrite_sender(idx_ws_first_row_tile));
  }
}

// Normalize column vectors
template <class T, Device D, class KSender>
void normalizeEvecs(GlobalTileIndex idx_gl_begin, LocalTileIndex idx_loc_begin,
                    LocalTileSize sz_loc_tiles, KSender&& k, Matrix<const T, D>& ws,
                    Matrix<T, D>& evecs) {
  const matrix::Distribution& dist = evecs.distribution();

  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    LocalTileIndex idx_ws_first_local_row(idx_loc_begin.row(), idx_loc_tile.col());
    divideColsByFirstRowAsync<D>(k, sz_gl_el.rows(), sz_gl_el.cols(),
                                 ws.read_sender(idx_ws_first_local_row),
                                 evecs.readwrite_sender(idx_loc_tile));
  }
}

template <class T, Device D, class KSender>
void setUnitDiag(SizeType i_begin, SizeType i_end, KSender&& k, Matrix<T, D>& mat) {
  // Iterate over diagonal tiles
  const matrix::Distribution& distr = mat.distribution();
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType tile_begin = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);

    setUnitDiagonalAsync<D>(k, tile_begin, mat.readwrite_sender(GlobalTileIndex(i_tile, i_tile)));
  }
}

template <Backend backend, Device device, class T, class RhoSender>
void mergeSubproblems(SizeType i_begin, SizeType i_split, SizeType i_end, RhoSender&& rho,
                      WorkSpace<T, device>& ws, WorkSpaceHostMirror<T, device>& ws_h,
                      Matrix<T, device>& evals, Matrix<T, device>& evecs) {
  namespace ex = pika::execution::experimental;

  GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  LocalTileIndex idx_loc_begin(i_begin, i_begin);
  SizeType nrtiles = i_end - i_begin + 1;
  LocalTileSize sz_loc_tiles(nrtiles, nrtiles);

  LocalTileIndex idx_begin_tiles_vec(i_begin, 0);
  LocalTileSize sz_tiles_vec(nrtiles, 1);

  // Calculate the size of the upper subproblem
  SizeType n1 = problemSize(i_begin, i_split, evecs.distribution());

  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleZVec(i_begin, i_split, i_end, rho, evecs, ws.z);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  auto scaled_rho = scaleRho(std::move(rho)) | ex::split();

  // Calculate the tolerance used for deflation
  auto tol = calcTolerance(i_begin, i_end, evals, ws.z);

  // Initialize the column types vector `c`
  initColTypes(i_begin, i_split, i_end, ws.c);

  // Step #1
  //
  //    i1 (out) : initial <--- initial (identity map)
  //    i2 (out) : initial <--- pre_sorted
  //
  // - deflate `d`, `z` and `c`
  // - apply Givens rotations to `Q` - `evecs`
  //
  initIndex(i_begin, i_end, ws.i1);
  sortIndex(i_begin, i_end, ex::just(n1), evals, ws.i1, ws.i2);

  // --- copy from GPU to CPU if on GPU
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.i2, ws_h.i2);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.z, ws_h.z);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.c, ws_h.c);
  copy(idx_begin_tiles_vec, sz_tiles_vec, evals, ws_h.evals);

  auto rots =
      applyDeflation(i_begin, i_end, scaled_rho, std::move(tol), ws_h.i2, ws_h.evals, ws_h.z, ws_h.c);

  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.z, ws.z);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.c, ws.c);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.evals, evals);
  // ---

  applyGivensRotationsToMatrixColumns(i_begin, i_end, std::move(rots), evecs);

  // Step #2
  //
  //    i2 (in)  : initial <--- pre_sorted
  //    i3 (out) : initial <--- deflated
  //
  // - reorder `d -> dtmp`, `z -> ztmp`, using `i3` such that deflated entries are at the bottom.
  // - solve the rank-1 problem and save eigenvalues in `dtmp` and eigenvectors in `mat1`.
  // - set deflated diagonal entries of `U` to 1 (temporary solution until optimized GEMM is implemented)
  //
  auto k = stablePartitionIndexForDeflation(i_begin, i_end, ws.c, ws.i2, ws.i3) | ex::split();
  applyIndex(i_begin, i_end, ws.i3, evals, ws.dtmp);
  applyIndex(i_begin, i_end, ws.i3, ws.z, ws.ztmp);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.dtmp, evals);

  // --- copy from GPU to CPU if on GPU
  copy(idx_begin_tiles_vec, sz_tiles_vec, evals, ws_h.evals);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.dtmp, ws_h.dtmp);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.ztmp, ws_h.ztmp);

  matrix::util::set0<Backend::MC>(pika::execution::thread_priority::normal, idx_loc_begin, sz_loc_tiles,
                                  ws_h.mat1);
  solveRank1Problem(i_begin, i_end, k, scaled_rho, ws_h.evals, ws_h.ztmp, ws_h.dtmp, ws_h.mat1);

  copy(idx_loc_begin, sz_loc_tiles, ws_h.mat1, ws.mat1);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.dtmp, ws.dtmp);
  // ---

  // formEvecs(i_begin, i_end, k, evals, ws.ztmp, ws.mat2, ws.mat1);
  initWeightVector(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, evals, ws.mat1, ws.mat2);
  formEvecsUsingWeightVec(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, ws.ztmp, ws.mat2, ws.mat1);
  sumsqEvecs(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, ws.mat1, ws.mat2);
  normalizeEvecs(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, ws.mat2, ws.mat1);
  setUnitDiag(i_begin, i_end, k, ws.mat1);

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
  // - reorder columns in `evecs` using `i2` such that eigenvectors match eigenvalues
  //
  sortIndex(i_begin, i_end, std::move(k), ws.dtmp, ws.i1, ws.i2);
  applyIndex(i_begin, i_end, ws.i2, ws.dtmp, evals);
  dlaf::permutations::permute<backend, device, T, Coord::Col>(i_begin, i_end, ws.i2, ws.mat1, evecs);
}

// The bottom row of Q1 and the top row of Q2. The bottom row of Q1 is negated if `rho < 0`.
//
// Note that the norm of `z` is sqrt(2) because it is a concatination of two normalized vectors. Hence
// to normalize `z` we have to divide by sqrt(2).
template <class T, Device D, class RhoSender>
void assembleDistZVec(comm::CommunicatorGrid grid, common::Pipeline<comm::Communicator>& full_task_chain,
                      SizeType i_begin, SizeType i_split, SizeType i_end, RhoSender&& rho,
                      Matrix<const T, D>& evecs, Matrix<T, D>& z) {
  namespace ex = pika::execution::experimental;

  const matrix::Distribution& dist = evecs.distribution();
  comm::Index2D this_rank = dist.rankIndex();

  // Iterate over tiles of Q1 and Q2 around the split row `i_middle`.
  for (SizeType i = i_begin; i <= i_end; ++i) {
    // True if tile is in Q1
    bool top_tile = i <= i_split;
    // Move to the row below `i_middle` for `Q2`
    SizeType evecs_row = i_split + ((top_tile) ? 0 : 1);
    GlobalTileIndex idx_evecs(evecs_row, i);
    GlobalTileIndex z_idx(i, 0);

    // Copy the last row of a `Q1` tile or the first row of a `Q2` tile into a column vector `z` tile
    comm::Index2D evecs_tile_rank = dist.rankGlobalTile(idx_evecs);
    if (evecs_tile_rank == this_rank) {
      // Copy the row into the column vector `z`
      assembleRank1UpdateVectorTileAsync<T, D>(top_tile, rho, evecs.read_sender(idx_evecs),
                                               z.readwrite_sender(z_idx));
      ex::start_detached(comm::scheduleSendBcast(ex::make_unique_any_sender(full_task_chain()),
                                                 ex::make_unique_any_sender(z.read_sender(z_idx))));
    }
    else {
      comm::IndexT_MPI root_rank = grid.rankFullCommunicator(evecs_tile_rank);
      ex::start_detached(comm::scheduleRecvBcast(ex::make_unique_any_sender(full_task_chain()),
                                                 root_rank,
                                                 ex::make_unique_any_sender(z.readwrite_sender(z_idx))));
    }
  }
}

template <class T, Device D, class KSender>
void setUnitDiagDist(SizeType i_begin, SizeType i_end, KSender&& k, Matrix<T, D>& mat) {
  // Iterate over diagonal tiles
  const matrix::Distribution& dist = mat.distribution();
  comm::Index2D this_rank = dist.rankIndex();
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    GlobalTileIndex ii_tile(i_tile, i_tile);
    comm::Index2D diag_tile_rank = dist.rankGlobalTile(ii_tile);
    if (diag_tile_rank == this_rank) {
      SizeType tile_begin = dist.globalTileElementDistance<Coord::Row>(i_begin, i_tile);

      setUnitDiagonalAsync<D>(k, tile_begin, mat.readwrite_sender(ii_tile));
    }
  }
}

// (All)Reduce-multiply row-wise the first local columns of the distributed matrix @p mat of size (n, n).
// The local column on each rank is first offloaded to a local communication buffer @p comm_vec of size (n, 1).
template <class T, Device D, class KSender>
void reduceMultiplyWeightVector(common::Pipeline<comm::Communicator>& row_task_chain, SizeType i_begin,
                                SizeType i_end, LocalTileIndex idx_loc_begin, LocalTileSize sz_loc_tiles,
                                KSender&& k, Matrix<T, D>& mat, Matrix<T, D>& comm_vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const matrix::Distribution& dist = mat.distribution();

  // If the rank doesn't have local matrix tiles, participate in the AllReduce() call by sending tiles
  // from the communication buffer filled with `1`.
  if (sz_loc_tiles.isEmpty()) {
    comm::Index2D this_rank = dist.rankIndex();
    for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
      if (this_rank.row() == dist.rankGlobalTile<Coord::Row>(i_tile)) {
        GlobalTileIndex idx_gl_comm(i_tile, 0);
        auto laset_sender =
            di::whenAllLift(blas::Uplo::General, T(1), T(1), comm_vec.readwrite_sender(idx_gl_comm));
        ex::start_detached(tile::laset(di::Policy<DefaultBackend_v<D>>(), std::move(laset_sender)));
        ex::start_detached(comm::scheduleAllReduceInPlace(ex::make_unique_any_sender(row_task_chain()),
                                                          MPI_PROD,
                                                          ex::make_unique_any_sender(
                                                              comm_vec.readwrite_sender(idx_gl_comm))));
      }
    }
    return;
  }

  GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  LocalTileSize sz_loc_first_column(sz_loc_tiles.rows(), 1);
  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_first_column)) {
    GlobalTileIndex idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    GlobalElementSize sz_subm = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    GlobalTileIndex idx_gl_comm(idx_gl_tile.row(), 0);

    // set buffer to 1
    ex::start_detached(
        di::whenAllLift(blas::Uplo::General, T(1), T(1), comm_vec.readwrite_sender(idx_gl_comm)) |
        tile::laset(di::Policy<DefaultBackend_v<D>>()));

    // copy the first column of the matrix tile into the column tile of the buffer
    copy1DAsync<D>(k, sz_subm.rows(), sz_subm.cols(), Coord::Col, mat.read_sender(idx_loc_tile),
                   Coord::Col, comm_vec.readwrite_sender(idx_gl_comm));

    ex::start_detached(comm::scheduleAllReduceInPlace(ex::make_unique_any_sender(row_task_chain()),
                                                      MPI_PROD,
                                                      ex::make_unique_any_sender(
                                                          comm_vec.readwrite_sender(idx_gl_comm))));

    // copy the column tile of the buffer into the first column of the matrix tile
    copy1DAsync<D>(k, sz_subm.rows(), sz_subm.cols(), Coord::Col, comm_vec.read_sender(idx_gl_comm),
                   Coord::Col, mat.readwrite_sender(idx_loc_tile));
  }
}

// (All)Reduce-sum column-wise the first local rows of the distributed matrix @p mat of size (n, n).
// The local row on each rank is first offloaded to a local communication buffer @p comm_vec of size (n, 1).
template <class T, Device D, class KSender>
void reduceSumScalingVector(common::Pipeline<comm::Communicator>& col_task_chain, SizeType i_begin,
                            SizeType i_end, LocalTileIndex idx_loc_begin, LocalTileSize sz_loc_tiles,
                            KSender&& k, Matrix<T, D>& mat, Matrix<T, D>& comm_vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const matrix::Distribution& dist = mat.distribution();

  // If the rank doesn't have local matrix tiles, participate in the AllReduce() call by sending tiles
  // from the communication buffer filled with `0`.
  if (sz_loc_tiles.isEmpty()) {
    comm::Index2D this_rank = dist.rankIndex();
    for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
      if (this_rank.col() == dist.rankGlobalTile<Coord::Col>(i_tile)) {
        GlobalTileIndex idx_gl_comm(i_tile, 0);
        ex::start_detached(
            tile::set0(di::Policy<DefaultBackend_v<D>>(), comm_vec.readwrite_sender(idx_gl_comm)));
        ex::start_detached(comm::scheduleAllReduceInPlace(ex::make_unique_any_sender(col_task_chain()),
                                                          MPI_SUM,
                                                          ex::make_unique_any_sender(
                                                              comm_vec.readwrite_sender(idx_gl_comm))));
      }
    }
    return;
  }

  GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  LocalTileSize sz_first_local_row(1, sz_loc_tiles.cols());
  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_first_local_row)) {
    GlobalTileIndex idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    GlobalElementSize sz_subm = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    GlobalTileIndex idx_gl_comm(idx_gl_tile.col(), 0);

    // set buffer to zero
    ex::start_detached(comm_vec.readwrite_sender(idx_gl_comm) |
                       tile::set0(di::Policy<DefaultBackend_v<D>>()));

    // copy the first row of the matrix tile into the column tile of the buffer
    copy1DAsync<D>(k, sz_subm.rows(), sz_subm.cols(), Coord::Row, mat.read_sender(idx_loc_tile),
                   Coord::Col, comm_vec.readwrite_sender(idx_gl_comm));

    ex::start_detached(comm::scheduleAllReduceInPlace(ex::make_unique_any_sender(col_task_chain()),
                                                      MPI_SUM,
                                                      ex::make_unique_any_sender(
                                                          comm_vec.readwrite_sender(idx_gl_comm))));

    // copy the column tile of the buffer into the first column of the matrix tile
    copy1DAsync<D>(k, sz_subm.rows(), sz_subm.cols(), Coord::Col, comm_vec.read_sender(idx_gl_comm),
                   Coord::Row, mat.readwrite_sender(idx_loc_tile));
  }
}

template <class T, class KSender, class RhoSender>
void solveRank1ProblemDist(SizeType i_begin, SizeType i_end, LocalTileIndex idx_loc_begin,
                           LocalTileSize sz_loc_tiles, KSender&& k, RhoSender&& rho,
                           Matrix<const T, Device::CPU>& d, Matrix<const T, Device::CPU>& z,
                           Matrix<T, Device::CPU>& evals, Matrix<T, Device::CPU>& delta,
                           Matrix<T, Device::CPU>& evecs) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const matrix::Distribution& dist = evecs.distribution();
  auto rank1_fn = [i_begin, idx_loc_begin, sz_loc_tiles,
                   dist](const auto& k, const auto& rho, const auto& d_sfut_tile_arr,
                         const auto& z_sfut_tile_arr, const auto& eval_tiles, const auto& delta_tile_arr,
                         const auto& evec_tile_arr) {
    const T* d_ptr = d_sfut_tile_arr[0].get().ptr();
    const T* z_ptr = z_sfut_tile_arr[0].get().ptr();
    T* eval_ptr = eval_tiles[0].ptr();
    T* delta_ptr = delta_tile_arr[0].ptr();

    // Iterate over the columns of the local submatrix tile grid
    for (SizeType j_loc_subm_tile = 0; j_loc_subm_tile < sz_loc_tiles.cols(); ++j_loc_subm_tile) {
      // The tile column in the local matrix tile grid
      SizeType j_loc_tile = idx_loc_begin.col() + j_loc_subm_tile;
      // The tile column in the global matrix tile grid
      SizeType j_gl_tile = dist.globalTileFromLocalTile<Coord::Col>(j_loc_tile);
      // The element distance between the current tile column and the initial tile column in the global
      // matrix tile grid
      SizeType j_gl_subm_el = dist.globalTileElementDistance<Coord::Col>(i_begin, j_gl_tile);

      // Skip columns that are in the deflation zone
      if (j_gl_subm_el >= k)
        return;

      // Iterate over the elements of the column tile
      SizeType ncols = std::min(dist.tileSize<Coord::Col>(j_gl_tile), k - j_gl_subm_el);
      for (SizeType j_tile_el = 0; j_tile_el < ncols; ++j_tile_el) {
        // The global submatrix column
        SizeType j_gl_el = j_gl_subm_el + j_tile_el;

        // Solve the deflated rank-1 problem
        T& eigenval = eval_ptr[to_sizet(j_gl_el)];
        lapack::laed4(static_cast<int>(k), static_cast<int>(j_gl_el), d_ptr, z_ptr, delta_ptr, rho,
                      &eigenval);

        // Iterate over the rows of the local submatrix tile grid and copy the parts from delta stored on this rank.
        for (SizeType i_loc_subm_tile = 0; i_loc_subm_tile < sz_loc_tiles.rows(); ++i_loc_subm_tile) {
          SizeType i_subm_evec_arr = i_loc_subm_tile + j_loc_subm_tile * sz_loc_tiles.rows();
          auto& evec_tile = evec_tile_arr[to_sizet(i_subm_evec_arr)];
          // The tile row in the local matrix tile grid
          SizeType i_loc_tile = idx_loc_begin.row() + i_loc_subm_tile;
          // The tile row in the global matrix tile grid
          SizeType i_gl_tile = dist.globalTileFromLocalTile<Coord::Row>(i_loc_tile);
          // The element distance between the current tile row and the initial tile row in the
          // global matrix tile grid
          SizeType i_gl_subm_el = dist.globalTileElementDistance<Coord::Row>(i_begin, i_gl_tile);

          // Skip rows that are in the deflation zone
          if (i_gl_subm_el >= k)
            break;

          // The tile row in the global submatrix tile grid
          SizeType i_subm_delta_arr = dist.globalTileFromLocalTile<Coord::Row>(i_loc_tile) - i_begin;
          auto& delta_tile = delta_tile_arr[to_sizet(i_subm_delta_arr)];

          // Copy from delta buffer into local matrix tile
          SizeType nrows = std::min(dist.tileSize<Coord::Row>(i_gl_tile), k - i_gl_subm_el);
          tile::lacpy(TileElementSize(nrows, 1), TileElementIndex(0, 0), delta_tile,
                      TileElementIndex(0, j_tile_el), evec_tile);
        }
      }
    }
  };

  TileCollector tc{i_begin, i_end};

  auto sender =
      ex::when_all(std::forward<KSender>(k), std::forward<RhoSender>(rho),
                   ex::when_all_vector(tc.read(d)), ex::when_all_vector(tc.read(z)),
                   ex::when_all_vector(tc.readwrite(evals)), ex::when_all_vector(tc.readwrite(delta)),
                   ex::when_all_vector(tc.readwrite(evecs)));

  ex::start_detached(di::transform(di::Policy<Backend::MC>(), std::move(rank1_fn), std::move(sender)));
}

// Assembles the local matrix of eigenvalues @p evals with size (n, 1) by communication tiles row-wise.
template <class T, Device D>
void assembleDistEvalsVec(common::Pipeline<comm::Communicator>& row_task_chain, SizeType i_begin,
                          SizeType i_end, const matrix::Distribution& dist_evecs, Matrix<T, D>& evals) {
  namespace ex = pika::execution::experimental;

  comm::Index2D this_rank = dist_evecs.rankIndex();
  for (SizeType i = i_begin; i <= i_end; ++i) {
    GlobalTileIndex evals_idx(i, 0);

    comm::IndexT_MPI evecs_tile_rank = dist_evecs.rankGlobalTile<Coord::Col>(i);
    if (evecs_tile_rank == this_rank.col()) {
      ex::start_detached(
          comm::scheduleSendBcast(ex::make_unique_any_sender(row_task_chain()),
                                  ex::make_unique_any_sender(evals.read_sender(evals_idx))));
    }
    else {
      ex::start_detached(
          comm::scheduleRecvBcast(ex::make_unique_any_sender(row_task_chain()), evecs_tile_rank,
                                  ex::make_unique_any_sender(evals.readwrite_sender(evals_idx))));
    }
  }
}

// Distributed version of the tridiagonal solver on CPUs
template <Backend B, class T, Device D, class RhoSender>
void mergeDistSubproblems(comm::CommunicatorGrid grid,
                          common::Pipeline<comm::Communicator>& full_task_chain,
                          common::Pipeline<comm::Communicator>& row_task_chain,
                          common::Pipeline<comm::Communicator>& col_task_chain, SizeType i_begin,
                          SizeType i_split, SizeType i_end, RhoSender&& rho, WorkSpace<T, D>& ws,
                          WorkSpaceHostMirror<T, D>& ws_h, Matrix<T, D>& evals, Matrix<T, D>& evecs) {
  namespace ex = pika::execution::experimental;

  const matrix::Distribution& dist_evecs = evecs.distribution();

  // Calculate the size of the upper subproblem
  SizeType n1 = dist_evecs.globalTileElementDistance<Coord::Row>(i_begin, i_split + 1);

  // The local size of the subproblem
  GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  LocalTileIndex idx_loc_begin{dist_evecs.nextLocalTileFromGlobalTile<Coord::Row>(i_begin),
                               dist_evecs.nextLocalTileFromGlobalTile<Coord::Col>(i_begin)};
  LocalTileIndex idx_loc_end{dist_evecs.nextLocalTileFromGlobalTile<Coord::Row>(i_end + 1) - 1,
                             dist_evecs.nextLocalTileFromGlobalTile<Coord::Col>(i_end + 1) - 1};
  LocalTileSize sz_loc_tiles{idx_loc_end.row() - idx_loc_begin.row() + 1,
                             idx_loc_end.col() - idx_loc_begin.col() + 1};
  LocalTileIndex idx_begin_tiles_vec(i_begin, 0);
  LocalTileSize sz_tiles_vec(i_end - i_begin + 1, 1);

  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleDistZVec(grid, full_task_chain, i_begin, i_split, i_end, rho, evecs, ws.z);

  // Calculate the tolerance used for deflation
  auto tol = calcTolerance(i_begin, i_end, evals, ws.z);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  auto scaled_rho = scaleRho(std::move(rho)) | ex::split();

  // Initialize the column types vector `c`
  initColTypes(i_begin, i_split, i_end, ws.c);

  // Step #1
  //
  //    i1 (out) : initial <--- initial (identity map)
  //    i2 (out) : initial <--- pre_sorted
  //
  // - deflate `d`, `z` and `c`
  // - apply Givens rotations to `Q` - `evecs`
  //
  initIndex(i_begin, i_end, ws.i1);
  sortIndex(i_begin, i_end, ex::just(n1), evals, ws.i1, ws.i2);

  // --- copy from GPU to CPU if on GPU
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.i2, ws_h.i2);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.z, ws_h.z);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.c, ws_h.c);
  copy(idx_begin_tiles_vec, sz_tiles_vec, evals, ws_h.evals);

  auto rots =
      applyDeflation(i_begin, i_end, scaled_rho, std::move(tol), ws_h.i2, ws_h.evals, ws_h.z, ws_h.c);

  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.z, ws.z);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.c, ws.c);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.evals, evals);
  // ---

  // Make sure Isend/Irecv messages don't match between calls by providing a unique `tag`
  //
  // Note: this is unique because i_[begin|split|end] < nrtiles
  SizeType nrtiles = dist_evecs.nrTiles().rows();
  comm::IndexT_MPI tag = to_int(i_begin + i_split * nrtiles + i_end * nrtiles * nrtiles);
  applyGivensRotationsToMatrixColumns(grid.rowCommunicator(), tag, i_begin, i_end, std::move(rots),
                                      evecs);

  // Step #2
  //
  //    i2 (in)  : initial <--- pre_sorted
  //    i3 (out) : initial <--- deflated
  //
  // - reorder `d -> dtmp`, `z -> ztmp`, using `i3` such that deflated entries are at the bottom.
  // - solve the rank-1 problem and save eigenvalues in `dtmp` and eigenvectors in `mat1`.
  // - set deflated diagonal entries of `U` to 1 (temporary solution until optimized GEMM is implemented)
  //
  auto k = stablePartitionIndexForDeflation(i_begin, i_end, ws.c, ws.i2, ws.i3) | ex::split();
  applyIndex(i_begin, i_end, ws.i3, evals, ws.dtmp);
  applyIndex(i_begin, i_end, ws.i3, ws.z, ws.ztmp);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.dtmp, evals);

  // --- copy from GPU to CPU if on GPU
  copy(idx_begin_tiles_vec, sz_tiles_vec, evals, ws_h.evals);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.dtmp, ws_h.dtmp);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.ztmp, ws_h.ztmp);

  // Note: here ws_h.z is used as a contiguous buffer for the laed4 call
  matrix::util::set0<Backend::MC>(pika::execution::thread_priority::normal, idx_loc_begin, sz_loc_tiles,
                                  ws_h.mat1);
  solveRank1ProblemDist(i_begin, i_end, idx_loc_begin, sz_loc_tiles, k, std::move(scaled_rho),
                        ws_h.evals, ws_h.ztmp, ws_h.dtmp, ws_h.z, ws_h.mat1);

  copy(idx_loc_begin, sz_loc_tiles, ws_h.mat1, ws.mat1);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_h.dtmp, ws.dtmp);
  // ---

  assembleDistEvalsVec(row_task_chain, i_begin, i_end, dist_evecs, ws.dtmp);

  // Eigenvector formation: `ws.mat1` stores the eigenvectors, `ws.mat2` is used as an additional workspace
  initWeightVector(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, evals, ws.mat1, ws.mat2);
  reduceMultiplyWeightVector(row_task_chain, i_begin, i_end, idx_loc_begin, sz_loc_tiles, k, ws.mat2,
                             ws.z);
  formEvecsUsingWeightVec(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, ws.ztmp, ws.mat2, ws.mat1);
  sumsqEvecs(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, ws.mat1, ws.mat2);
  reduceSumScalingVector(col_task_chain, i_begin, i_end, idx_loc_begin, sz_loc_tiles, k, ws.mat2, ws.z);
  normalizeEvecs(idx_gl_begin, idx_loc_begin, sz_loc_tiles, k, ws.mat2, ws.mat1);
  setUnitDiagDist(i_begin, i_end, k, ws.mat1);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  //    i3 (in)  : initial <--- deflated
  //    i2 (out) : initial ---> deflated
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  //
  invertIndex(i_begin, i_end, ws.i3, ws.i2);

  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.i2, ws_h.i2);
  copy(idx_loc_begin, sz_loc_tiles, ws.mat1, ws_h.mat1);
  dlaf::permutations::permute<Backend::MC, Device::CPU, T, Coord::Row>(grid, col_task_chain, i_begin,
                                                                       i_end, ws_h.i2, ws_h.mat1,
                                                                       ws_h.mat2);
  copy(idx_loc_begin, sz_loc_tiles, ws_h.mat2, ws.mat2);

  dlaf::multiplication::generalSubMatrix<B, D, T>(grid, row_task_chain, col_task_chain, i_begin, i_end,
                                                  T(1), evecs, ws.mat2, T(0), ws.mat1);

  // Step #4: Final sorting of eigenvalues and eigenvectors
  //
  //    i1 (in)  : deflated <--- deflated  (identity map)
  //    i2 (out) : deflated <--- post_sorted
  //
  // - reorder `dtmp -> d` using the `i2` such that `d` values (eigenvalues and deflated values) are in
  //   ascending order
  // - reorder columns in `evecs` using `i2` such that eigenvectors match eigenvalues
  //
  sortIndex(i_begin, i_end, std::move(k), ws.dtmp, ws.i1, ws.i2);
  applyIndex(i_begin, i_end, ws.i2, ws.dtmp, evals);

  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.i2, ws_h.i2);
  copy(idx_loc_begin, sz_loc_tiles, ws.mat1, ws_h.mat1);
  dlaf::permutations::permute<Backend::MC, Device::CPU, T, Coord::Col>(grid, row_task_chain, i_begin,
                                                                       i_end, ws_h.i2, ws_h.mat1,
                                                                       ws_h.evecs);
  copy(idx_loc_begin, sz_loc_tiles, ws_h.evecs, evecs);
}

}
