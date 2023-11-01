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

#include <algorithm>
#include <functional>
#include <numeric>

#include <pika/barrier.hpp>
#include <pika/execution.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/kernels.h>
#include <dlaf/eigensolver/internal/get_tridiag_rank1_barrier_busy_wait.h>
#include <dlaf/eigensolver/internal/get_tridiag_rank1_nworkers.h>
#include <dlaf/eigensolver/tridiag_solver/coltype.h>
#include <dlaf/eigensolver/tridiag_solver/index_manipulation.h>
#include <dlaf/eigensolver/tridiag_solver/kernels.h>
#include <dlaf/eigensolver/tridiag_solver/rot.h>
#include <dlaf/eigensolver/tridiag_solver/tile_collector.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/multiplication/general.h>
#include <dlaf/permutations/general.h>
#include <dlaf/permutations/general/impl.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/make_sender_algorithm_overloads.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/transform.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

namespace dlaf::eigensolver::internal {

// Auxiliary matrix and vectors used for the D&C algorithm
//
// - e0: (matrix)
//     In: Holds the eigenvectors of the two subproblems (same order as d0).
//     Out: Holds the eigenvectors of the merged problem (same order as d0).
// - e1: (matrix)
//     Holds the deflated eigenvectors.
// - e2: (matrix)
//     (The original evecs matrix used as workspace) Holds the rank-1 eigenvectors.
//
// - d0: (vector)
//     In: Holds the eigenvalues of the two subproblems (in ascending order if permuted with i1).
//     Out: Holds the eigenvalues of the merged problem (in ascending order if permuted with i1).
// - d1: (vector)
//     Holds the values of the deflated diagonal sorted in ascending order
// - z0: (vector)
//     Holds the values of Cuppen's rank-1 vector
//     Also used as temporary workspace
// - z1: (vector)
//     Holds the values of the rank-1 update vector (same order as d1)
//
// - c:
//     Assigns a type to each column of Q which is used to calculate the permutation indices for Q and U
//     that bring them in matrix multiplication form.
//
// - i1, i2, i3: (vectors of indices)
//     Hold the permutation indices.
//     In: i1 contains the permutation to sort d0 of the two subproblems in ascending order.
//         If the subproblem involves a single tile the values of i1 are replaced with an identity permutation.
//     Out: i1 and i2(Device::CPU) contain the permutation to sort d0 of the merged problem in ascending
//          order.
//
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
// 3. Sort index based on updated diagonal values in ascending order. The diagonal contains eigenvalues
//    of the deflated problem and deflated entries from the initial diagonal
//
//        deflated <--- post_sorted
//
// 4. Sort index based on column types such that matrices `Q` and `U` are in matrix multiplication form.
//
//        post_sorted <--- matmul
//

template <class T, Device D>
struct WorkSpace {
  Matrix<T, D> e0;
  Matrix<T, D> e1;
  Matrix<T, D>& e2;  // Reference to reuse evecs

  Matrix<T, D>& d1;  // Reference to reuse evals

  Matrix<T, D> z0;
  Matrix<T, D> z1;

  Matrix<SizeType, D> i2;
  Matrix<SizeType, D> i5;
};

template <class T>
struct WorkSpaceHost {
  Matrix<T, Device::CPU> d0;

  Matrix<ColType, Device::CPU> c;

  Matrix<SizeType, Device::CPU> i1;
  Matrix<SizeType, Device::CPU> i3;
  Matrix<SizeType, Device::CPU> i4;
};

template <class T, Device D>
using HostMirrorMatrix =
    std::conditional_t<D == Device::CPU, Matrix<T, Device::CPU>&, Matrix<T, Device::CPU>>;

template <class T, Device D>
struct WorkSpaceHostMirror {
  HostMirrorMatrix<T, D> e2;

  HostMirrorMatrix<T, D> d1;

  HostMirrorMatrix<T, D> z0;
  HostMirrorMatrix<T, D> z1;

  HostMirrorMatrix<SizeType, D> i2;
  HostMirrorMatrix<SizeType, D> i5;
};

template <class T, Device D>
struct DistWorkSpaceHostMirror {
  HostMirrorMatrix<T, D> e0;
  HostMirrorMatrix<T, D> e2;

  HostMirrorMatrix<T, D> d1;

  HostMirrorMatrix<T, D> z0;
  HostMirrorMatrix<T, D> z1;

  HostMirrorMatrix<SizeType, D> i2;
};

template <class T>
Matrix<T, Device::CPU> initMirrorMatrix(Matrix<T, Device::GPU>& mat) {
  return Matrix<T, Device::CPU>(mat.distribution());
}

template <class T>
Matrix<T, Device::CPU>& initMirrorMatrix(Matrix<T, Device::CPU>& mat) {
  return mat;
}

// The bottom row of Q1 and the top row of Q2. The bottom row of Q1 is negated if `rho < 0`.
//
// Note that the norm of `z` is sqrt(2) because it is a concatination of two normalized vectors. Hence
// to normalize `z` we have to divide by sqrt(2).
template <class T, Device D, class RhoSender>
void assembleZVec(const SizeType i_begin, const SizeType i_split, const SizeType i_end, RhoSender&& rho,
                  Matrix<const T, D>& evecs, Matrix<T, D>& z) {
  // Iterate over tiles of Q1 and Q2 around the split row `i_split`.
  for (SizeType i = i_begin; i < i_end; ++i) {
    // True if tile is in Q1
    const bool top_tile = i < i_split;
    // Move to the row below `i_split` for `Q2`
    const SizeType evecs_row = i_split - ((top_tile) ? 1 : 0);
    const GlobalTileIndex idx_evecs(evecs_row, i);
    // Take the last row of a `Q1` tile or the first row of a `Q2` tile
    const GlobalTileIndex z_idx(i, 0);

    // Copy the row into the column vector `z`
    assembleRank1UpdateVectorTileAsync<T, D>(top_tile, rho, evecs.read(idx_evecs), z.readwrite(z_idx));
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
//
template <class T>
auto maxVectorElement(const SizeType i_begin, const SizeType i_end, Matrix<const T, Device::CPU>& vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<ex::unique_any_sender<T>> tiles_max;
  tiles_max.reserve(to_sizet(i_end - i_begin));
  for (SizeType i = i_begin; i < i_end; ++i) {
    tiles_max.push_back(di::whenAllLift(lapack::Norm::Max, vec.read(LocalTileIndex(i, 0))) |
                        di::transform(di::Policy<Backend::MC>(), tile::internal::lange_o));
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
template <class T>
auto calcTolerance(const SizeType i_begin, const SizeType i_end, Matrix<const T, Device::CPU>& d,
                   Matrix<const T, Device::CPU>& z) {
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

// This function returns number of non-deflated eigenvectors, together with a permutation @p out_ptr
// that represent mapping (sorted non-deflated | sorted deflated) -> initial.
//
// The permutation will allow to keep the mapping between sorted eigenvalues and unsorted eigenvectors,
// which is useful since eigenvectors are more expensive to permuted, so we can keep them in their initial order.
//
// @param n         number of eigenvalues
// @param c_ptr     array[n] containing the column type of each eigenvector after deflation (initial order)
// @param evals_ptr array[n] of eigenvalues sorted as in_ptr
// @param in_ptr    array[n] representing permutation current -> initial (i.e. evals[i] -> c_ptr[in_ptr[i]])
// @param out_ptr   array[n] permutation (sorted non-deflated | sorted deflated) -> initial
//
// @return k        number of non-deflated eigenvectors
template <class T>
SizeType stablePartitionIndexForDeflationArrays(const SizeType n, const ColType* c_ptr,
                                                const T* evals_ptr, const SizeType* in_ptr,
                                                SizeType* out_ptr) {
  // Get the number of non-deflated entries
  SizeType k = 0;
  for (SizeType i = 0; i < n; ++i) {
    if (c_ptr[i] != ColType::Deflated)
      ++k;
  }

  // Create the permutation (sorted non-deflated | sorted deflated) -> initial
  // Note:
  // Since during deflation, eigenvalues related to deflated eigenvectors, might not be sorted anymore,
  // this step also take care of sorting eigenvalues (actually just their related index) by their ascending value.
  SizeType i1 = 0;  // index of non-deflated values in out
  SizeType i2 = k;  // index of deflated values
  for (SizeType i = 0; i < n; ++i) {
    const SizeType ii = in_ptr[i];

    // non-deflated are untouched, just squeeze them at the beginning as they appear
    if (c_ptr[ii] != ColType::Deflated) {
      out_ptr[i1] = ii;
      ++i1;
    }
    // deflated are the ones that can have been moved "out-of-order" by deflation...
    // ... so each time insert it in the right place based on eigenvalue value
    else {
      const T a = evals_ptr[ii];

      SizeType j = i2;
      // shift to right all greater values (shift just indices)
      for (; j > k; --j) {
        const T b = evals_ptr[out_ptr[j - 1]];
        if (a > b) {
          break;
        }
        out_ptr[j] = out_ptr[j - 1];
      }
      // and insert the current index in the empty place, such that eigenvalues are sorted.
      out_ptr[j] = ii;
      ++i2;
    }
  }
  return k;
}

// This function returns number of non-deflated eigenvectors, together with two permutations
// - @p index_sorted          (sort(non-deflated)|sort(deflated)) -> initial.
// - @p index_sorted_coltype  (upper|dense|lower|sort(deflated)) -> initial
//
// The permutations will allow to keep the mapping between sorted eigenvalues and unsorted eigenvectors,
// which is useful since eigenvectors are more expensive to permuted, so we can keep them in their
// initial order.
//
// @param n                     number of eigenvalues
// @param types                 array[n] column type of each eigenvector after deflation (initial order)
// @param evals                 array[n] of eigenvalues sorted as perm_sorted
// @param perm_sorted           array[n] current -> initial (i.e. evals[i] -> types[perm_sorted[i]])
// @param index_sorted          array[n] (sort(non-deflated)|sort(deflated)) -> initial
// @param index_sorted_coltype  array[n] (upper|dense|lower|sort(deflated)) -> initial
//
// @return k                    number of non-deflated eigenvectors
template <class T>
SizeType stablePartitionIndexForDeflationArrays(const SizeType n, const ColType* types, const T* evals,
                                                const SizeType* perm_sorted, SizeType* index_sorted,
                                                SizeType* index_sorted_coltype) {
  // Note:
  // (in)  types
  //    column type of the initial indexing
  // (in)  perm_sorted
  //    initial <-- sorted by ascending eigenvalue
  // (out) index_sorted
  //    initial <-- (sort(non-deflated) | sort(deflated))
  // (out) index_sorted_coltype
  //    initial <-- (upper | dense | lower | sort(deflated))

  // Note:
  // This is the order how we want the eigenvectors to be sorted, since it leads to a nicer matrix
  // shape that allows to reduce the number of following operations (i.e. gemm)
  auto coltype_index = [](const ColType coltype) -> std::size_t {
    switch (coltype) {
      case ColType::UpperHalf:
        return 0;
      case ColType::Dense:
        return 1;
      case ColType::LowerHalf:
        return 2;
      case ColType::Deflated:
        return 3;
    }
    return DLAF_UNREACHABLE(std::size_t);
  };

  std::array<std::size_t, 4> offsets{0, 0, 0, 0};
  std::for_each(types, types + n, [&offsets, &coltype_index](const auto& coltype) {
    if (coltype != ColType::Deflated)
      offsets[1 + coltype_index(coltype)]++;
  });
  std::partial_sum(offsets.cbegin(), offsets.cend(), offsets.begin());

  const SizeType k = to_SizeType(offsets[coltype_index(ColType::Deflated)]);

  // Create the permutation (sorted non-deflated | sorted deflated) -> initial
  // Note:
  // Since during deflation, eigenvalues related to deflated eigenvectors, might not be sorted anymore,
  // this step also take care of sorting eigenvalues (actually just their related index) by their ascending value.
  SizeType i1 = 0;  // index of non-deflated values in out
  SizeType i2 = k;  // index of deflated values
  for (SizeType i = 0; i < n; ++i) {
    const SizeType ii = perm_sorted[i];

    // non-deflated are untouched, just squeeze them at the beginning as they appear
    if (types[ii] != ColType::Deflated) {
      index_sorted[i1] = ii;
      ++i1;
    }
    // deflated are the ones that can have been moved "out-of-order" by deflation...
    // ... so each time insert it in the right place based on eigenvalue value
    else {
      const T a = evals[ii];

      SizeType j = i2;
      // shift to right all greater values (shift just indices)
      for (; j > k; --j) {
        const T b = evals[index_sorted[j - 1]];
        if (a > b) {
          break;
        }
        index_sorted[j] = index_sorted[j - 1];
      }
      // and insert the current index in the empty place, such that eigenvalues are sorted.
      index_sorted[j] = ii;
      ++i2;
    }
  }

  // Create the permutation (upper|dense|lower|sort(deflated)) -> initial
  // Note:
  // non-deflated part is created starting from the initial order, because we are not interested
  // in having them sorted.
  // on the other hand, deflated part has to be sorted, so we copy the work from the index_sorted,
  // where they have been already sorted (post-deflation).
  for (SizeType j = 0; j < n; ++j) {
    const ColType& coltype = types[to_sizet(j)];
    if (coltype != ColType::Deflated) {
      auto& index_for_coltype = offsets[coltype_index(coltype)];
      index_sorted_coltype[index_for_coltype] = j;
      ++index_for_coltype;
    }
  }
  std::copy(index_sorted + k, index_sorted + n, index_sorted_coltype + k);

  return k;
}

template <class T>
auto stablePartitionIndexForDeflation(const SizeType i_begin, const SizeType i_end,
                                      Matrix<const ColType, Device::CPU>& c,
                                      Matrix<const T, Device::CPU>& evals,
                                      Matrix<const SizeType, Device::CPU>& in,
                                      Matrix<SizeType, Device::CPU>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto part_fn = [n](const auto& c_tiles_futs, const auto& evals_tiles_fut, const auto& in_tiles_futs,
                     const auto& out_tiles) {
    const TileElementIndex zero_idx(0, 0);
    const ColType* c_ptr = c_tiles_futs[0].get().ptr(zero_idx);
    const T* evals_ptr = evals_tiles_fut[0].get().ptr(zero_idx);
    const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero_idx);
    SizeType* out_ptr = out_tiles[0].ptr(zero_idx);

    return stablePartitionIndexForDeflationArrays(n, c_ptr, evals_ptr, in_ptr, out_ptr);
  };

  TileCollector tc{i_begin, i_end};
  return ex::when_all(ex::when_all_vector(tc.read(c)), ex::when_all_vector(tc.read(evals)),
                      ex::when_all_vector(tc.read(in)), ex::when_all_vector(tc.readwrite(out))) |
         di::transform(di::Policy<Backend::MC>(), std::move(part_fn));
}

template <class T>
auto stablePartitionIndexForDeflation(
    const SizeType i_begin, const SizeType i_end, Matrix<const ColType, Device::CPU>& c,
    Matrix<const T, Device::CPU>& evals, Matrix<const SizeType, Device::CPU>& in,
    Matrix<SizeType, Device::CPU>& out, Matrix<SizeType, Device::CPU>& out_by_coltype) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto part_fn = [n](const auto& c_tiles_futs, const auto& evals_tiles_futs, const auto& in_tiles_futs,
                     const auto& out_tiles, const auto& out_coltype_tiles) {
    const TileElementIndex zero_idx(0, 0);
    const ColType* c_ptr = c_tiles_futs[0].get().ptr(zero_idx);
    const T* evals_ptr = evals_tiles_futs[0].get().ptr(zero_idx);
    const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero_idx);
    SizeType* out_ptr = out_tiles[0].ptr(zero_idx);
    SizeType* out_coltype_ptr = out_coltype_tiles[0].ptr(zero_idx);

    return stablePartitionIndexForDeflationArrays(n, c_ptr, evals_ptr, in_ptr, out_ptr, out_coltype_ptr);
  };

  TileCollector tc{i_begin, i_end};
  return ex::when_all(ex::when_all_vector(tc.read(c)), ex::when_all_vector(tc.read(evals)),
                      ex::when_all_vector(tc.read(in)), ex::when_all_vector(tc.readwrite(out)),
                      ex::when_all_vector(tc.readwrite(out_by_coltype))) |
         di::transform(di::Policy<Backend::MC>(), std::move(part_fn));
}

inline void initColTypes(const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                         Matrix<ColType, Device::CPU>& coltypes) {
  namespace di = dlaf::internal;

  for (SizeType i = i_begin; i < i_end; ++i) {
    const ColType val = (i < i_split) ? ColType::UpperHalf : ColType::LowerHalf;
    di::transformDetach(
        di::Policy<Backend::MC>(),
        [](const ColType& ct, const matrix::Tile<ColType, Device::CPU>& tile) {
          for (SizeType i = 0; i < tile.size().rows(); ++i) {
            tile(TileElementIndex(i, 0)) = ct;
          }
        },
        di::whenAllLift(val, coltypes.readwrite(LocalTileIndex(i, 0))));
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
std::vector<GivensRotation<T>> applyDeflationToArrays(T rho, T tol, const SizeType len,
                                                      const SizeType* i_ptr, T* d_ptr, T* z_ptr,
                                                      ColType* c_ptr) {
  std::vector<GivensRotation<T>> rots;
  rots.reserve(to_sizet(len));

  SizeType i1 = 0;  // index of 1st element in the Givens rotation
  // Iterate over the indices of the sorted elements in pair (i1, i2) where i1 < i2 for every iteration
  for (SizeType i2 = 1; i2 < len; ++i2) {
    const SizeType i1s = i_ptr[i1];
    const SizeType i2s = i_ptr[i2];
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
    T r = std::hypot(z1, z2);
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
    //  Set the `i1` column as "Dense" if the `i2` column has opposite non-zero structure (i.e if one
    //  comes from Q1 and the other from Q2 or vice-versa)
    if ((c1 == ColType::UpperHalf && c2 == ColType::LowerHalf) ||
        (c1 == ColType::LowerHalf && c2 == ColType::UpperHalf)) {
      c1 = ColType::Dense;
    }
    c2 = ColType::Deflated;
  }

  return rots;
}

template <class T, class RhoSender, class TolSender>
auto applyDeflation(const SizeType i_begin, const SizeType i_end, RhoSender&& rho, TolSender&& tol,
                    Matrix<const SizeType, Device::CPU>& index, Matrix<T, Device::CPU>& d,
                    Matrix<T, Device::CPU>& z, Matrix<ColType, Device::CPU>& c) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, index.distribution());

  auto deflate_fn = [n](auto rho, auto tol, auto index_tiles_futs, auto d_tiles, auto z_tiles,
                        auto c_tiles) {
    const TileElementIndex zero_idx(0, 0);
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

// z is an input whose values are destroyed by this call (input + workspace)
template <class T, class KSender, class RhoSender>
void solveRank1Problem(const SizeType i_begin, const SizeType i_end, KSender&& k, RhoSender&& rho,
                       Matrix<const T, Device::CPU>& d, Matrix<T, Device::CPU>& z,
                       Matrix<T, Device::CPU>& evals, Matrix<const SizeType, Device::CPU>& i2,
                       Matrix<T, Device::CPU>& evecs) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, evals.distribution());
  const SizeType nb = evals.distribution().blockSize().rows();

  TileCollector tc{i_begin, i_end};

  // Note: at least two column of tiles per-worker, in the range [1, getTridiagRank1NWorkers()]
  const std::size_t nthreads = [nrtiles = (i_end - i_begin)]() {
    const std::size_t min_workers = 1;
    const std::size_t available_workers = getTridiagRank1NWorkers();
    const std::size_t ideal_workers = util::ceilDiv(to_sizet(nrtiles), to_sizet(2));
    return std::clamp(ideal_workers, min_workers, available_workers);
  }();

  ex::start_detached(
      ex::when_all(ex::just(std::make_unique<pika::barrier<>>(nthreads)), std::forward<KSender>(k),
                   std::forward<RhoSender>(rho), ex::when_all_vector(tc.read(d)),
                   ex::when_all_vector(tc.readwrite(z)), ex::when_all_vector(tc.readwrite(evals)),
                   ex::when_all_vector(tc.read(i2)), ex::when_all_vector(tc.readwrite(evecs)),
                   ex::just(std::vector<memory::MemoryView<T, Device::CPU>>())) |
      ex::transfer(di::getBackendScheduler<Backend::MC>(pika::execution::thread_priority::high)) |
      ex::bulk(nthreads, [nthreads, n, nb](std::size_t thread_idx, auto& barrier_ptr, auto& k, auto& rho,
                                           auto& d_tiles_futs, auto& z_tiles, auto& eval_tiles,
                                           const auto& i2_tile_arr, auto& evec_tiles, auto& ws_vecs) {
        const matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

        const SizeType* i2_perm = i2_tile_arr[0].get().ptr();

        const auto barrier_busy_wait = getTridiagRank1BarrierBusyWait();
        const std::size_t batch_size = util::ceilDiv(to_sizet(k), nthreads);
        const std::size_t begin = thread_idx * batch_size;
        const std::size_t end = std::min(thread_idx * batch_size + batch_size, to_sizet(k));

        // STEP 0a: Fill ones for deflated Eigenvectors. (single-thread)
        // Note: this step is completely independent from the rest, but it is small and it is going
        // to be dropped soon.
        // Note: use last thread that in principle should have less work to do
        if (thread_idx == nthreads - 1) {
          for (SizeType j = k; j < n; ++j) {
            const GlobalElementIndex jj(j, j);
            const auto linear_jj = distr.globalTileLinearIndex(jj);
            const auto jj_el = distr.tileElementIndex(jj);

            evec_tiles[to_sizet(linear_jj)](jj_el) = 1;
          }
        }

        // STEP 0b: Initialize workspaces (single-thread)
        if (thread_idx == 0) {
          ws_vecs.reserve(nthreads);
          for (std::size_t i = 0; i < nthreads; ++i)
            ws_vecs.emplace_back(to_sizet(k));
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 1: LAED4 (multi-thread)
        const T* d_ptr = d_tiles_futs[0].get().ptr();
        const T* z_ptr = z_tiles[0].ptr();

        {
          common::internal::SingleThreadedBlasScope single;

          T* eval_ptr = eval_tiles[0].ptr();

          for (std::size_t i = begin; i < end; ++i) {
            T& eigenval = eval_ptr[i];

            const SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, to_SizeType(i)));
            const SizeType i_col = distr.tileElementFromGlobalElement<Coord::Col>(to_SizeType(i));
            T* delta = evec_tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_col));

            lapack::laed4(to_int(k), to_int(i), d_ptr, z_ptr, delta, rho, &eigenval);
          }

          // Note: laed4 handles k <= 2 cases differently
          if (k <= 2) {
            // Note: The rows should be permuted for the k=2 case as well.
            if (k == 2) {
              T* ws = ws_vecs[thread_idx]();
              for (SizeType j = to_SizeType(begin); j < to_SizeType(end); ++j) {
                const SizeType j_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, j));
                const SizeType j_col = distr.tileElementFromGlobalElement<Coord::Col>(j);
                T* evec = evec_tiles[to_sizet(j_tile)].ptr(TileElementIndex(0, j_col));

                std::copy(evec, evec + k, ws);
                std::fill_n(evec, k, 0);  // by default "deflated"
                for (SizeType i = 0; i < n; ++i) {
                  const SizeType ii = i2_perm[i];
                  if (ii < k)
                    evec[i] = ws[ii];
                }
              }
            }
            return;
          }
        }

        // Note: This barrier ensures that LAED4 finished, so from now on values are available
        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 2a Compute weights (multi-thread)
        auto& q = evec_tiles;
        T* w = ws_vecs[thread_idx]();

        // - copy diagonal from q -> w (or just initialize with 1)
        if (thread_idx == 0) {
          for (SizeType i = 0; i < k; ++i) {
            const GlobalElementIndex kk(i, i);
            const auto diag_tile = distr.globalTileLinearIndex(kk);
            const auto diag_element = distr.tileElementIndex(kk);

            w[i] = q[to_sizet(diag_tile)](diag_element);
          }
        }
        else {
          std::fill_n(w, k, T(1));
        }

        // - compute productorial
        auto compute_w = [&](const GlobalElementIndex ij) {
          const auto q_tile = distr.globalTileLinearIndex(ij);
          const auto q_ij = distr.tileElementIndex(ij);

          const SizeType i = ij.row();
          const SizeType j = ij.col();

          w[i] *= q[to_sizet(q_tile)](q_ij) / (d_ptr[to_sizet(i)] - d_ptr[to_sizet(j)]);
        };

        for (SizeType j = to_SizeType(begin); j < to_SizeType(end); ++j) {
          for (SizeType i = 0; i < j; ++i)
            compute_w({i, j});

          for (SizeType i = j + 1; i < k; ++i)
            compute_w({i, j});
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 2B: reduce, then finalize computation with sign and square root (single-thread)
        if (thread_idx == 0) {
          for (SizeType i = 0; i < k; ++i) {
            for (std::size_t tidx = 1; tidx < nthreads; ++tidx) {
              const T* w_partial = ws_vecs[tidx]();
              w[i] *= w_partial[i];
            }
            z_tiles[0].ptr()[i] = std::copysign(std::sqrt(-w[i]), z_ptr[to_sizet(i)]);
          }
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 3: Compute eigenvectors of the modified rank-1 modification (normalize) (multi-thread)
        {
          common::internal::SingleThreadedBlasScope single;

          const T* w = z_ptr;
          T* s = ws_vecs[thread_idx]();

          for (SizeType j = to_SizeType(begin); j < to_SizeType(end); ++j) {
            for (SizeType i = 0; i < k; ++i) {
              const auto q_tile = distr.globalTileLinearIndex({i, j});
              const auto q_ij = distr.tileElementIndex({i, j});

              s[i] = w[i] / q[to_sizet(q_tile)](q_ij);
            }

            const T vec_norm = blas::nrm2(k, s, 1);

            for (SizeType i = 0; i < k; ++i) {
              const SizeType ii = i2_perm[i];
              const auto q_tile = distr.globalTileLinearIndex({i, j});
              const auto q_ij = distr.tileElementIndex({i, j});

              q[to_sizet(q_tile)](q_ij) = s[ii] / vec_norm;
            }
          }
        }
      }));
}

template <Backend B, Device D, class T, class RhoSender>
void mergeSubproblems(const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                      RhoSender&& rho, WorkSpace<T, D>& ws, WorkSpaceHost<T>& ws_h,
                      WorkSpaceHostMirror<T, D>& ws_hm) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  const LocalTileIndex idx_loc_begin(i_begin, i_begin);
  const SizeType nrtiles = i_end - i_begin;
  const LocalTileSize sz_loc_tiles(nrtiles, nrtiles);

  const LocalTileIndex idx_begin_tiles_vec(i_begin, 0);
  const LocalTileSize sz_tiles_vec(nrtiles, 1);

  // Calculate the size of the upper subproblem
  const SizeType n1 = problemSize(i_begin, i_split, ws.e0.distribution());

  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleZVec(i_begin, i_split, i_end, rho, ws.e0, ws.z0);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.z0, ws_hm.z0);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  auto scaled_rho = scaleRho(std::move(rho)) | ex::split();

  // Calculate the tolerance used for deflation
  auto tol = calcTolerance(i_begin, i_end, ws_h.d0, ws_hm.z0);

  // Initialize the column types vector `c`
  initColTypes(i_begin, i_split, i_end, ws_h.c);

  // Initialize `i1` as identity just for single tile sub-problems
  if (i_split == i_begin + 1) {
    initIndex(i_begin, i_split, ws_h.i1);
  }
  if (i_split + 1 == i_end) {
    initIndex(i_split, i_end, ws_h.i1);
  }

  // Update indices of second sub-problem
  addIndex(i_split, i_end, n1, ws_h.i1);

  // Step #1
  //
  //    i1 (in)  : initial <--- pre_sorted per sub-problem
  //    i2 (out) : initial <--- pre_sorted
  //
  // - deflate `d`, `z` and `c`
  // - apply Givens rotations to `Q` - `evecs`
  //
  sortIndex(i_begin, i_end, ex::just(n1), ws_h.d0, ws_h.i1, ws_hm.i2);

  auto rots =
      applyDeflation(i_begin, i_end, scaled_rho, std::move(tol), ws_hm.i2, ws_h.d0, ws_hm.z0, ws_h.c);

  applyGivensRotationsToMatrixColumns(i_begin, i_end, std::move(rots), ws.e0);

  // Step #2
  //
  //    i2 (in)  : initial  <--- pre_sorted
  //    i5 (out) : initial  <--- sorted by coltype
  //    i3 (out) : initial  <--- deflated
  //    i4 (out) : deflated <--- sorted by coltype
  //
  // Note: `i3[k:] == i5[k:]` (i.e. deflated part are sorted in the same way)
  //
  // - permute eigenvectors in `e0` using `i5` so that they are sorted by column type in `e1`
  // - reorder `d0 -> d1`, `z0 -> z1`, using `i3` such that deflated entries are at the bottom.
  // - compute permutation `i4`: sorted by col type ---> deflated
  // - solve rank-1 problem and save eigenvalues in `d0` and `d1` (copy) and eigenvectors in `e2` (sorted
  // by coltype)
  // - set deflated diagonal entries of `U` to 1 (temporary solution until optimized GEMM is implemented)
  //
  //  | U | U | D | D |   |   | DF | DF |  U:  UpperHalf
  //  | U | U | D | D |   |   | DF | DF |  D:  Dense
  //  |   |   | D | D | L | L | DF | DF |  L:  LowerHalf
  //  |   |   | D | D | L | L | DF | DF |  DF: Deflated
  //  |   |   | D | D | L | L | DF | DF |
  //
  auto k =
      stablePartitionIndexForDeflation(i_begin, i_end, ws_h.c, ws_h.d0, ws_hm.i2, ws_h.i3, ws_hm.i5) |
      ex::split();

  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.i5, ws.i5);
  dlaf::permutations::permute<B, D, T, Coord::Col>(i_begin, i_end, ws.i5, ws.e0, ws.e1);

  applyIndex(i_begin, i_end, ws_h.i3, ws_h.d0, ws_hm.d1);
  applyIndex(i_begin, i_end, ws_h.i3, ws_hm.z0, ws_hm.z1);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.d1, ws_h.d0);

  //
  //    i3 (in)  : initial  <--- deflated
  //    i2 (out) : deflated <--- initial
  //
  invertIndex(i_begin, i_end, ws_h.i3, ws_hm.i2);

  //
  //    i5 (in)  : initial  <--- sort by coltype
  //    i2 (in)  : deflated <--- initial
  //    i4 (out) : deflated <--- sort by col type
  //
  // This allows to work in rank1 solver with columns sorted by type, so that they are well-shaped for
  // an optimized gemm, but still keeping track of where the actual position sorted by eigenvalues is.
  applyIndex(i_begin, i_end, ws_hm.i5, ws_hm.i2, ws_h.i4);

  // Note:
  // This is needed to set to zero elements of e2 outside of the k by k top-left part.
  // The input is not required to be zero for solveRank1Problem.
  matrix::util::set0<Backend::MC>(pika::execution::thread_priority::normal, idx_loc_begin, sz_loc_tiles,
                                  ws_hm.e2);
  solveRank1Problem(i_begin, i_end, k, scaled_rho, ws_hm.d1, ws_hm.z1, ws_h.d0, ws_h.i4, ws_hm.e2);
  copy(idx_loc_begin, sz_loc_tiles, ws_hm.e2, ws.e2);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  dlaf::multiplication::internal::generalSubMatrix<B, D, T>(i_begin, i_end, blas::Op::NoTrans,
                                                            blas::Op::NoTrans, T(1), ws.e1, ws.e2, T(0),
                                                            ws.e0);

  // Step #4: Final permutation to sort eigenvalues and eigenvectors
  //
  //    i1 (in)  : deflated <--- deflated  (identity map)
  //    i2 (out) : deflated <--- post_sorted
  //
  initIndex(i_begin, i_end, ws_h.i1);
  sortIndex(i_begin, i_end, std::move(k), ws_h.d0, ws_h.i1, ws_hm.i2);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.i2, ws_h.i1);
}

// The bottom row of Q1 and the top row of Q2. The bottom row of Q1 is negated if `rho < 0`.
//
// Note that the norm of `z` is sqrt(2) because it is a concatination of two normalized vectors. Hence
// to normalize `z` we have to divide by sqrt(2).
template <class T, Device D, class RhoSender>
void assembleDistZVec(comm::CommunicatorGrid& grid,
                      common::Pipeline<comm::Communicator>& full_task_chain, const SizeType i_begin,
                      const SizeType i_split, const SizeType i_end, RhoSender&& rho,
                      Matrix<const T, D>& evecs, Matrix<T, D>& z) {
  namespace ex = pika::execution::experimental;

  const matrix::Distribution& dist = evecs.distribution();
  comm::Index2D this_rank = dist.rankIndex();

  // Iterate over tiles of Q1 and Q2 around the split row `i_split`.
  for (SizeType i = i_begin; i < i_end; ++i) {
    // True if tile is in Q1
    bool top_tile = i < i_split;
    // Move to the row below `i_split` for `Q2`
    const SizeType evecs_row = i_split - ((top_tile) ? 1 : 0);
    const GlobalTileIndex idx_evecs(evecs_row, i);
    const GlobalTileIndex z_idx(i, 0);

    // Copy the last row of a `Q1` tile or the first row of a `Q2` tile into a column vector `z` tile
    comm::Index2D evecs_tile_rank = dist.rankGlobalTile(idx_evecs);
    if (evecs_tile_rank == this_rank) {
      // Copy the row into the column vector `z`
      assembleRank1UpdateVectorTileAsync<T, D>(top_tile, rho, evecs.read(idx_evecs), z.readwrite(z_idx));
      ex::start_detached(comm::scheduleSendBcast(full_task_chain.readwrite(), z.read(z_idx)));
    }
    else {
      const comm::IndexT_MPI root_rank = grid.rankFullCommunicator(evecs_tile_rank);
      ex::start_detached(comm::scheduleRecvBcast(full_task_chain.readwrite(), root_rank,
                                                 z.readwrite(z_idx)));
    }
  }
}

template <class T, class CommSender, class KSender, class RhoSender>
void solveRank1ProblemDist(CommSender&& row_comm, CommSender&& col_comm, const SizeType i_begin,
                           const SizeType i_end, const LocalTileIndex ij_begin_lc,
                           const LocalTileSize sz_loc_tiles, KSender&& k, RhoSender&& rho,
                           Matrix<const T, Device::CPU>& d, Matrix<T, Device::CPU>& z,
                           Matrix<T, Device::CPU>& evals, Matrix<const SizeType, Device::CPU>& i2,
                           Matrix<T, Device::CPU>& evecs) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  namespace tt = pika::this_thread::experimental;

  const matrix::Distribution& dist = evecs.distribution();

  TileCollector tc{i_begin, i_end};

  const SizeType n = problemSize(i_begin, i_end, dist);

  const SizeType m_subm_el_lc = [=]() {
    const auto i_loc_begin = ij_begin_lc.row();
    const auto i_loc_end = ij_begin_lc.row() + sz_loc_tiles.rows();
    return dist.localElementDistanceFromLocalTile<Coord::Row>(i_loc_begin, i_loc_end);
  }();

  const SizeType n_subm_el_lc = [=]() {
    const auto i_loc_begin = ij_begin_lc.col();
    const auto i_loc_end = ij_begin_lc.col() + sz_loc_tiles.cols();
    return dist.localElementDistanceFromLocalTile<Coord::Col>(i_loc_begin, i_loc_end);
  }();

  auto bcast_evals = [i_begin, i_end,
                      dist](common::Pipeline<comm::Communicator>& row_comm_chain,
                            const std::vector<matrix::Tile<T, Device::CPU>>& eval_tiles) {
    using dlaf::comm::internal::sendBcast_o;
    using dlaf::comm::internal::recvBcast_o;

    const comm::Index2D this_rank = dist.rankIndex();

    std::vector<ex::unique_any_sender<>> comms;
    comms.reserve(to_sizet(i_end - i_begin));

    for (SizeType i = i_begin; i < i_end; ++i) {
      const comm::IndexT_MPI evecs_tile_rank = dist.rankGlobalTile<Coord::Col>(i);
      auto& tile = eval_tiles[to_sizet(i - i_begin)];

      if (evecs_tile_rank == this_rank.col())
        comms.emplace_back(ex::when_all(row_comm_chain.readwrite(), ex::just(std::cref(tile))) |
                           transformMPI(sendBcast_o));
      else
        comms.emplace_back(ex::when_all(row_comm_chain.readwrite(),
                                        ex::just(evecs_tile_rank, std::cref(tile))) |
                           transformMPI(recvBcast_o));
    }

    return ex::ensure_started(ex::when_all_vector(std::move(comms)));
  };

  auto all_reduce_in_place = [](const dlaf::comm::Communicator& comm, MPI_Op reduce_op, const auto& data,
                                MPI_Request* req) {
    auto msg = comm::make_message(data);
    DLAF_MPI_CHECK_ERROR(MPI_Iallreduce(MPI_IN_PLACE, msg.data(), msg.count(), msg.mpi_type(), reduce_op,
                                        comm, req));
  };

  // Note: at least two column of tiles per-worker, in the range [1, getTridiagRank1NWorkers()]
  const std::size_t nthreads = [nrtiles = sz_loc_tiles.cols()]() {
    const std::size_t min_workers = 1;
    const std::size_t available_workers = getTridiagRank1NWorkers();
    const std::size_t ideal_workers = util::ceilDiv(to_sizet(nrtiles), to_sizet(2));
    return std::clamp(ideal_workers, min_workers, available_workers);
  }();

  ex::start_detached(
      ex::when_all(ex::just(std::make_unique<pika::barrier<>>(nthreads)),
                   std::forward<CommSender>(row_comm), std::forward<CommSender>(col_comm),
                   std::forward<KSender>(k), std::forward<RhoSender>(rho),
                   ex::when_all_vector(tc.read(d)), ex::when_all_vector(tc.readwrite(z)),
                   ex::when_all_vector(tc.readwrite(evals)), ex::when_all_vector(tc.read(i2)),
                   ex::when_all_vector(tc.readwrite(evecs)),
                   // additional workspaces
                   ex::just(std::vector<memory::MemoryView<T, Device::CPU>>()),
                   ex::just(memory::MemoryView<T, Device::CPU>())) |
      ex::transfer(di::getBackendScheduler<Backend::MC>(pika::execution::thread_priority::high)) |
      ex::bulk(nthreads, [nthreads, n, n_subm_el_lc, m_subm_el_lc, i_begin, ij_begin_lc, sz_loc_tiles,
                          dist, bcast_evals, all_reduce_in_place](
                             const std::size_t thread_idx, auto& barrier_ptr, auto& row_comm_wrapper,
                             auto& col_comm_wrapper, const auto& k, const auto& rho,
                             const auto& d_tiles_futs, auto& z_tiles, const auto& eval_tiles,
                             const auto& i2_tile_arr, const auto& evec_tiles, auto& ws_cols,
                             auto& ws_row) {
        using dlaf::comm::internal::transformMPI;

        common::Pipeline<comm::Communicator> row_comm_chain(row_comm_wrapper.get());
        const dlaf::comm::Communicator& col_comm = col_comm_wrapper.get();

        const auto barrier_busy_wait = getTridiagRank1BarrierBusyWait();
        const std::size_t batch_size =
            std::max<std::size_t>(2, util::ceilDiv(to_sizet(sz_loc_tiles.cols()), nthreads));
        const SizeType begin = to_SizeType(thread_idx * batch_size);
        const SizeType end = std::min(to_SizeType((thread_idx + 1) * batch_size), sz_loc_tiles.cols());

        // STEP 0a: Fill ones for deflated Eigenvectors. (single-thread)
        // Note: this step is completely independent from the rest, but it is small and it is going
        // to be dropped soon.
        // Note: use last threads that in principle should have less work to do
        if (thread_idx == nthreads - 1) {
          // just if there are deflated eigenvectors
          if (k < n) {
            const GlobalElementSize origin_el(i_begin * dist.blockSize().rows(),
                                              i_begin * dist.blockSize().cols());
            const SizeType* i2_perm = i2_tile_arr[0].get().ptr();

            for (SizeType i_subm_el = 0; i_subm_el < n; ++i_subm_el) {
              const SizeType j_subm_el = i2_perm[i_subm_el];

              // if it is a deflated vector
              if (j_subm_el >= k) {
                const GlobalElementIndex ij_el(origin_el.rows() + i_subm_el,
                                               origin_el.cols() + j_subm_el);
                const GlobalTileIndex ij = dist.globalTileIndex(ij_el);

                if (dist.rankIndex() == dist.rankGlobalTile(ij)) {
                  const LocalTileIndex ij_lc = dist.localTileIndex(ij);
                  const SizeType linear_subm_lc =
                      (ij_lc.row() - ij_begin_lc.row()) +
                      (ij_lc.col() - ij_begin_lc.col()) * sz_loc_tiles.rows();
                  const TileElementIndex ij_el_tl = dist.tileElementIndex(ij_el);
                  evec_tiles[to_sizet(linear_subm_lc)](ij_el_tl) = T{1};
                }
              }
            }
          }
        }

        // STEP 0b: Initialize workspaces (single-thread)
        if (thread_idx == 0) {
          // Note:
          // - nthreads are used for both LAED4 and weight calculation (one per worker thread)
          // - last one is used for reducing weights from all workers
          ws_cols.reserve(nthreads + 1);

          // Note:
          // Considering that
          // - LAED4 requires working on k elements
          // - Weight computation requires working on m_subm_el_lc
          //
          // and they are needed at two steps that cannot happen in parallel, we opted for allocating
          // the workspace with the highest requirement of memory, and reuse them for both steps.
          const SizeType max_size = std::max(k, m_subm_el_lc);
          for (std::size_t i = 0; i < nthreads; ++i)
            ws_cols.emplace_back(max_size);
          ws_cols.emplace_back(m_subm_el_lc);

          ws_row = memory::MemoryView<T, Device::CPU>(n_subm_el_lc);
          std::fill_n(ws_row(), n_subm_el_lc, 0);
        }

        // Note: we have to wait that LAED4 workspaces are ready to be used
        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        const T* d_ptr = d_tiles_futs[0].get().ptr();
        const T* z_ptr = z_tiles[0].ptr();

        // STEP 1: LAED4 (multi-thread)
        {
          common::internal::SingleThreadedBlasScope single;

          T* eval_ptr = eval_tiles[0].ptr();
          T* delta_ptr = ws_cols[thread_idx]();

          for (SizeType j_subm_lc = begin; j_subm_lc < end; ++j_subm_lc) {
            const SizeType j_lc = ij_begin_lc.col() + to_SizeType(j_subm_lc);
            const SizeType j = dist.globalTileFromLocalTile<Coord::Col>(j_lc);
            const SizeType n_subm_el = dist.globalTileElementDistance<Coord::Col>(i_begin, j);

            // Skip columns that are in the deflation zone
            if (n_subm_el >= k)
              break;

            const SizeType n_el_tl = std::min(dist.tileSize<Coord::Col>(j), k - n_subm_el);
            for (SizeType j_el_tl = 0; j_el_tl < n_el_tl; ++j_el_tl) {
              const SizeType j_el = n_subm_el + j_el_tl;

              // Solve the deflated rank-1 problem
              T& eigenval = eval_ptr[to_sizet(j_el)];
              lapack::laed4(to_int(k), to_int(j_el), d_ptr, z_ptr, delta_ptr, rho, &eigenval);

              // copy the parts from delta stored on this rank
              for (SizeType i_subm_lc = 0; i_subm_lc < sz_loc_tiles.rows(); ++i_subm_lc) {
                const SizeType linear_subm_lc = i_subm_lc + to_SizeType(j_subm_lc) * sz_loc_tiles.rows();
                auto& evec_tile = evec_tiles[to_sizet(linear_subm_lc)];

                const SizeType i_lc = ij_begin_lc.row() + i_subm_lc;
                const SizeType i = dist.globalTileFromLocalTile<Coord::Row>(i_lc);
                const SizeType m_subm_el = dist.globalTileElementDistance<Coord::Row>(i_begin, i);

                const SizeType i_subm = i - i_begin;
                const auto& i2_perm = i2_tile_arr[to_sizet(i_subm)].get();

                const SizeType m_el_tl = std::min(dist.tileSize<Coord::Row>(i), n - m_subm_el);
                for (SizeType i_el_tl = 0; i_el_tl < m_el_tl; ++i_el_tl) {
                  const SizeType jj_subm_el = i2_perm({i_el_tl, 0});
                  if (jj_subm_el < k)
                    evec_tile({i_el_tl, j_el_tl}) = delta_ptr[jj_subm_el];
                }
              }
            }
          }
        }

        // Note: This barrier ensures that LAED4 finished, so from now on values are available
        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 2: Broadcast evals

        // Note: this ensures that evals broadcasting finishes before bulk releases resources
        struct sync_wait_on_exit_t {
          ex::unique_any_sender<> sender_;

          ~sync_wait_on_exit_t() {
            if (sender_)
              tt::sync_wait(std::move(sender_));
          }
        } bcast_barrier;

        if (thread_idx == 0)
          bcast_barrier.sender_ = bcast_evals(row_comm_chain, eval_tiles);

        // Note: laed4 handles k <= 2 cases differently
        if (k <= 2)
          return;

        // STEP 2 Compute weights (multi-thread)
        auto& q = evec_tiles;
        T* w = ws_cols[thread_idx]();

        // STEP 2a: copy diagonal from q -> w (or just initialize with 1)
        if (thread_idx == 0) {
          for (SizeType i_subm_lc = 0; i_subm_lc < sz_loc_tiles.rows(); ++i_subm_lc) {
            const SizeType i_lc = ij_begin_lc.row() + i_subm_lc;
            const SizeType i = dist.globalTileFromLocalTile<Coord::Row>(i_lc);
            const SizeType i_subm_el = dist.globalTileElementDistance<Coord::Row>(i_begin, i);
            const SizeType m_subm_el_lc =
                dist.localElementDistanceFromLocalTile<Coord::Row>(ij_begin_lc.row(), i_lc);
            const auto& i2 = i2_tile_arr[to_sizet(i - i_begin)].get();

            const SizeType m_el_tl = std::min(dist.tileSize<Coord::Row>(i), n - i_subm_el);
            for (SizeType i_el_tl = 0; i_el_tl < m_el_tl; ++i_el_tl) {
              const SizeType i_subm_el_lc = m_subm_el_lc + i_el_tl;

              const SizeType jj_subm_el = i2({i_el_tl, 0});
              const SizeType n_el = dist.globalTileElementDistance<Coord::Col>(0, i_begin);
              const SizeType jj_el = n_el + jj_subm_el;
              const SizeType jj = dist.globalTileFromGlobalElement<Coord::Col>(jj_el);

              if (dist.rankGlobalTile<Coord::Col>(jj) == dist.rankIndex().col()) {
                const SizeType jj_lc = dist.localTileFromGlobalTile<Coord::Col>(jj);
                const SizeType jj_subm_lc = jj_lc - ij_begin_lc.col();
                const SizeType jj_el_tl = dist.tileElementFromGlobalElement<Coord::Col>(jj_el);

                const SizeType linear_subm_lc = i_subm_lc + sz_loc_tiles.rows() * jj_subm_lc;

                w[i_subm_el_lc] = q[to_sizet(linear_subm_lc)]({i_el_tl, jj_el_tl});
              }
              else {
                w[i_subm_el_lc] = T(1);
              }
            }
          }
        }
        else {  // other workers
          std::fill_n(w, m_subm_el_lc, T(1));
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 2b: compute weights
        for (SizeType j_subm_lc = begin; j_subm_lc < end; ++j_subm_lc) {
          const SizeType j_lc = ij_begin_lc.col() + to_SizeType(j_subm_lc);
          const SizeType j = dist.globalTileFromLocalTile<Coord::Col>(j_lc);
          const SizeType n_subm_el = dist.globalTileElementDistance<Coord::Col>(i_begin, j);

          // Skip columns that are in the deflation zone
          if (n_subm_el >= k)
            break;

          const SizeType n_el_tl = std::min(dist.tileSize<Coord::Col>(j), k - n_subm_el);
          for (SizeType j_el_tl = 0; j_el_tl < n_el_tl; ++j_el_tl) {
            const SizeType j_subm_el = n_subm_el + j_el_tl;
            for (SizeType i_subm_lc = 0; i_subm_lc < sz_loc_tiles.rows(); ++i_subm_lc) {
              const SizeType i_lc = ij_begin_lc.row() + i_subm_lc;
              const SizeType i = dist.globalTileFromLocalTile<Coord::Row>(i_lc);
              const SizeType m_subm_el = dist.globalTileElementDistance<Coord::Row>(i_begin, i);

              auto& i2_perm = i2_tile_arr[to_sizet(i - i_begin)].get();

              const SizeType m_el_tl = std::min(dist.tileSize<Coord::Row>(i), n - m_subm_el);
              for (SizeType i_el_tl = 0; i_el_tl < m_el_tl; ++i_el_tl) {
                const SizeType ii_subm_el = i2_perm({i_el_tl, 0});

                // deflated zone
                if (ii_subm_el >= k)
                  continue;

                // diagonal
                if (ii_subm_el == j_subm_el)
                  continue;

                const SizeType linear_subm_lc = i_subm_lc + sz_loc_tiles.rows() * j_subm_lc;
                const SizeType i_subm_el_lc = i_subm_lc * dist.blockSize().rows() + i_el_tl;

                w[i_subm_el_lc] *= q[to_sizet(linear_subm_lc)]({i_el_tl, j_el_tl}) /
                                   (d_ptr[to_sizet(ii_subm_el)] - d_ptr[to_sizet(j_subm_el)]);
              }
            }
          }
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 2c: reduce, then finalize computation with sign and square root (single-thread)
        if (thread_idx == 0) {
          // local reduction from all bulk workers
          for (SizeType i = 0; i < m_subm_el_lc; ++i) {
            for (std::size_t tidx = 1; tidx < nthreads; ++tidx) {
              const T* w_partial = ws_cols[tidx]();
              w[i] *= w_partial[i];
            }
          }

          tt::sync_wait(ex::when_all(row_comm_chain.readwrite(),
                                     ex::just(MPI_PROD, common::make_data(w, m_subm_el_lc))) |
                        transformMPI(all_reduce_in_place));

          T* weights = ws_cols[nthreads]();
          for (SizeType i_subm_el_lc = 0; i_subm_el_lc < m_subm_el_lc; ++i_subm_el_lc) {
            const SizeType i_subm_lc = i_subm_el_lc / dist.blockSize().rows();
            const SizeType i_lc = ij_begin_lc.row() + i_subm_lc;
            const SizeType i = dist.globalTileFromLocalTile<Coord::Row>(i_lc);
            const SizeType i_subm = i - i_begin;
            const SizeType i_subm_el =
                i_subm * dist.blockSize().rows() + i_subm_el_lc % dist.blockSize().rows();

            const auto* i2_perm = i2_tile_arr[0].get().ptr();
            const SizeType ii_subm_el = i2_perm[i_subm_el];
            weights[to_sizet(i_subm_el_lc)] =
                std::copysign(std::sqrt(-w[i_subm_el_lc]), z_ptr[to_sizet(ii_subm_el)]);
          }
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 3: Compute eigenvectors of the modified rank-1 modification (normalize) (multi-thread)

        // STEP 3a: Form evecs using weights vector and compute (local) sum of squares
        {
          common::internal::SingleThreadedBlasScope single;

          const T* w = ws_cols[nthreads]();
          T* sum_squares = ws_row();

          for (SizeType j_subm_lc = begin; j_subm_lc < end; ++j_subm_lc) {
            const SizeType j_lc = ij_begin_lc.col() + to_SizeType(j_subm_lc);
            const SizeType j = dist.globalTileFromLocalTile<Coord::Col>(j_lc);
            const SizeType n_subm_el = dist.globalTileElementDistance<Coord::Col>(i_begin, j);

            // Skip columns that are in the deflation zone
            if (n_subm_el >= k)
              break;

            const SizeType n_el_tl = std::min(dist.tileSize<Coord::Col>(j), k - n_subm_el);
            for (SizeType j_el_tl = 0; j_el_tl < n_el_tl; ++j_el_tl) {
              const SizeType j_subm_el_lc = j_subm_lc * dist.blockSize().cols() + j_el_tl;
              for (SizeType i_subm_lc = 0; i_subm_lc < sz_loc_tiles.rows(); ++i_subm_lc) {
                const SizeType i_lc = ij_begin_lc.row() + i_subm_lc;
                const SizeType i = dist.globalTileFromLocalTile<Coord::Row>(i_lc);
                const SizeType m_subm_el = dist.globalTileElementDistance<Coord::Row>(i_begin, i);

                const SizeType i_subm = i - i_begin;
                const auto& i2_perm = i2_tile_arr[to_sizet(i_subm)].get();

                const SizeType linear_subm_lc = i_subm_lc + sz_loc_tiles.rows() * j_subm_lc;
                const auto& q_tile = q[to_sizet(linear_subm_lc)];

                const SizeType m_el_tl = std::min(dist.tileSize<Coord::Row>(i), n - m_subm_el);
                for (SizeType i_el_tl = 0; i_el_tl < m_el_tl; ++i_el_tl) {
                  const SizeType ii_subm_el = i2_perm({i_el_tl, 0});

                  const SizeType i_subm_el_lc = i_subm_lc * dist.blockSize().rows() + i_el_tl;
                  if (ii_subm_el >= k)
                    q_tile({i_el_tl, j_el_tl}) = 0;
                  else
                    q_tile({i_el_tl, j_el_tl}) = w[i_subm_el_lc] / q_tile({i_el_tl, j_el_tl});
                }

                sum_squares[j_subm_el_lc] +=
                    blas::dot(m_el_tl, q_tile.ptr({0, j_el_tl}), 1, q_tile.ptr({0, j_el_tl}), 1);
              }
            }
          }
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 3b: Reduce to get the sum of all squares on all ranks
        if (thread_idx == 0)
          tt::sync_wait(ex::just(std::cref(col_comm), MPI_SUM,
                                 common::make_data(ws_row(), n_subm_el_lc)) |
                        transformMPI(all_reduce_in_place));

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 3c: Normalize (compute norm of each column and scale column vector)
        {
          common::internal::SingleThreadedBlasScope single;

          const T* sum_squares = ws_row();

          for (SizeType j_subm_lc = begin; j_subm_lc < end; ++j_subm_lc) {
            const SizeType j_lc = ij_begin_lc.col() + to_SizeType(j_subm_lc);
            const SizeType j = dist.globalTileFromLocalTile<Coord::Col>(j_lc);
            const SizeType n_subm_el = dist.globalTileElementDistance<Coord::Col>(i_begin, j);

            // Skip columns that are in the deflation zone
            if (n_subm_el >= k)
              break;

            const SizeType n_el_tl = std::min(dist.tileSize<Coord::Col>(j), k - n_subm_el);
            for (SizeType j_el_tl = 0; j_el_tl < n_el_tl; ++j_el_tl) {
              const SizeType j_subm_el_lc = j_subm_lc * dist.blockSize().cols() + j_el_tl;
              const T vec_norm = std::sqrt(sum_squares[j_subm_el_lc]);

              for (SizeType i_subm_lc = 0; i_subm_lc < sz_loc_tiles.rows(); ++i_subm_lc) {
                const SizeType linear_subm_lc = i_subm_lc + sz_loc_tiles.rows() * j_subm_lc;
                const SizeType i_lc = ij_begin_lc.row() + i_subm_lc;
                const SizeType i = dist.globalTileFromLocalTile<Coord::Row>(i_lc);
                const SizeType m_subm_el = dist.globalTileElementDistance<Coord::Row>(i_begin, i);

                const SizeType m_el_tl = std::min(dist.tileSize<Coord::Row>(i), n - m_subm_el);
                blas::scal(m_el_tl, 1 / vec_norm, q[to_sizet(linear_subm_lc)].ptr({0, j_el_tl}), 1);
              }
            }
          }
        }
      }));
}

// Distributed version of the tridiagonal solver on CPUs
template <Backend B, class T, Device D, class RhoSender>
void mergeDistSubproblems(comm::CommunicatorGrid& grid,
                          common::Pipeline<comm::Communicator>& full_task_chain,
                          common::Pipeline<comm::Communicator>& row_task_chain,
                          common::Pipeline<comm::Communicator>& col_task_chain, const SizeType i_begin,
                          const SizeType i_split, const SizeType i_end, RhoSender&& rho,
                          WorkSpace<T, D>& ws, WorkSpaceHost<T>& ws_h,
                          DistWorkSpaceHostMirror<T, D>& ws_hm) {
  namespace ex = pika::execution::experimental;

  const matrix::Distribution& dist_evecs = ws.e0.distribution();

  // Calculate the size of the upper subproblem
  const SizeType n1 = dist_evecs.globalTileElementDistance<Coord::Row>(i_begin, i_split);

  // The local size of the subproblem
  const GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  const LocalTileIndex idx_loc_begin{dist_evecs.nextLocalTileFromGlobalTile<Coord::Row>(i_begin),
                                     dist_evecs.nextLocalTileFromGlobalTile<Coord::Col>(i_begin)};
  const LocalTileIndex idx_loc_end{dist_evecs.nextLocalTileFromGlobalTile<Coord::Row>(i_end),
                                   dist_evecs.nextLocalTileFromGlobalTile<Coord::Col>(i_end)};
  const LocalTileSize sz_loc_tiles = idx_loc_end - idx_loc_begin;
  const LocalTileIndex idx_begin_tiles_vec(i_begin, 0);
  const LocalTileSize sz_tiles_vec(i_end - i_begin, 1);

  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleDistZVec(grid, full_task_chain, i_begin, i_split, i_end, rho, ws.e0, ws.z0);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.z0, ws_hm.z0);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  auto scaled_rho = scaleRho(std::move(rho)) | ex::split();

  // Calculate the tolerance used for deflation
  auto tol = calcTolerance(i_begin, i_end, ws_h.d0, ws_hm.z0);

  // Initialize the column types vector `c`
  initColTypes(i_begin, i_split, i_end, ws_h.c);

  // Step #1
  //
  //    i1 (out) : initial <--- initial (identity map)
  //    i2 (out) : initial <--- pre_sorted
  //
  // - deflate `d`, `z` and `c`
  // - apply Givens rotations to `Q` - `evecs`
  //
  if (i_split == i_begin + 1) {
    initIndex(i_begin, i_split, ws_h.i1);
  }
  if (i_split + 1 == i_end) {
    initIndex(i_split, i_end, ws_h.i1);
  }
  addIndex(i_split, i_end, n1, ws_h.i1);
  sortIndex(i_begin, i_end, ex::just(n1), ws_h.d0, ws_h.i1, ws_hm.i2);

  auto rots =
      applyDeflation(i_begin, i_end, scaled_rho, std::move(tol), ws_hm.i2, ws_h.d0, ws_hm.z0, ws_h.c);

  // Make sure Isend/Irecv messages don't match between calls by providing a unique `tag`
  //
  // Note: i_split is unique
  const comm::IndexT_MPI tag = to_int(i_split);
  applyGivensRotationsToMatrixColumns(grid, row_task_chain, tag, i_begin, i_end, std::move(rots), ws.e0);
  // Placeholder for rearranging the eigenvectors: (local permutation)
  copy(idx_loc_begin, sz_loc_tiles, ws.e0, ws.e1);

  // Step #2
  //
  //    i2 (in)  : initial <--- pre_sorted
  //    i3 (out) : initial <--- deflated
  //
  // - reorder `d0 -> d1`, `z0 -> z1`, using `i3` such that deflated entries are at the bottom.
  // - solve the rank-1 problem and save eigenvalues in `d0` and `d1` (copy) and eigenvectors in `e2`.
  // - set deflated diagonal entries of `U` to 1 (temporary solution until optimized GEMM is implemented)
  //
  auto k =
      stablePartitionIndexForDeflation(i_begin, i_end, ws_h.c, ws_h.d0, ws_hm.i2, ws_h.i3) | ex::split();
  applyIndex(i_begin, i_end, ws_h.i3, ws_h.d0, ws_hm.d1);
  applyIndex(i_begin, i_end, ws_h.i3, ws_hm.z0, ws_hm.z1);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.d1, ws_h.d0);

  //
  //    i3 (in)  : initial <--- deflated
  //    i2 (out) : initial ---> deflated
  //
  invertIndex(i_begin, i_end, ws_h.i3, ws_hm.i2);

  // Note: here ws_hm.z0 is used as a contiguous buffer for the laed4 call
  matrix::util::set0<Backend::MC>(pika::execution::thread_priority::normal, idx_loc_begin, sz_loc_tiles,
                                  ws_hm.e2);
  solveRank1ProblemDist(row_task_chain.readwrite(), col_task_chain.readwrite(), i_begin, i_end,
                        idx_loc_begin, sz_loc_tiles, k, std::move(scaled_rho), ws_hm.d1, ws_hm.z1,
                        ws_h.d0, ws_hm.i2, ws_hm.e2);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  copy(idx_loc_begin, sz_loc_tiles, ws_hm.e2, ws.e2);
  dlaf::multiplication::internal::generalSubMatrix<B, D, T>(grid, row_task_chain, col_task_chain,
                                                            i_begin, i_end, T(1), ws.e1, ws.e2, T(0),
                                                            ws.e0);

  // Step #4: Final permutation to sort eigenvalues and eigenvectors
  //
  //    i1 (in)  : deflated <--- deflated  (identity map)
  //    i2 (out) : deflated <--- post_sorted
  //
  initIndex(i_begin, i_end, ws_h.i1);
  sortIndex(i_begin, i_end, std::move(k), ws_h.d0, ws_h.i1, ws_hm.i2);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.i2, ws_h.i1);
}
}
