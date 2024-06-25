//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <pika/barrier.hpp>
#include <pika/execution.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/kernels.h>
#include <dlaf/communication/kernels/internal/broadcast.h>
#include <dlaf/eigensolver/internal/get_tridiag_rank1_barrier_busy_wait.h>
#include <dlaf/eigensolver/internal/get_tridiag_rank1_nworkers.h>
#include <dlaf/eigensolver/tridiag_solver/coltype.h>
#include <dlaf/eigensolver/tridiag_solver/index_manipulation.h>
#include <dlaf/eigensolver/tridiag_solver/kernels_async.h>
#include <dlaf/eigensolver/tridiag_solver/rot.h>
#include <dlaf/eigensolver/tridiag_solver/tile_collector.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/distribution_extensions.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/multiplication/general.h>
#include <dlaf/multiplication/general/api.h>
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
  Matrix<SizeType, D> i5b;
  Matrix<SizeType, D> i6;
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
  HostMirrorMatrix<SizeType, D> i5;
  HostMirrorMatrix<SizeType, D> i5b;
  HostMirrorMatrix<SizeType, D> i6;
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
  using pika::execution::thread_stacksize;
  return std::forward<RhoSender>(rho) | di::transform(di::Policy<Backend::MC>(thread_stacksize::nostack),
                                                      [](auto rho) { return 2 * std::abs(rho); });
}

// Returns the maximum element of a portion of a column vector from tile indices `i_begin` to `i_end`
//
template <class T>
auto maxVectorElement(const SizeType i_begin, const SizeType i_end, Matrix<const T, Device::CPU>& vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  std::vector<ex::unique_any_sender<T>> tiles_max;
  tiles_max.reserve(to_sizet(i_end - i_begin));
  for (SizeType i = i_begin; i < i_end; ++i) {
    tiles_max.push_back(di::whenAllLift(lapack::Norm::Max, vec.read(LocalTileIndex(i, 0))) |
                        di::transform(di::Policy<Backend::MC>(thread_stacksize::nostack),
                                      tile::internal::lange_o));
  }

  auto tol_calc_fn = [](const std::vector<T>& maxvals) {
    return *std::max_element(maxvals.begin(), maxvals.end());
  };

  return ex::when_all_vector(std::move(tiles_max)) |
         di::transform(di::Policy<Backend::MC>(thread_stacksize::nostack), std::move(tol_calc_fn));
}

// The tolerance calculation is the same as the one used in LAPACK's stedc implementation [1].
//
// [1] LAPACK 3.10.0, file dlaed2.f, line 315, variable TOL
template <class T>
auto calcTolerance(const SizeType i_begin, const SizeType i_end, Matrix<const T, Device::CPU>& d,
                   Matrix<const T, Device::CPU>& z) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  auto dmax = maxVectorElement(i_begin, i_end, d);
  auto zmax = maxVectorElement(i_begin, i_end, z);

  auto tol_fn = [](T dmax, T zmax) {
    return 8 * std::numeric_limits<T>::epsilon() * std::max(dmax, zmax);
  };

  return ex::when_all(std::move(dmax), std::move(zmax)) |
         di::transform(di::Policy<Backend::MC>(thread_stacksize::nostack), std::move(tol_fn)) |
         // TODO: This releases the tiles that are kept in the operation state.
         // This is a temporary fix and needs to be replaced by a different
         // adaptor or different lifetime guarantees. This is tracked in
         // https://github.com/pika-org/pika/issues/479.
         ex::ensure_started();
}

// Note:
// This is the order how we want the eigenvectors to be sorted, since it leads to a nicer matrix
// shape that allows to reduce the number of following operations (i.e. gemm)
inline std::size_t ev_sort_order(const ColType coltype) {
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
}

// This function returns number of non-deflated eigenvectors and a tuple with number of upper, dense
// and lower non-deflated eigenvectors, together with two permutations:
// - @p index_sorted          (sort(non-deflated)|sorted(deflated) -> initial.
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
// @return n_udl                tuple with number of upper, dense and lower eigenvectors
template <class T>
auto stablePartitionIndexForDeflationArrays(const SizeType n, const ColType* types, const T* evals,
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

  std::array<std::size_t, 4> offsets{0, 0, 0, 0};
  std::for_each(types, types + n, [&offsets](const auto& coltype) {
    if (coltype != ColType::Deflated)
      offsets[1 + ev_sort_order(coltype)]++;
  });

  std::array<std::size_t, 3> n_udl{offsets[1 + ev_sort_order(ColType::UpperHalf)],
                                   offsets[1 + ev_sort_order(ColType::Dense)],
                                   offsets[1 + ev_sort_order(ColType::LowerHalf)]};

  std::partial_sum(offsets.cbegin(), offsets.cend(), offsets.begin());

  const SizeType k = to_SizeType(offsets[ev_sort_order(ColType::Deflated)]);

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
      auto& index_for_coltype = offsets[ev_sort_order(coltype)];
      index_sorted_coltype[index_for_coltype] = j;
      ++index_for_coltype;
    }
  }
  std::copy(index_sorted + k, index_sorted + n, index_sorted_coltype + k);

  return std::tuple(k, std::move(n_udl));
}

// This function returns number of global non-deflated eigenvectors, together with two permutations:
// - @p index_sorted          (sort(non-deflated)|sort(deflated)) -> initial.
// - @p index_sorted_coltype  (sort(upper)|sort(dense)|sort(lower)|sort(deflated)) -> initial
//
// Both permutations are represented using global indices, but:
// - @p index_sorted          sorts "globally", i.e. considering all evecs across ranks
// - @p index_sorted_coltype  sorts "locally", i.e. considering just evecs from the same rank (global indices)
// - @p i5_lc                 sorts "locally", i.e. considering just evecs from current rank (local indices)
//
// In addition, even if all ranks have the full permutation, it is important to highlight that
// thanks to how it is built, i.e. rank-independent permutations, @p index_sorted_coltype can be used as
// if it was distributed, since each "local" tile would contain global indices that are valid
// just on the related rank.
//
// rank                   |     0     |     1     |     0     |
// initial                | 0U| 1L| 2X| 3U| 4U| 5X| 6L| 7L| 8X|
// index_sorted           | 0U| 1L| 3U| 4U| 6L| 7L| 2X| 5X| 8X| -> sort(non-deflated) | sort(deflated)
// index_sorted_col_type  | 0U| 1L| 6L| 3U| 4U| 5X| 7L| 2X| 8X| -> rank0(ULLLXX) - rank1(UUX) (global indices)
// i5_lc                  | 0U| 1L| 3L|...........| 4L| 2X| 5X| -> rank0(ULLLXX)  (local indices)
// i5_lc                  |...........| 0U| 1U| 2X|...........| -> rank1(UUX)     (local indices)
//
// index_sorted_col_type can be used "locally":
// on rank0               | 0U| 1L| 6L| --| --| --| 7L| 2X| 8X| -> ULLLXX
// on rank1               | --| --| --| 3U| 4U| 5X| --| --| --| -> UUX
//
// i5_lc is just local to the current rank, and it is a view over index_sorted_col_type where indices are local.
//
// where U: Upper, D: Dense, L: Lower, X: Deflated
//
// The permutations will allow to keep the mapping between sorted eigenvalues and unsorted eigenvectors,
// which is useful since eigenvectors are more expensive to permute, so we can keep them in their
// initial order.
//
// @param types                 array[n]    column type of each eigenvector after deflation (initial order)
// @param evals                 array[n]    of eigenvalues sorted as perm_sorted
// @param perm_sorted           array[n]    current -> initial (i.e. evals[i] -> types[perm_sorted[i]])
// @param index_sorted          array[n]    global(sort(non-deflated)|sort(deflated))) -> initial
// @param index_sorted_coltype  array[n]    local(sort(upper)|sort(dense)|sort(lower)|sort(deflated))) -> initial
// @param i5_lc                 array[n_lc] local(sort(upper)|sort(dense)|sort(lower)|sort(deflated))) -> initial
//
// @return k                    number of non-deflated eigenvectors
// @return k_local              number of local non-deflated eigenvectors
// @return n_udl                tuple with global indices for [first_dense, last_dense, last_lower]
template <class T>
auto stablePartitionIndexForDeflationArrays(const matrix::Distribution& dist_sub, const ColType* types,
                                            const T* evals, SizeType* perm_sorted,
                                            SizeType* index_sorted, SizeType* index_sorted_coltype,
                                            SizeType* i5_lc, SizeType* i4, SizeType* i6) {
  const SizeType n = dist_sub.size().cols();
  const SizeType k = std::count_if(types, types + n,
                                   [](const ColType coltype) { return ColType::Deflated != coltype; });

  // Create the permutation (sorted non-deflated | sorted deflated) -> initial
  // Note:
  // Since during deflation, eigenvalues related to deflated eigenvectors, might not be sorted anymore,
  // this step also take care of sorting eigenvalues (actually just their related index) by their ascending value.
  SizeType i_nd = 0;  // index of non-deflated values
  SizeType i_x = k;   // index of deflated values
  for (SizeType i = 0; i < n; ++i) {
    const SizeType ii = perm_sorted[i];

    // non-deflated are untouched, just squeeze them at the beginning as they appear
    if (types[ii] != ColType::Deflated) {
      index_sorted[i_nd] = ii;
      ++i_nd;
    }
    // deflated are the ones that can have been moved "out-of-order" by deflation...
    // ... so each time insert it in the right place based on eigenvalue value
    else {
      const T a = evals[ii];

      SizeType j = i_x;
      // shift to right all greater values (just the indices)
      for (; j > k; --j) {
        const T b = evals[index_sorted[j - 1]];
        if (a > b) {
          break;
        }
        index_sorted[j] = index_sorted[j - 1];
      }
      // and insert the current index in the empty place, such that eigenvalues are sorted.
      index_sorted[j] = ii;
      ++i_x;
    }
  }

  // Create the permutation (sort(upper)|sort(dense)|sort(lower)|sort(deflated)) -> initial
  // Note:
  // index_sorted is used as "reference" in order to deal with deflated vectors in the right sorted order.
  // In this way, also non-deflated are considered in a sorted way, which is not a requirement,
  // but it does not hurt either.

  // Detect how many non-deflated per type (on each rank)
  using offsets_t = std::array<std::size_t, 4>;
  std::vector<offsets_t> offsets(to_sizet(dist_sub.grid_size().cols()), {0, 0, 0, 0});

  for (SizeType j_el = 0; j_el < n; ++j_el) {
    const SizeType jj_el = index_sorted[to_sizet(j_el)];
    const ColType coltype = types[to_sizet(jj_el)];

    const comm::IndexT_MPI rank = dist_sub.rank_global_element<Coord::Col>(jj_el);
    offsets_t& rank_offsets = offsets[to_sizet(rank)];

    if (coltype != ColType::Deflated)
      ++rank_offsets[1 + ev_sort_order(coltype)];
  }
  std::for_each(offsets.begin(), offsets.end(), [](offsets_t& rank_offsets) {
    std::partial_sum(rank_offsets.cbegin(), rank_offsets.cend(), rank_offsets.begin());
  });

  const SizeType k_lc =
      to_SizeType(offsets[to_sizet(dist_sub.rankIndex().col())][ev_sort_order(ColType::Deflated)]);

  // Each rank computes all rank permutations.
  // Using previously calculated offsets (per rank), the permutation is already split in column types,
  // so this loops over indices, checks the column type and eventually put the index in the right bin.
  for (SizeType j_el = 0; j_el < n; ++j_el) {
    const SizeType jj_el = index_sorted[to_sizet(j_el)];
    const ColType coltype = types[to_sizet(jj_el)];

    const comm::IndexT_MPI rank = dist_sub.rank_global_element<Coord::Col>(jj_el);
    offsets_t& rank_offsets = offsets[to_sizet(rank)];

    const SizeType jjj_el_lc = to_SizeType(rank_offsets[ev_sort_order(coltype)]++);
    using matrix::internal::distribution::global_element_from_local_element_on_rank;
    const SizeType jjj_el =
        global_element_from_local_element_on_rank<Coord::Col>(dist_sub, rank, jjj_el_lc);

    index_sorted_coltype[to_sizet(jjj_el)] = jj_el;
  }

  // This is the local version of the previous one, just for the current rank
  for (SizeType i_lc = 0; i_lc < dist_sub.local_size().cols(); ++i_lc) {
    const SizeType i = dist_sub.global_element_from_local_element<Coord::Col>(i_lc);
    i5_lc[i_lc] = dist_sub.local_element_from_global_element<Coord::Col>(index_sorted_coltype[i]);
  }

  std::array<SizeType, 3> n_udl = [&]() {
    SizeType first_dense;
    for (first_dense = 0; first_dense < n; ++first_dense) {
      const SizeType initial_el = index_sorted_coltype[to_sizet(first_dense)];
      const ColType coltype = types[to_sizet(initial_el)];
      if (ColType::UpperHalf != coltype)
        break;
    }

    // Note:
    // Eigenvectors will be sorted according index_sorted_coltype, i.e. local sort by coltype.
    // Since it is a local order, it is legit if deflated are globally interlaced with other column
    // types. However, GEMM will be able to skip just the last global contiguous group of deflated
    // eigenvectors, but not the ones interlaced with others.
    SizeType last_lower;
    for (last_lower = n - 1; last_lower >= 0; --last_lower) {
      const SizeType initial_el = index_sorted_coltype[to_sizet(last_lower)];
      const ColType coltype = types[to_sizet(initial_el)];
      if (ColType::Deflated != coltype)
        break;
    }

    SizeType last_dense;
    for (last_dense = last_lower; last_dense >= 0; --last_dense) {
      const SizeType initial_el = index_sorted_coltype[to_sizet(last_dense)];
      const ColType coltype = types[to_sizet(initial_el)];
      if (ColType::LowerHalf != coltype && ColType::Deflated != coltype)
        break;
    }

    return std::array<SizeType, 3>{first_dense, last_dense + 1, last_lower + 1};
  }();

  // invert i3 and store it in i2 (temporary)
  //    i3 (in)  : initial  <--- deflated
  //    i2 (out) : deflated <--- initial
  SizeType* i2 = perm_sorted;
  for (SizeType i = 0; i < n; ++i)
    i2[index_sorted[i]] = i;

  // compose i5 with i3 inverse (i2 temporarily)
  //    i5 (in)  : initial  <--- sort by coltype
  //    i2 (in)  : deflated <--- initial
  //    i4 (out) : deflated <--- sort by col type
  for (SizeType i = 0; i < n; ++i)
    i4[i] = i2[index_sorted_coltype[i]];

  // create i6
  //    i6 (out) : deflated <--- local(deflated)
  for (SizeType j_el = 0, jnd_el = 0; j_el < n; ++j_el) {
    const SizeType jj_el = index_sorted_coltype[to_sizet(j_el)];
    const ColType coltype = types[to_sizet(jj_el)];

    if (ColType::Deflated != coltype) {
      i6[j_el] = jnd_el;
      ++jnd_el;
    }
    else {
      i6[j_el] = i4[j_el];
    }
  }

  // invert i6 -> i2
  //    i6 (in)  : deflated        <--- local(deflated)
  //    i2 (out) : local(deflated) <--- deflated
  for (SizeType i = 0; i < n; ++i)
    i2[i6[i]] = i;

  return std::tuple(k, k_lc, n_udl);
}

template <class T>
auto stablePartitionIndexForDeflation(
    const SizeType i_begin, const SizeType i_end, Matrix<const ColType, Device::CPU>& c,
    Matrix<const T, Device::CPU>& evals, Matrix<const SizeType, Device::CPU>& in,
    Matrix<SizeType, Device::CPU>& out, Matrix<SizeType, Device::CPU>& out_by_coltype) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  const SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto part_fn = [n](const auto& c_tiles, const auto& evals_tiles, const auto& in_tiles,
                     const auto& out_tiles, const auto& out_coltype_tiles) {
    const TileElementIndex zero_idx(0, 0);
    const ColType* c_ptr = c_tiles[0].get().ptr(zero_idx);
    const T* evals_ptr = evals_tiles[0].get().ptr(zero_idx);
    const SizeType* in_ptr = in_tiles[0].get().ptr(zero_idx);
    SizeType* out_ptr = out_tiles[0].ptr(zero_idx);
    SizeType* out_coltype_ptr = out_coltype_tiles[0].ptr(zero_idx);

    return stablePartitionIndexForDeflationArrays(n, c_ptr, evals_ptr, in_ptr, out_ptr, out_coltype_ptr);
  };

  TileCollector tc{i_begin, i_end};
  return ex::when_all(ex::when_all_vector(tc.read(c)), ex::when_all_vector(tc.read(evals)),
                      ex::when_all_vector(tc.read(in)), ex::when_all_vector(tc.readwrite(out)),
                      ex::when_all_vector(tc.readwrite(out_by_coltype))) |
         di::transform(di::Policy<Backend::MC>(thread_stacksize::nostack), std::move(part_fn));
}

template <class T>
auto stablePartitionIndexForDeflation(
    const matrix::Distribution& dist_evecs, const SizeType i_begin, const SizeType i_end,
    Matrix<const ColType, Device::CPU>& c, Matrix<const T, Device::CPU>& evals,
    Matrix<SizeType, Device::CPU>& in, Matrix<SizeType, Device::CPU>& out,
    Matrix<SizeType, Device::CPU>& out_by_coltype, Matrix<SizeType, Device::CPU>& i5_lc,
    Matrix<SizeType, Device::CPU>& i4, Matrix<SizeType, Device::CPU>& i6) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  const SizeType n = problemSize(i_begin, i_end, in.distribution());

  const matrix::Distribution dist_evecs_sub(
      dist_evecs, {dist_evecs.global_element_index({i_begin, i_begin}, {0, 0}), {n, n}});

  auto part_fn = [dist_evecs_sub](const auto& c_tiles, const auto& evals_tiles, const auto& in_tiles,
                                  const auto& out_tiles, const auto& out_coltype_tiles,
                                  const auto& i5_lc_tiles, const auto& i4_tiles, const auto& i6_tiles) {
    const TileElementIndex zero_idx(0, 0);
    const ColType* c_ptr = c_tiles[0].get().ptr(zero_idx);
    const T* evals_ptr = evals_tiles[0].get().ptr(zero_idx);
    SizeType* in_ptr = in_tiles[0].ptr(zero_idx);
    SizeType* out_ptr = out_tiles[0].ptr(zero_idx);
    SizeType* out_coltype_ptr = out_coltype_tiles[0].ptr(zero_idx);
    SizeType* i5_lc_ptr = i5_lc_tiles.size() > 0 ? i5_lc_tiles[0].ptr(zero_idx) : nullptr;
    SizeType* i4_ptr = i4_tiles[0].ptr(zero_idx);
    SizeType* i6_ptr = i6_tiles[0].ptr(zero_idx);

    return stablePartitionIndexForDeflationArrays(dist_evecs_sub, c_ptr, evals_ptr, in_ptr, out_ptr,
                                                  out_coltype_ptr, i5_lc_ptr, i4_ptr, i6_ptr);
  };

  const SizeType i_begin_lc = dist_evecs.next_local_tile_from_global_tile<Coord::Col>(i_begin);
  const SizeType i_end_lc = dist_evecs.next_local_tile_from_global_tile<Coord::Col>(i_end);
  auto i5_lc_snd =
      select(i5_lc, common::iterate_range2d(LocalTileIndex(i_begin_lc, 0), LocalTileIndex(i_end_lc, 1)));

  TileCollector tc{i_begin, i_end};
  return ex::when_all(ex::when_all_vector(tc.read(c)), ex::when_all_vector(tc.read(evals)),
                      ex::when_all_vector(tc.readwrite(in)), ex::when_all_vector(tc.readwrite(out)),
                      ex::when_all_vector(tc.readwrite(out_by_coltype)),
                      ex::when_all_vector(std::move(i5_lc_snd)), ex::when_all_vector(tc.readwrite(i4)),
                      ex::when_all_vector(tc.readwrite(i6))) |
         di::transform(di::Policy<Backend::MC>(thread_stacksize::nostack), std::move(part_fn));
}

inline void initColTypes(const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                         Matrix<ColType, Device::CPU>& coltypes) {
  namespace di = dlaf::internal;
  using pika::execution::thread_stacksize;

  for (SizeType i = i_begin; i < i_end; ++i) {
    const ColType val = (i < i_split) ? ColType::UpperHalf : ColType::LowerHalf;
    di::transformDetach(
        di::Policy<Backend::MC>(thread_stacksize::nostack),
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
  using pika::execution::thread_stacksize;

  const SizeType n = problemSize(i_begin, i_end, index.distribution());

  auto deflate_fn = [n](auto rho, auto tol, auto index_tiles, auto d_tiles, auto z_tiles, auto c_tiles) {
    const TileElementIndex zero_idx(0, 0);
    const SizeType* i_ptr = index_tiles[0].get().ptr(zero_idx);
    T* d_ptr = d_tiles[0].ptr(zero_idx);
    T* z_ptr = z_tiles[0].ptr(zero_idx);
    ColType* c_ptr = c_tiles[0].ptr(zero_idx);
    return applyDeflationToArrays(rho, tol, n, i_ptr, d_ptr, z_ptr, c_ptr);
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(std::forward<RhoSender>(rho), std::forward<TolSender>(tol),
                             ex::when_all_vector(tc.read(index)), ex::when_all_vector(tc.readwrite(d)),
                             ex::when_all_vector(tc.readwrite(z)), ex::when_all_vector(tc.readwrite(c)));

  return di::transform(di::Policy<Backend::MC>(thread_stacksize::nostack), std::move(deflate_fn),
                       std::move(sender)) |
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
  using pika::execution::thread_priority;

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
      ex::transfer(di::getBackendScheduler<Backend::MC>(thread_priority::high)) |
      ex::bulk(nthreads, [nthreads, n, nb](std::size_t thread_idx, auto& barrier_ptr, auto& k, auto& rho,
                                           auto& d_tiles, auto& z_tiles, auto& eval_tiles,
                                           const auto& i2_tile_arr, auto& evec_tiles, auto& ws_vecs) {
        const matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

        const SizeType* i2_perm = i2_tile_arr[0].get().ptr();

        const auto barrier_busy_wait = getTridiagRank1BarrierBusyWait();
        const std::size_t batch_size = util::ceilDiv(to_sizet(k), nthreads);
        const std::size_t begin = thread_idx * batch_size;
        const std::size_t end = std::min(thread_idx * batch_size + batch_size, to_sizet(k));

        // STEP 0: Initialize workspaces (single-thread)
        if (thread_idx == 0) {
          ws_vecs.reserve(nthreads);
          for (std::size_t i = 0; i < nthreads; ++i)
            ws_vecs.emplace_back(to_sizet(k));
        }

        barrier_ptr->arrive_and_wait(barrier_busy_wait);

        // STEP 1: LAED4 (multi-thread)
        const T* d_ptr = d_tiles[0].get().ptr();
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

template <Backend B, class T, Device D, class KSender, class UDLSenders>
void multiplyEigenvectors(const SizeType sub_offset, const SizeType n, const SizeType n_upper,
                          const SizeType n_lower, Matrix<T, D>& e0, Matrix<T, D>& e1, Matrix<T, D>& e2,
                          KSender&& k, UDLSenders&& n_udl) {
  // Note:
  // This function computes E0 = E1 . E2
  //
  // where E1 is the matrix with eigenvectors and it looks like this
  //
  //               ┌──────────┐ k
  //               │    a     │ │
  //                            ▼
  //          ┌──  ┌───┬──────┬─┬────┐
  //          │    │UUU│DDDDDD│ │XXXX│
  //          │    │UUU│DDDDDD│ │XXXX│
  //  n_upper │    │UUU│DDDDDD│ │XXXX│
  //          │    │UUU│DDDDDD│ │XXXX│
  //          │    │UUU│DDDDDD│ │XXXX│
  //          ├──  ├───┼──────┼─┤XXXX│
  //          │    │   │DDDDDD│L│XXXX│
  //  n_lower │    │   │DDDDDD│L│XXXX│
  //          │    │   │DDDDDD│L│XXXX│
  //          └──  └───┴──────┴─┴────┘
  //                   │   b    │
  //                   └────────┘
  //
  // The multiplication in two different steps in order to skip zero blocks of the matrix, created by
  // the grouping of eigenvectors of different lengths (UPPER, DENSE and LOWER).
  //
  // 1. GEMM1 = TL . TOP
  // 2. GEMM2 = BR . BOTTOM
  // 3. copy DEFLATED
  //
  //                      ┌────────────┬────┐
  //                      │            │    │
  //                      │            │    │
  //                      │   T O P    │    │
  //                      │            │    │
  //                      │            │    │
  //                      ├────────────┤    │
  //                      │            │    │
  //                      │            │    │
  //                      │B O T T O M │    │
  //                      │            │    │
  //                      └────────────┴────┘
  //
  // ┌──────────┬─┬────┐  ┌────────────┬────┐
  // │          │0│    │  │            │    │
  // │          │0│ D  │  │            │    │
  // │   TL     │0│ E  │  │  GEMM 1    │ C  │
  // │          │0│ F  │  │            │    │
  // │          │0│ L  │  │            │ O  │
  // ├───┬──────┴─┤ A  │  ├────────────┤    │
  // │000│        │ T  │  │            │ P  │
  // │000│        │ E  │  │            │    │
  // │000│  BR    │ D  │  │  GEMM 2    │ Y  │
  // │000│        │    │  │            │    │
  // └───┴────────┴────┘  └────────────┴────┘

  namespace ex = pika::execution::experimental;
  using pika::execution::thread_priority;

  ex::start_detached(
      ex::when_all(std::forward<KSender>(k), std::forward<UDLSenders>(n_udl)) |
      ex::transfer(dlaf::internal::getBackendScheduler<Backend::MC>(thread_priority::high)) |
      ex::then([sub_offset, n, n_upper, n_lower, e0 = e0.subPipeline(), e1 = e1.subPipelineConst(),
                e2 = e2.subPipelineConst()](const SizeType k, std::array<std::size_t, 3> n_udl) mutable {
        using dlaf::matrix::internal::MatrixRef;

        const SizeType n_uh = to_SizeType(n_udl[ev_sort_order(ColType::UpperHalf)]);
        const SizeType n_de = to_SizeType(n_udl[ev_sort_order(ColType::Dense)]);
        const SizeType n_lh = to_SizeType(n_udl[ev_sort_order(ColType::LowerHalf)]);

        const SizeType a = n_uh + n_de;
        const SizeType b = n_de + n_lh;

        using GEMM = dlaf::multiplication::internal::General<B, D, T>;
        {
          MatrixRef<const T, D> e1_sub(e1, {{sub_offset, sub_offset}, {n_upper, a}});
          MatrixRef<const T, D> e2_sub(e2, {{sub_offset, sub_offset}, {a, k}});
          MatrixRef<T, D> e0_sub(e0, {{sub_offset, sub_offset}, {n_upper, k}});
          GEMM::callNN(T(1), e1_sub, e2_sub, T(0), e0_sub);
        }

        {
          MatrixRef<const T, D> e1_sub(e1, {{sub_offset + n_upper, sub_offset + n_uh}, {n_lower, b}});
          MatrixRef<const T, D> e2_sub(e2, {{sub_offset + n_uh, sub_offset}, {b, k}});
          MatrixRef<T, D> e0_sub(e0, {{sub_offset + n_upper, sub_offset}, {n_lower, k}});

          GEMM::callNN(T(1), e1_sub, e2_sub, T(0), e0_sub);
        }

        {
          const matrix::internal::SubMatrixSpec deflated_submat{{sub_offset, sub_offset + k},
                                                                {n, n - k}};
          MatrixRef<T, D> sub_e0(e0, deflated_submat);
          MatrixRef<const T, D> sub_e1(e1, deflated_submat);

          copy(sub_e1, sub_e0);
        }
      }));
}

template <Backend B, Device D, class T, class RhoSender>
void mergeSubproblems(const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                      RhoSender&& rho, WorkSpace<T, D>& ws, WorkSpaceHost<T>& ws_h,
                      WorkSpaceHostMirror<T, D>& ws_hm) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  using pika::execution::thread_priority;

  const GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  const LocalTileIndex idx_loc_begin(i_begin, i_begin);
  const SizeType nrtiles = i_end - i_begin;
  const LocalTileSize sz_loc_tiles(nrtiles, nrtiles);

  const LocalTileIndex idx_begin_tiles_vec(i_begin, 0);
  const LocalTileSize sz_tiles_vec(nrtiles, 1);

  const auto& dist = ws.e0.distribution();

  // Calculate the size of the upper, lower and full problem
  const SizeType n = problemSize(i_begin, i_end, dist);
  const SizeType n_upper = problemSize(i_begin, i_split, dist);
  const SizeType n_lower = problemSize(i_split, i_end, dist);

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
  addIndex(i_split, i_end, n_upper, ws_h.i1);

  // Step #1
  //
  //    i1 (in)  : initial <--- pre_sorted per sub-problem
  //    i2 (out) : initial <--- pre_sorted
  //
  // - deflate `d`, `z` and `c`
  // - apply Givens rotations to `Q` - `evecs`
  //
  sortIndex(i_begin, i_end, ex::just(n_upper), ws_h.d0, ws_h.i1, ws_hm.i2);

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
  auto [k_unique, n_udl] =
      stablePartitionIndexForDeflation(i_begin, i_end, ws_h.c, ws_h.d0, ws_hm.i2, ws_h.i3, ws_hm.i5) |
      ex::split_tuple();
  auto k = ex::split(std::move(k_unique));

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
  solveRank1Problem(i_begin, i_end, k, scaled_rho, ws_hm.d1, ws_hm.z1, ws_h.d0, ws_h.i4, ws_hm.e2);
  copy(idx_loc_begin, sz_loc_tiles, ws_hm.e2, ws.e2);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  const SizeType sub_offset = dist.template globalTileElementDistance<Coord::Row>(0, i_begin);

  multiplyEigenvectors<B>(sub_offset, n, n_upper, n_lower, ws.e0, ws.e1, ws.e2, k, std::move(n_udl));

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
void assembleDistZVec(comm::CommunicatorPipeline<comm::CommunicatorType::Full>& full_task_chain,
                      const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                      RhoSender&& rho, Matrix<const T, D>& evecs, Matrix<T, D>& z) {
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
      if (full_task_chain.size() > 1) {
        ex::start_detached(comm::schedule_bcast_send(full_task_chain.exclusive(), z.read(z_idx)));
      }
    }
    else {
      const comm::IndexT_MPI root_rank = full_task_chain.rank_full_communicator(evecs_tile_rank);
      ex::start_detached(comm::schedule_bcast_recv(full_task_chain.exclusive(), root_rank,
                                                   z.readwrite(z_idx)));
    }
  }
}

// It ensures that the internal sender, if set, will be waited for once this object will go out of scope.
// It might be useful in situation where you want to ensure that a sender representing a "checkpoint" for
// a computation is completed before leaving the scope (e.g. for related resource lifetime management)
//
// Currently it proved to be useful just in one place, and it was actually define nested where it was
// needed. Unfortunately, due to a problem with clang@15, we had to move it outside as a workaround.
//
// See https://github.com/eth-cscs/DLA-Future/issues/1017
struct ScopedSenderWait {
  pika::execution::experimental::unique_any_sender<> sender_;

  ~ScopedSenderWait() {
    if (sender_)
      pika::this_thread::experimental::sync_wait(std::move(sender_));
  }
};

template <class T, class CommSender, class KSender, class KLcSender, class RhoSender>
void solveRank1ProblemDist(CommSender&& row_comm, CommSender&& col_comm, const SizeType i_begin,
                           const SizeType i_end, KSender&& k, KLcSender&& k_lc, RhoSender&& rho,
                           Matrix<const T, Device::CPU>& d, Matrix<T, Device::CPU>& z,
                           Matrix<T, Device::CPU>& evals, Matrix<const SizeType, Device::CPU>& i4,
                           Matrix<const SizeType, Device::CPU>& i6,
                           Matrix<const SizeType, Device::CPU>& i2, Matrix<T, Device::CPU>& evecs) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  namespace tt = pika::this_thread::experimental;
  using pika::execution::thread_priority;

  const matrix::Distribution& dist = evecs.distribution();

  TileCollector tc{i_begin, i_end};

  const SizeType n = problemSize(i_begin, i_end, dist);

  namespace dist_extra = dlaf::matrix::internal::distribution;
  const matrix::Distribution dist_sub(
      dist, {{i_begin * dist.block_size().rows(), i_begin * dist.block_size().cols()},
             {dist_extra::global_tile_element_distance<Coord::Row>(dist, i_begin, i_end),
              dist_extra::global_tile_element_distance<Coord::Col>(dist, i_begin, i_end)}});

  auto bcast_evals = [i_begin, i_end,
                      dist](comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_comm_chain,
                            const std::vector<matrix::Tile<T, Device::CPU>>& eval_tiles) {
    using dlaf::comm::internal::sendBcast_o;
    using dlaf::comm::internal::recvBcast_o;

    const comm::Index2D this_rank = dist.rank_index();

    std::vector<ex::unique_any_sender<>> comms;
    comms.reserve(to_sizet(i_end - i_begin));

    for (SizeType i = i_begin; i < i_end; ++i) {
      const comm::IndexT_MPI evecs_tile_rank = dist.rank_global_tile<Coord::Col>(i);
      auto& tile = eval_tiles[to_sizet(i - i_begin)];

      if (evecs_tile_rank == this_rank.col())
        comms.emplace_back(ex::when_all(row_comm_chain.exclusive(), ex::just(std::cref(tile))) |
                           transformMPI(sendBcast_o));
      else
        comms.emplace_back(ex::when_all(row_comm_chain.exclusive(),
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

  const auto hp_scheduler = di::getBackendScheduler<Backend::MC>(thread_priority::high);
  ex::start_detached(
      ex::when_all(std::forward<CommSender>(row_comm), std::forward<CommSender>(col_comm),
                   std::forward<KSender>(k), std::forward<KLcSender>(k_lc), std::forward<RhoSender>(rho),
                   ex::when_all_vector(tc.read(d)), ex::when_all_vector(tc.readwrite(z)),
                   ex::when_all_vector(tc.readwrite(evals)), ex::when_all_vector(tc.read(i4)),
                   ex::when_all_vector(tc.read(i6)), ex::when_all_vector(tc.read(i2)),
                   ex::when_all_vector(tc.readwrite(evecs)),
                   // additional workspaces
                   ex::just(std::vector<memory::MemoryView<T, Device::CPU>>()),
                   ex::just(memory::MemoryView<T, Device::CPU>())) |
      ex::transfer(hp_scheduler) |
      ex::let_value([n, dist_sub, bcast_evals, all_reduce_in_place, hp_scheduler](
                        auto& row_comm_wrapper, auto& col_comm_wrapper, const SizeType k,
                        const SizeType k_lc, const auto& rho, const auto& d_tiles, auto& z_tiles,
                        const auto& eval_tiles, const auto& i4_tiles_arr, const auto& i6_tiles_arr,
                        const auto& i2_tiles_arr, const auto& evec_tiles, auto& ws_cols, auto& ws_row) {
        using pika::execution::thread_priority;

        const std::size_t nthreads = [dist_sub, k_lc] {
          const std::size_t workload = to_sizet(dist_sub.localSize().rows() * k_lc);
          const std::size_t workload_unit = 2 * to_sizet(dist_sub.tile_size().linear_size());

          const std::size_t min_workers = 1;
          const std::size_t available_workers = getTridiagRank1NWorkers();

          const std::size_t ideal_workers = util::ceilDiv(to_sizet(workload), workload_unit);
          return std::clamp(ideal_workers, min_workers, available_workers);
        }();

        return ex::just(std::make_unique<pika::barrier<>>(nthreads)) | ex::transfer(hp_scheduler) |
               ex::bulk(nthreads, [&row_comm_wrapper, &col_comm_wrapper, k, k_lc, &rho, &d_tiles,
                                   &z_tiles, &eval_tiles, &i4_tiles_arr, &i6_tiles_arr, &i2_tiles_arr,
                                   &evec_tiles, &ws_cols, &ws_row, nthreads, n, dist_sub, bcast_evals,
                                   all_reduce_in_place](const std::size_t thread_idx,
                                                        auto& barrier_ptr) {
                 using dlaf::comm::internal::transformMPI;

                 comm::CommunicatorPipeline<comm::CommunicatorType::Row> row_comm_chain(
                     row_comm_wrapper.get());
                 const dlaf::comm::Communicator& col_comm = col_comm_wrapper.get();

                 const SizeType m_lc = dist_sub.local_nr_tiles().rows();
                 const SizeType m_el_lc = dist_sub.local_size().rows();
                 const SizeType n_el_lc = dist_sub.local_size().cols();

                 const auto barrier_busy_wait = getTridiagRank1BarrierBusyWait();

                 const SizeType* i4 = i4_tiles_arr[0].get().ptr();
                 const SizeType* i2 = i2_tiles_arr[0].get().ptr();
                 const SizeType* i6 = i6_tiles_arr[0].get().ptr();

                 // STEP 0a: Permute eigenvalues for deflated eigenvectors (single-thread)
                 // Note: use last threads that in principle should have less work to do
                 if (k < n && thread_idx == nthreads - 1) {
                   const T* eval_initial_ptr = d_tiles[0].get().ptr();
                   T* eval_ptr = eval_tiles[0].ptr();

                   for (SizeType j_el_lc = k_lc; j_el_lc < n_el_lc; ++j_el_lc) {
                     const SizeType j_el = dist_sub.globalElementFromLocalElement<Coord::Col>(j_el_lc);
                     eval_ptr[j_el] = eval_initial_ptr[i6[j_el]];
                   }
                 }

                 const std::size_t batch_size = util::ceilDiv(to_sizet(k_lc), nthreads);
                 const SizeType begin = to_SizeType(thread_idx * batch_size);
                 const SizeType end = std::min(to_SizeType(thread_idx * batch_size + batch_size), k_lc);

                 // STEP 0b: Initialize workspaces (single-thread)
                 if (thread_idx == 0) {
                   // Note:
                   // - nthreads are used for both LAED4 and weight calculation (one per worker thread)
                   // - last one is used for reducing weights from all workers
                   ws_cols.reserve(nthreads + 1);

                   // Note:
                   // Considering that
                   // - LAED4 requires working on k elements
                   // - Weight computation requires working on m_el_lc
                   //
                   // and they are needed at two steps that cannot happen in parallel, we opted for allocating
                   // the workspace with the highest requirement of memory, and reuse them for both steps.
                   const SizeType max_size = std::max(k, m_el_lc);
                   for (std::size_t i = 0; i < nthreads; ++i)
                     ws_cols.emplace_back(max_size);
                   ws_cols.emplace_back(m_el_lc);

                   ws_row = memory::MemoryView<T, Device::CPU>(n_el_lc);
                   std::fill_n(ws_row(), n_el_lc, 0);
                 }

                 // Note: we have to wait that LAED4 workspaces are ready to be used
                 barrier_ptr->arrive_and_wait(barrier_busy_wait);

                 const T* d_ptr = d_tiles[0].get().ptr();
                 const T* z_ptr = z_tiles[0].ptr();

                 // STEP 1: LAED4 (multi-thread)
                 {
                   common::internal::SingleThreadedBlasScope single;

                   T* eval_ptr = eval_tiles[0].ptr();
                   T* delta_ptr = ws_cols[thread_idx]();

                   for (SizeType j_el_lc = begin; j_el_lc < end; ++j_el_lc) {
                     const SizeType j_el =
                         dist_sub.global_element_from_local_element<Coord::Col>(j_el_lc);
                     const SizeType j_lc = dist_sub.local_tile_from_local_element<Coord::Col>(j_el_lc);

                     // Solve the deflated rank-1 problem
                     // Note:
                     // Input eigenvalues are stored "deflated" with i3, but laed4 is going to store them
                     // "locally" deflated, i.e. locally it is valid sort(non-deflated)|sort(deflated)
                     const SizeType js_el = i6[j_el];
                     T& eigenval = eval_ptr[to_sizet(j_el)];
                     lapack::laed4(to_signed<int64_t>(k), to_signed<int64_t>(js_el), d_ptr, z_ptr,
                                   delta_ptr, rho, &eigenval);

                     // Now laed4 result has to be copied in the right spot
                     const SizeType j_el_tl =
                         dist_sub.tile_element_from_global_element<Coord::Col>(j_el);

                     for (SizeType i_lc = 0; i_lc < m_lc; ++i_lc) {
                       const SizeType i = dist_sub.global_tile_from_local_tile<Coord::Row>(i_lc);
                       const SizeType m_el_tl = dist_sub.tile_size_of<Coord::Row>(i);
                       const SizeType linear_lc =
                           dist_extra::local_tile_linear_index(dist_sub, {i_lc, j_lc});
                       const auto& evec = evec_tiles[to_sizet(linear_lc)];
                       for (SizeType i_el_tl = 0; i_el_tl < m_el_tl; ++i_el_tl) {
                         const SizeType i_el =
                             dist_sub.global_element_from_local_tile_and_tile_element<Coord::Row>(
                                 i_lc, i_el_tl);
                         DLAF_ASSERT_HEAVY(i_el < n, i_el, n);
                         const SizeType is_el = i4[i_el];

                         // just non-deflated, because deflated have been already set to 0
                         if (is_el < k)
                           evec({i_el_tl, j_el_tl}) = delta_ptr[is_el];
                       }
                     }
                   }
                 }
                 // Note: This barrier ensures that LAED4 finished, so from now on values are available
                 barrier_ptr->arrive_and_wait(barrier_busy_wait);

                 // STEP 2: Broadcast evals

                 // Note: this ensures that evals broadcasting finishes before bulk releases resources
                 ScopedSenderWait bcast_barrier;
                 if (thread_idx == 0 && row_comm_chain.size() > 1)
                   bcast_barrier.sender_ = bcast_evals(row_comm_chain, eval_tiles);

                 // Note: laed4 handles k <= 2 cases differently
                 if (k <= 2)
                   return;

                 // STEP 2 Compute weights (multi-thread)
                 auto& q = evec_tiles;
                 T* w = ws_cols[thread_idx]();

                 // STEP 2a: copy diagonal from q -> w (or just initialize with 1)
                 if (thread_idx == 0) {
                   for (SizeType i_el_lc = 0; i_el_lc < m_el_lc; ++i_el_lc) {
                     const SizeType i_el =
                         dist_sub.global_element_from_local_element<Coord::Row>(i_el_lc);
                     const SizeType is_el = i4[i_el];

                     if (is_el >= k) {
                       w[i_el_lc] = T{0};
                       continue;
                     }

                     const SizeType js_el = is_el;
                     const SizeType j_el = i2[js_el];

                     const GlobalElementIndex ij_subm_el(i_el, j_el);

                     if (dist_sub.rank_index().col() == dist_sub.rank_global_element<Coord::Col>(j_el)) {
                       const SizeType linear_subm_lc = dist_extra::local_tile_linear_index(
                           dist_sub, {dist_sub.local_tile_from_local_element<Coord::Row>(i_el_lc),
                                      dist_sub.local_tile_from_global_element<Coord::Col>(j_el)});
                       const TileElementIndex ij_tl = dist_sub.tile_element_index(ij_subm_el);
                       w[i_el_lc] = q[to_sizet(linear_subm_lc)](ij_tl);
                     }
                     else {
                       w[i_el_lc] = T{1};
                     }
                   }
                 }
                 else {  // other workers
                   std::fill_n(w, m_el_lc, T(1));
                 }

                 barrier_ptr->arrive_and_wait(barrier_busy_wait);

                 // STEP 2b: compute weights
                 {
                   for (SizeType j_el_lc = begin; j_el_lc < end; ++j_el_lc) {
                     const SizeType j_el =
                         dist_sub.global_element_from_local_element<Coord::Col>(j_el_lc);
                     const SizeType j_lc = dist_sub.local_tile_from_global_element<Coord::Col>(j_el);
                     const SizeType js_el = i6[j_el];
                     const T delta_j = d_ptr[to_sizet(js_el)];

                     const SizeType j_el_tl =
                         dist_sub.tile_element_from_local_element<Coord::Col>(j_el_lc);

                     for (SizeType i_lc = 0; i_lc < m_lc; ++i_lc) {
                       const SizeType i = dist_sub.global_tile_from_local_tile<Coord::Row>(i_lc);
                       const SizeType m_el_tl = dist_sub.tile_size_of<Coord::Row>(i);
                       const SizeType linear_lc =
                           dist_extra::local_tile_linear_index(dist_sub, {i_lc, j_lc});
                       const auto& q_tile = q[to_sizet(linear_lc)];

                       for (SizeType i_el_tl = 0; i_el_tl < m_el_tl; ++i_el_tl) {
                         const SizeType i_el =
                             dist_sub.global_element_from_global_tile_and_tile_element<Coord::Row>(
                                 i, i_el_tl);
                         DLAF_ASSERT_HEAVY(i_el < n, i_el, n);
                         const SizeType is_el = i4[i_el];

                         // skip if deflated
                         if (is_el >= k)
                           continue;

                         // skip if originally it was on the diagonal
                         if (is_el == js_el)
                           continue;

                         const SizeType i_el_lc =
                             dist_sub.local_element_from_local_tile_and_tile_element<Coord::Row>(
                                 i_lc, i_el_tl);
                         const TileElementIndex ij_tl(i_el_tl, j_el_tl);

                         w[i_el_lc] *= q_tile(ij_tl) / (d_ptr[to_sizet(is_el)] - delta_j);
                       }
                     }
                   }
                 }

                 barrier_ptr->arrive_and_wait(barrier_busy_wait);

                 // STEP 2c: reduce, then finalize computation with sign and square root (single-thread)
                 if (thread_idx == 0) {
                   // local reduction from all bulk workers
                   for (SizeType i_el_lc = 0; i_el_lc < m_el_lc; ++i_el_lc) {
                     for (std::size_t tidx = 1; tidx < nthreads; ++tidx) {
                       const T* w_partial = ws_cols[tidx]();
                       w[i_el_lc] *= w_partial[i_el_lc];
                     }
                   }

                   if (row_comm_chain.size() > 1) {
                     tt::sync_wait(ex::when_all(row_comm_chain.exclusive(),
                                                ex::just(MPI_PROD, common::make_data(w, m_el_lc))) |
                                   transformMPI(all_reduce_in_place));
                   }

#ifdef DLAF_ASSERT_HEAVY_ENABLE
                   // Note: all input for weights computation of non-deflated rows should be strictly less than 0
                   for (SizeType i_el_lc = 0; i_el_lc < m_el_lc; ++i_el_lc) {
                     const SizeType i_el =
                         dist_sub.global_element_from_local_element<Coord::Row>(i_el_lc);
                     const SizeType is = i4[i_el];
                     if (is < k)
                       DLAF_ASSERT_HEAVY(w[i_el_lc] < 0, w[i_el_lc]);
                   }
#endif

                   T* weights = ws_cols[nthreads]();
                   for (SizeType i_el_lc = 0; i_el_lc < m_el_lc; ++i_el_lc) {
                     const SizeType i_el =
                         dist_sub.global_element_from_local_element<Coord::Row>(i_el_lc);
                     const SizeType is_el = i4[i_el];
                     weights[to_sizet(i_el_lc)] =
                         std::copysign(std::sqrt(-w[i_el_lc]), z_ptr[to_sizet(is_el)]);
                   }
                 }

                 barrier_ptr->arrive_and_wait(barrier_busy_wait);

                 // STEP 3: Compute eigenvectors of the modified rank-1 modification (normalize) (multi-thread)

                 // STEP 3a: Form evecs using weights vector and compute (local) sum of squares
                 {
                   common::internal::SingleThreadedBlasScope single;

                   const T* w = ws_cols[nthreads]();
                   T* sum_squares = ws_row();

                   for (SizeType j_el_lc = begin; j_el_lc < end; ++j_el_lc) {
                     const SizeType j_lc = dist_sub.local_tile_from_local_element<Coord::Col>(j_el_lc);
                     const SizeType j_el_tl =
                         dist_sub.tile_element_from_local_element<Coord::Col>(j_el_lc);

                     for (SizeType i_lc = 0; i_lc < m_lc; ++i_lc) {
                       const SizeType i = dist_sub.global_tile_from_local_tile<Coord::Row>(i_lc);
                       const SizeType m_el_tl = dist_sub.tile_size_of<Coord::Row>(i);
                       const SizeType linear_lc =
                           dist_extra::local_tile_linear_index(dist_sub, {i_lc, j_lc});
                       const auto& q_tile = q[to_sizet(linear_lc)];

                       for (SizeType i_el_tl = 0; i_el_tl < m_el_tl; ++i_el_tl) {
                         const SizeType i_el =
                             dist_sub.global_element_from_global_tile_and_tile_element<Coord::Row>(
                                 i, i_el_tl);

                         DLAF_ASSERT_HEAVY(i_el < n, i_el, n);
                         const SizeType is_el = i4[i_el];

                         // it is a deflated row, skip it (it should be already 0)
                         if (is_el >= k)
                           continue;

                         const SizeType i_el_lc =
                             dist_sub.local_element_from_local_tile_and_tile_element<Coord::Row>(
                                 i_lc, i_el_tl);
                         const TileElementIndex ij_el_tl(i_el_tl, j_el_tl);

                         q_tile(ij_el_tl) = w[i_el_lc] / q_tile(ij_el_tl);
                       }

                       const T* partial_evec = q_tile.ptr({0, j_el_tl});
                       sum_squares[j_el_lc] += blas::dot(m_el_tl, partial_evec, 1, partial_evec, 1);
                     }
                   }
                 }

                 barrier_ptr->arrive_and_wait(barrier_busy_wait);

                 // STEP 3b: Reduce to get the sum of all squares on all ranks
                 if (thread_idx == 0 && col_comm.size() > 1)
                   tt::sync_wait(ex::just(std::cref(col_comm), MPI_SUM,
                                          common::make_data(ws_row(), k_lc)) |
                                 transformMPI(all_reduce_in_place));

                 barrier_ptr->arrive_and_wait(barrier_busy_wait);

                 // STEP 3c: Normalize (compute norm of each column and scale column vector)
                 {
                   common::internal::SingleThreadedBlasScope single;

                   const T* sum_squares = ws_row();

                   for (SizeType j_el_lc = begin; j_el_lc < end; ++j_el_lc) {
                     const SizeType j_lc = dist_sub.local_tile_from_local_element<Coord::Col>(j_el_lc);
                     const SizeType j_el_tl =
                         dist_sub.tile_element_from_local_element<Coord::Col>(j_el_lc);

                     const T vec_norm = std::sqrt(sum_squares[j_el_lc]);

                     for (SizeType i_lc = 0; i_lc < m_lc; ++i_lc) {
                       const LocalTileIndex ij_lc(i_lc, j_lc);
                       const SizeType ij_linear = dist_extra::local_tile_linear_index(dist_sub, ij_lc);

                       T* partial_evec = q[to_sizet(ij_linear)].ptr({0, j_el_tl});

                       const SizeType i = dist_sub.global_tile_from_local_tile<Coord::Row>(i_lc);
                       const SizeType m_el_tl = dist_sub.tile_size_of<Coord::Row>(i);
                       blas::scal(m_el_tl, 1 / vec_norm, partial_evec, 1);
                     }
                   }
                 }
               });
      }));
}

template <Backend B, class T, Device D, class KLcSender, class UDLSenders>
void multiplyEigenvectors(const GlobalElementIndex sub_offset, const matrix::Distribution& dist_sub,
                          comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                          comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain,
                          const SizeType n_upper, const SizeType n_lower, Matrix<T, D>& e0,
                          Matrix<T, D>& e1, Matrix<T, D>& e2, KLcSender&& k_lc, UDLSenders&& n_udl) {
  // Note:
  // This function computes E0 = E1 . E2
  //
  // where E1 is the matrix with eigenvectors and it looks like this
  //
  //               ┌──────────┐ k
  //               │    b     │ │
  //                            ▼
  //          ┌──  ┌───┬──────┬─┬────┐
  //          │    │UUU│DDDDDD│ │XXXX│
  //          │    │UUU│DDDDDD│ │XXXX│
  //  n_upper │    │UUU│DDDDDD│ │XXXX│
  //          │    │UUU│DDDDDD│ │XXXX│
  //          │    │UUU│DDDDDD│ │XXXX│
  //          ├──  ├───┼──────┼─┤XXXX│
  //          │    │   │DDDDDD│L│XXXX│
  //  n_lower │    │   │DDDDDD│L│XXXX│
  //          │    │   │DDDDDD│L│XXXX│
  //          └──  └───┴──────┴─┴────┘
  //               │ a │
  //               └───┘
  //               │      c     │
  //               └────────────┘
  //
  // Where (a, b, c) are the values from n_udl
  //
  // Note:
  // E1 matrix does not have all deflated values at the end, indeed part of them are "interlaced" with
  // others. The GEMM will perform anyway a computation for deflated eigenvectors (which are zeroed out)
  // while the copy step will be performed at "local" level, so even interlaced ones will get copied
  // in the right spot.
  //
  // The multiplication in two different steps in order to skip zero blocks of the matrix, created by
  // the grouping of eigenvectors of different lengths (UPPER, DENSE and LOWER).
  //
  // 1. GEMM1 = TL . TOP
  // 2. GEMM2 = BR . BOTTOM
  // 3. copy DEFLATED
  //
  //                      ┌────────────┬────┐
  //                      │            │    │
  //                      │            │    │
  //                      │   T O P    │    │
  //                      │            │    │
  //                      │            │    │
  //                      ├────────────┤    │
  //                      │            │    │
  //                      │            │    │
  //                      │B O T T O M │    │
  //                      │            │    │
  //                      └────────────┴────┘
  //
  // ┌──────────┬─┬────┐  ┌────────────┬────┐
  // │          │0│    │  │            │    │
  // │          │0│ D  │  │            │    │
  // │   TL     │0│ E  │  │  GEMM 1    │ C  │
  // │          │0│ F  │  │            │    │
  // │          │0│ L  │  │            │ O  │
  // ├───┬──────┴─┤ A  │  ├────────────┤    │
  // │000│        │ T  │  │            │ P  │
  // │000│        │ E  │  │            │    │
  // │000│  BR    │ D  │  │  GEMM 2    │ Y  │
  // │000│        │    │  │            │    │
  // └───┴────────┴────┘  └────────────┴────┘

  namespace ex = pika::execution::experimental;
  using pika::execution::thread_priority;

  ex::start_detached(
      ex::when_all(std::forward<KLcSender>(k_lc), std::forward<UDLSenders>(n_udl)) |
      ex::transfer(dlaf::internal::getBackendScheduler<Backend::MC>(thread_priority::high)) |
      ex::then([dist_sub, sub_offset, n_upper, n_lower, e0 = e0.subPipeline(),
                e1 = e1.subPipelineConst(), e2 = e2.subPipelineConst(),
                sub_comm_row = row_task_chain.sub_pipeline(),
                sub_comm_col = col_task_chain.sub_pipeline()](
                   const SizeType k_lc, const std::array<SizeType, 3>& n_udl) mutable {
        using dlaf::matrix::internal::MatrixRef;

        const SizeType n = dist_sub.size().cols();
        const auto [a, b, c] = n_udl;

        using GEMM = dlaf::multiplication::internal::General<B, D, T>;
        {
          MatrixRef<const T, D> e1_sub(e1, {sub_offset, {n_upper, b}});
          MatrixRef<const T, D> e2_sub(e2, {sub_offset, {b, c}});
          MatrixRef<T, D> e0_sub(e0, {sub_offset, {n_upper, c}});

          GEMM::callNN(sub_comm_row, sub_comm_col, T(1), e1_sub, e2_sub, T(0), e0_sub);
        }

        {
          MatrixRef<const T, D> e1_sub(e1, {{sub_offset.row() + n_upper, sub_offset.col() + a},
                                            {n_lower, c - a}});
          MatrixRef<const T, D> e2_sub(e2, {{sub_offset.row() + a, sub_offset.col()}, {c - a, c}});
          MatrixRef<T, D> e0_sub(e0, {{sub_offset.row() + n_upper, sub_offset.col()}, {n_lower, c}});

          GEMM::callNN(sub_comm_row, sub_comm_col, T(1), e1_sub, e2_sub, T(0), e0_sub);
        }

        if (k_lc < dist_sub.local_size().cols()) {
          const SizeType k = dist_sub.global_element_from_local_element<Coord::Col>(k_lc);
          const matrix::internal::SubMatrixSpec deflated_submat{{sub_offset.row(), sub_offset.col() + k},
                                                                {n, n - k}};
          MatrixRef<T, D> sub_e0(e0, deflated_submat);
          MatrixRef<const T, D> sub_e1(e1, deflated_submat);

          copy(sub_e1, sub_e0);
        }
      }));
}

// Distributed version of the tridiagonal solver on CPUs
template <Backend B, class T, Device D, class RhoSender>
void mergeDistSubproblems(comm::CommunicatorPipeline<comm::CommunicatorType::Full>& full_task_chain,
                          comm::CommunicatorPipeline<comm::CommunicatorType::Row>& row_task_chain,
                          comm::CommunicatorPipeline<comm::CommunicatorType::Col>& col_task_chain,
                          const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                          RhoSender&& rho, WorkSpace<T, D>& ws, WorkSpaceHost<T>& ws_h,
                          DistWorkSpaceHostMirror<T, D>& ws_hm) {
  namespace ex = pika::execution::experimental;
  using matrix::internal::distribution::global_tile_element_distance;
  using pika::execution::thread_priority;

  const matrix::Distribution& dist = ws.e0.distribution();

  const GlobalElementIndex sub_offset{i_begin * dist.tile_size().rows(),
                                      i_begin * dist.tile_size().cols()};
  const GlobalElementSize sub_size{
      global_tile_element_distance<Coord::Row>(dist, i_begin, i_end),
      global_tile_element_distance<Coord::Col>(dist, i_begin, i_end),
  };
  const matrix::Distribution dist_sub(dist, {sub_offset, sub_size});

  // Calculate the size of the upper subproblem
  const SizeType n_upper = global_tile_element_distance<Coord::Row>(dist, i_begin, i_split);
  const SizeType n_lower = global_tile_element_distance<Coord::Row>(dist, i_split, i_end);

  // The local size of the subproblem
  const GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  const LocalTileIndex idx_loc_begin{dist.next_local_tile_from_global_tile<Coord::Row>(i_begin),
                                     dist.next_local_tile_from_global_tile<Coord::Col>(i_begin)};
  const LocalTileIndex idx_loc_end{dist.next_local_tile_from_global_tile<Coord::Row>(i_end),
                                   dist.next_local_tile_from_global_tile<Coord::Col>(i_end)};
  const LocalTileSize sz_loc_tiles = idx_loc_end - idx_loc_begin;
  const LocalTileIndex idx_begin_tiles_vec(i_begin, 0);
  const LocalTileSize sz_tiles_vec(i_end - i_begin, 1);

  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleDistZVec(full_task_chain, i_begin, i_split, i_end, rho, ws.e0, ws.z0);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws.z0, ws_hm.z0);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  auto scaled_rho = scaleRho(std::move(rho)) | ex::split();

  // Calculate the tolerance used for deflation
  auto tol = calcTolerance(i_begin, i_end, ws_h.d0, ws_hm.z0);

  // Initialize the column types vector `c`
  initColTypes(i_begin, i_split, i_end, ws_h.c);

  // Initialize i1 as identity just for single tile sub-problems
  if (i_split == i_begin + 1) {
    initIndex(i_begin, i_split, ws_h.i1);
  }
  if (i_split + 1 == i_end) {
    initIndex(i_split, i_end, ws_h.i1);
  }

  // Update indices of second sub-problem
  addIndex(i_split, i_end, n_upper, ws_h.i1);

  // Step #1
  //
  //    i1 (out) : initial <--- initial (or identity map)
  //    i2 (out) : initial <--- pre_sorted
  //
  // - deflate `d`, `z` and `c`
  // - apply Givens rotations to `Q` - `evecs`
  //
  sortIndex(i_begin, i_end, ex::just(n_upper), ws_h.d0, ws_h.i1, ws_hm.i2);

  auto rots =
      applyDeflation(i_begin, i_end, scaled_rho, std::move(tol), ws_hm.i2, ws_h.d0, ws_hm.z0, ws_h.c);

  // Make sure Isend/Irecv messages don't match between calls by providing a unique `tag`
  //
  // Note: i_split is unique
  const comm::IndexT_MPI tag = to_int(i_split);
  applyGivensRotationsToMatrixColumns(row_task_chain, tag, i_begin, i_end, std::move(rots), ws.e0);

  // Step #2
  //
  //    i2 (in)  : initial         <--- pre_sorted
  //
  //    i3 (out) : initial         <--- deflated
  //    i5 (out) : initial         <--- local(UDL|X)
  //    i4 (out) : deflated        <--- sort by col type
  //    i6 (out) : deflated        <--- local(deflated)
  //    i2 (out) : local(deflated) <--- deflated
  //
  // - reorder eigenvectors locally so that they are well-shaped for gemm optimization (i.e. UDLX)
  // - reorder `d0 -> d1`, `z0 -> z1`, using `i3` such that deflated entries are at the bottom.
  // - solve the rank-1 problem and save eigenvalues in `d0` and `d1` (copy) and eigenvectors in `e2`.
  // - set deflated diagonal entries of `U` to 1 (temporary solution until optimized GEMM is implemented)
  auto [k_unique, k_lc_unique, n_udl] =
      ex::split_tuple(stablePartitionIndexForDeflation(dist, i_begin, i_end, ws_h.c, ws_h.d0, ws_hm.i2,
                                                       ws_h.i3, ws_hm.i5, ws_hm.i5b, ws_h.i4, ws_hm.i6));

  auto k = ex::split(std::move(k_unique));
  auto k_lc = ex::split(std::move(k_lc_unique));

  // Reorder Eigenvectors
  copy(LocalTileIndex(idx_loc_begin.col(), 0), LocalTileSize(idx_loc_end.col() - idx_loc_begin.col(), 1),
       ws_hm.i5b, ws.i5b);
  permutations::permute<B, D, T, Coord::Col>(i_begin, i_end, ws.i5b, ws.e0, ws.e1);

  // Reorder Eigenvalues
  applyIndex(i_begin, i_end, ws_h.i3, ws_h.d0, ws_hm.d1);
  applyIndex(i_begin, i_end, ws_h.i3, ws_hm.z0, ws_hm.z1);

  // Note:
  // set0 is required because deflated eigenvectors rows won't be touched in rank1 and so they will be
  // neutral when used in GEMM (copy will take care of them later)
  matrix::util::set0<Backend::MC>(thread_priority::normal, idx_loc_begin, sz_loc_tiles, ws_hm.e2);
  solveRank1ProblemDist(row_task_chain.exclusive(), col_task_chain.exclusive(), i_begin, i_end, k, k_lc,
                        std::move(scaled_rho), ws_hm.d1, ws_hm.z1, ws_h.d0, ws_h.i4, ws_hm.i6, ws_hm.i2,
                        ws_hm.e2);
  copy(idx_loc_begin, sz_loc_tiles, ws_hm.e2, ws.e2);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  multiplyEigenvectors<B>(sub_offset, dist_sub, row_task_chain, col_task_chain, n_upper, n_lower, ws.e0,
                          ws.e1, ws.e2, std::move(k_lc), std::move(n_udl));

  // Step #4: Final permutation to sort eigenvalues and eigenvectors
  //
  //    i2 (in)  : local(deflated) <--- deflated
  //    i1 (out) : sorted          <--- local(deflated)
  sortIndex(i_begin, i_end, std::move(k), ws_h.d0, ws_hm.i2, ws_h.i1);
}
}
