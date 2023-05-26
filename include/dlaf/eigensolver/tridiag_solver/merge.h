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

#include <pika/algorithm.hpp>
#include <pika/barrier.hpp>
#include <pika/execution.hpp>

#include "dlaf/common/range2d.h"
#include "dlaf/common/single_threaded_blas.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/eigensolver/get_tridiag_rank1_nworkers.h"
#include "dlaf/eigensolver/tridiag_solver/coltype.h"
#include "dlaf/eigensolver/tridiag_solver/index_manipulation.h"
#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/eigensolver/tridiag_solver/rot.h"
#include "dlaf/eigensolver/tridiag_solver/tile_collector.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/memory/memory_view.h"
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
// 3. Sort index based on updated diagonal values in ascending order. The diagonal conatins eigenvalues
//    of the deflated problem and deflated entreis from the initial diagonal
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
};

template <class T>
struct WorkSpaceHost {
  Matrix<T, Device::CPU> d0;

  Matrix<ColType, Device::CPU> c;

  Matrix<SizeType, Device::CPU> i1;
  Matrix<SizeType, Device::CPU> i3;
};

// forward declaration : Device::GPU - unused
template <class T, Device D>
struct WorkSpaceHostMirror {
  Matrix<T, Device::CPU> e2;

  Matrix<T, Device::CPU> d1;

  Matrix<T, Device::CPU> z0;
  Matrix<T, Device::CPU> z1;

  Matrix<SizeType, Device::CPU> i2;
};

template <class T>
struct WorkSpaceHostMirror<T, Device::CPU> {
  Matrix<T, Device::CPU>& e2;

  Matrix<T, Device::CPU>& d1;

  Matrix<T, Device::CPU>& z0;
  Matrix<T, Device::CPU>& z1;

  Matrix<SizeType, Device::CPU>& i2;
};

// forward declaration : Device::GPU - unused
template <class T, Device D>
struct DistWorkSpaceHostMirror {
  Matrix<T, Device::CPU> e0;
  Matrix<T, Device::CPU> e2;

  Matrix<T, Device::CPU> d1;

  Matrix<T, Device::CPU> z0;
  Matrix<T, Device::CPU> z1;

  Matrix<SizeType, Device::CPU> i2;
};

template <class T>
struct DistWorkSpaceHostMirror<T, Device::CPU> {
  Matrix<T, Device::CPU>& e0;
  Matrix<T, Device::CPU>& e2;

  Matrix<T, Device::CPU>& d1;

  Matrix<T, Device::CPU>& z0;
  Matrix<T, Device::CPU>& z1;

  Matrix<SizeType, Device::CPU>& i2;
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
template <class T, Device D>
auto maxVectorElement(const SizeType i_begin, const SizeType i_end, Matrix<const T, D>& vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  std::vector<ex::unique_any_sender<T>> tiles_max;
  tiles_max.reserve(to_sizet(i_end - i_begin));
  for (SizeType i = i_begin; i < i_end; ++i) {
    tiles_max.push_back(maxElementInColumnTileAsync<T, D>(vec.read(LocalTileIndex(i, 0))));
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
auto calcTolerance(const SizeType i_begin, const SizeType i_end, Matrix<const T, D>& d,
                   Matrix<const T, D>& z) {
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

// The index array `out_ptr` holds the indices of elements of `c_ptr` that order it such that
// ColType::Deflated entries are moved to the end. The `c_ptr` array is implicitly ordered according to
// `in_ptr` on entry.
//
inline SizeType stablePartitionIndexForDeflationArrays(const SizeType n, const ColType* c_ptr,
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
    const SizeType ii = in_ptr[i];
    SizeType& io = (c_ptr[ii] != ColType::Deflated) ? i1 : i2;
    out_ptr[io] = ii;
    ++io;
  }
  return k;
}

template <Device D>
auto stablePartitionIndexForDeflation(const SizeType i_begin, const SizeType i_end,
                                      Matrix<const ColType, D>& c, Matrix<const SizeType, D>& in,
                                      Matrix<SizeType, D>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  constexpr auto backend = dlaf::DefaultBackend_v<D>;

  const SizeType n = problemSize(i_begin, i_end, in.distribution());
  if constexpr (D == Device::CPU) {
    auto part_fn = [n](const auto& c_tiles_futs, const auto& in_tiles_futs, const auto& out_tiles) {
      const TileElementIndex zero_idx(0, 0);
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
      const TileElementIndex zero_idx(0, 0);
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
void initColTypes(const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                  Matrix<ColType, D>& coltypes) {
  for (SizeType i = i_begin; i < i_end; ++i) {
    ColType val = (i < i_split) ? ColType::UpperHalf : ColType::LowerHalf;
    setColTypeTileAsync<D>(val, coltypes.readwrite(LocalTileIndex(i, 0)));
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
                       Matrix<T, Device::CPU>& evals, Matrix<T, Device::CPU>& evecs) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, evals.distribution());
  const SizeType nb = evals.distribution().blockSize().rows();

  TileCollector tc{i_begin, i_end};

  const std::size_t nthreads = getTridiagRank1NWorkers();
  ex::start_detached(
      ex::when_all(ex::just(std::make_shared<pika::barrier<>>(nthreads)), std::forward<KSender>(k),
                   std::forward<RhoSender>(rho), ex::when_all_vector(tc.read(d)),
                   ex::when_all_vector(tc.readwrite(z)), ex::when_all_vector(tc.readwrite(evals)),
                   ex::when_all_vector(tc.readwrite(evecs)),
                   ex::just(std::vector<memory::MemoryView<T, Device::CPU>>())) |
      ex::transfer(di::getBackendScheduler<Backend::MC>(pika::execution::thread_priority::high)) |
      ex::bulk(nthreads, [nthreads, n, nb](std::size_t thread_idx, auto& barrier_ptr, auto& k, auto& rho,
                                           auto& d_tiles_futs, auto& z_tiles, auto& eval_tiles,
                                           auto& evec_tiles, auto& ws_vecs) {
        common::internal::SingleThreadedBlasScope single;

        const matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

        const std::size_t batch_size = util::ceilDiv(to_sizet(k), nthreads);
        const std::size_t begin = thread_idx * batch_size;
        const std::size_t end = std::min(thread_idx * batch_size + batch_size, to_sizet(k));

        // STEP 0: Initialize workspaces (single-thread)
        if (thread_idx == 0) {
          ws_vecs.reserve(nthreads);
          for (std::size_t i = 0; i < nthreads; ++i)
            ws_vecs.emplace_back(to_sizet(k));
        }

        barrier_ptr->arrive_and_wait();

        // STEP 1: LAED4 (multi-thread)
        const T* d_ptr = d_tiles_futs[0].get().ptr();
        const T* z_ptr = z_tiles[0].ptr();
        T* eval_ptr = eval_tiles[0].ptr();

        for (std::size_t i = begin; i < end; ++i) {
          T& eigenval = eval_ptr[i];

          const SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, to_SizeType(i)));
          const SizeType i_col = distr.tileElementFromGlobalElement<Coord::Col>(to_SizeType(i));
          T* delta = evec_tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_col));

          lapack::laed4(to_int(k), to_int(i), d_ptr, z_ptr, delta, rho, &eigenval);
        }

        barrier_ptr->arrive_and_wait();

        // STEP 2a Compute weights
        auto& q = evec_tiles;
        T* w = ws_vecs[thread_idx]();

        // - copy diagonal from q -> w (or just initialize with 1)
        if (thread_idx == 0) {
          for (auto i = 0; i < k; ++i) {
            const GlobalElementIndex kk(i, i);
            const auto diag_tile = distr.globalTileLinearIndex(kk);
            const auto diag_element = distr.tileElementIndex(kk);

            w[i] = q[to_sizet(diag_tile)](diag_element);
          }
        }
        else {
          std::fill_n(w, k, T(1));
        }

        // - compute productorial (thread-local)
        auto compute_w = [&](const GlobalElementIndex ij) {
          const auto q_tile = distr.globalTileLinearIndex(ij);
          const auto q_ij = distr.tileElementIndex(ij);

          const SizeType i = ij.row();
          const SizeType j = ij.col();

          w[i] *= q[to_sizet(q_tile)](q_ij) / (d_ptr[to_sizet(i)] - d_ptr[to_sizet(j)]);
        };

        for (auto j = to_SizeType(begin); j < to_SizeType(end); ++j) {
          for (auto i = 0; i < j; ++i)
            compute_w({i, j});

          for (auto i = j + 1; i < k; ++i)
            compute_w({i, j});
        }

        barrier_ptr->arrive_and_wait();

        // STEP 2B: reduce, then finalize computation with sign and square root (single-thread)
        if (thread_idx == 0) {
          for (int i = 0; i < k; ++i) {
            for (std::size_t tidx = 1; tidx < nthreads; ++tidx) {
              const T* w_partial = ws_vecs[tidx]();
              w[i] *= w_partial[i];
            }
            z_tiles[0].ptr()[i] = std::copysign(std::sqrt(-w[i]), z_ptr[to_sizet(i)]);
          }
        }

        barrier_ptr->arrive_and_wait();

        // STEP 3: Compute eigenvectors of the modified rank-1 modification (normalize) (multi-thread)
        {
          const T* w = z_ptr;
          T* s = ws_vecs[thread_idx]();

          for (auto j = to_SizeType(begin); j < to_SizeType(end); ++j) {
            for (int i = 0; i < k; ++i) {
              const auto q_tile = distr.globalTileLinearIndex({i, j});
              const auto q_ij = distr.tileElementIndex({i, j});

              s[i] = w[i] / q[to_sizet(q_tile)](q_ij);
            }

            const T vec_norm = blas::nrm2(k, s, 1);

            for (auto i = 0; i < k; ++i) {
              const auto q_tile = distr.globalTileLinearIndex({i, j});
              const auto q_ij = distr.tileElementIndex({i, j});

              q[to_sizet(q_tile)](q_ij) = s[i] / vec_norm;
            }
          }
        }
      }));
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
template <class T, Device D, class KRSender, class KCSender>
void initWeightVector(const GlobalTileIndex idx_gl_begin, const LocalTileIndex idx_loc_begin,
                      const LocalTileSize sz_loc_tiles, KRSender&& k_row, KCSender&& k_col,
                      Matrix<const T, D>& diag, Matrix<const T, D>& diag_i2, Matrix<const T, D>& evecs,
                      Matrix<T, D>& ws) {
  const matrix::Distribution& dist = evecs.distribution();

  // Reduce by multiplication into the first local column of each tile of the workspace matrix `ws`
  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    // Divide the eigenvectors of the rank1 update problem `evecs` by it's diagonal matrix `diag` and
    // reduce multiply into the first column of each tile of the workspace matrix `ws`
    divideEvecsByDiagonalAsync<D>(k_row, k_col, sz_gl_el.rows(), sz_gl_el.cols(),
                                  diag_i2.read(GlobalTileIndex(idx_gl_tile.row(), 0)),
                                  diag.read(GlobalTileIndex(idx_gl_tile.col(), 0)),
                                  evecs.read(idx_loc_tile), ws.readwrite(idx_loc_tile));

    // skip the first local column
    if (idx_loc_tile.col() == idx_loc_begin.col())
      continue;

    // reduce-multiply the first column of each local tile of the workspace matrix into the first local
    // column of the matrix
    const LocalTileIndex idx_ws_first_col_tile(idx_loc_tile.row(), idx_loc_begin.col());
    multiplyFirstColumnsAsync<D>(k_row, k_col, sz_gl_el.rows(), sz_gl_el.cols(), ws.read(idx_loc_tile),
                                 ws.readwrite(idx_ws_first_col_tile));
  }
}

// References:
// - lapack 3.10.0, dlaed3.f, line 293
// - LAPACK Working Notes: lawn132, Parallelizing the Divide and Conquer Algorithm for the Symmetric
//   Tridiagonal Eigenvalue Problem on Distributed Memory Architectures, 4.2 Orthogonality
template <class T, Device D, class KRSender, class KCSender>
void formEvecsUsingWeightVec(const GlobalTileIndex idx_gl_begin, const LocalTileIndex idx_loc_begin,
                             const LocalTileSize sz_loc_tiles, KRSender&& k_row, KCSender&& k_col,
                             Matrix<const T, D>& z, Matrix<const T, D>& ws, Matrix<T, D>& evecs) {
  const matrix::Distribution& dist = evecs.distribution();

  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    const LocalTileIndex idx_ws_first_local_column(idx_loc_tile.row(), idx_loc_begin.col());

    calcEvecsFromWeightVecAsync<D>(k_row, k_col, sz_gl_el.rows(), sz_gl_el.cols(),
                                   z.read(GlobalTileIndex(idx_gl_tile.row(), 0)),
                                   ws.read(idx_ws_first_local_column), evecs.readwrite(idx_loc_tile));
  }
}

// Sum of squares of columns of @p evecs into the first row of @p ws
template <class T, Device D, class KRSender, class KCSender>
void sumsqEvecs(const GlobalTileIndex idx_gl_begin, const LocalTileIndex idx_loc_begin,
                LocalTileSize sz_loc_tiles, KRSender&& k_row, KCSender&& k_col,
                Matrix<const T, D>& evecs, Matrix<T, D>& ws) {
  const matrix::Distribution& dist = evecs.distribution();

  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    sumsqColsAsync<D>(k_row, k_col, sz_gl_el.rows(), sz_gl_el.cols(), evecs.read(idx_loc_tile),
                      ws.readwrite(idx_loc_tile));

    // skip the first local row
    if (idx_loc_tile.row() == idx_loc_begin.row())
      continue;

    const LocalTileIndex idx_ws_first_row_tile(idx_loc_begin.row(), idx_loc_tile.col());
    addFirstRowsAsync<D>(k_row, k_col, sz_gl_el.rows(), sz_gl_el.cols(), ws.read(idx_loc_tile),
                         ws.readwrite(idx_ws_first_row_tile));
  }
}

// Normalize column vectors
template <class T, Device D, class KRSender, class KCSender>
void normalizeEvecs(const GlobalTileIndex idx_gl_begin, const LocalTileIndex idx_loc_begin,
                    const LocalTileSize sz_loc_tiles, KRSender&& k_row, KCSender&& k_col,
                    Matrix<const T, D>& ws, Matrix<T, D>& evecs) {
  const matrix::Distribution& dist = evecs.distribution();

  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_tiles)) {
    auto idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    auto sz_gl_el = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    const LocalTileIndex idx_ws_first_local_row(idx_loc_begin.row(), idx_loc_tile.col());
    divideColsByFirstRowAsync<D>(k_row, k_col, sz_gl_el.rows(), sz_gl_el.cols(),
                                 ws.read(idx_ws_first_local_row), evecs.readwrite(idx_loc_tile));
  }
}

template <class T, Device D, class KSender>
void setUnitDiag(const SizeType i_begin, const SizeType i_end, KSender&& k, Matrix<T, D>& mat) {
  // Iterate over diagonal tiles
  const matrix::Distribution& distr = mat.distribution();
  for (SizeType i_tile = i_begin; i_tile < i_end; ++i_tile) {
    const SizeType tile_begin = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);

    setUnitDiagonalAsync<D>(k, tile_begin, mat.readwrite(GlobalTileIndex(i_tile, i_tile)));
  }
}

template <Backend B, Device D, class T, class RhoSender>
void mergeSubproblems(const SizeType i_begin, const SizeType i_split, const SizeType i_end,
                      RhoSender&& rho, WorkSpace<T, D>& ws, WorkSpaceHost<T>& ws_h,
                      WorkSpaceHostMirror<T, D>& ws_hm) {
  namespace ex = pika::execution::experimental;

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

  // ---

  applyGivensRotationsToMatrixColumns(i_begin, i_end, std::move(rots), ws.e0);
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
  auto k = stablePartitionIndexForDeflation(i_begin, i_end, ws_h.c, ws_hm.i2, ws_h.i3) | ex::split();

  applyIndex(i_begin, i_end, ws_h.i3, ws_h.d0, ws_hm.d1);
  applyIndex(i_begin, i_end, ws_h.i3, ws_hm.z0, ws_hm.z1);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.d1, ws_h.d0);

  //
  //    i3 (in)  : initial <--- deflated
  //    i2 (out) : initial ---> deflated
  //
  invertIndex(i_begin, i_end, ws_h.i3, ws_hm.i2);

  matrix::util::set0<Backend::MC>(pika::execution::thread_priority::normal, idx_loc_begin, sz_loc_tiles,
                                  ws_hm.e2);
  solveRank1Problem(i_begin, i_end, k, scaled_rho, ws_hm.d1, ws_hm.z1, ws_h.d0, ws_hm.e2);

  copy(idx_loc_begin, sz_loc_tiles, ws_hm.e2, ws.e2);

  setUnitDiag(i_begin, i_end, k, ws.e2);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  //
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.i2, ws.i2);
  // The following permutation will be removed in the future.
  // (The copy is needed to simplify the removal)
  dlaf::permutations::permute<B, D, T, Coord::Row>(i_begin, i_end, ws.i2, ws.e2, ws.e0);
  copy(idx_loc_begin, sz_loc_tiles, ws.e0, ws.e2);
  dlaf::multiplication::generalSubMatrix<B, D, T>(i_begin, i_end, blas::Op::NoTrans, blas::Op::NoTrans,
                                                  T(1), ws.e1, ws.e2, T(0), ws.e0);

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
void assembleDistZVec(comm::CommunicatorGrid grid, common::Pipeline<comm::Communicator>& full_task_chain,
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
      ex::start_detached(comm::scheduleSendBcast(full_task_chain(), z.read(z_idx)));
    }
    else {
      const comm::IndexT_MPI root_rank = grid.rankFullCommunicator(evecs_tile_rank);
      ex::start_detached(comm::scheduleRecvBcast(full_task_chain(), root_rank, z.readwrite(z_idx)));
    }
  }
}

// (All)Reduce-multiply row-wise the first local columns of the distributed matrix @p mat of size (n, n).
// The local column on each rank is first offloaded to a local communication buffer @p comm_vec of size (n, 1).
template <class T, Device D, class KRSender, class KCSender>
void reduceMultiplyWeightVector(common::Pipeline<comm::Communicator>& row_task_chain,
                                const SizeType i_begin, const SizeType i_end,
                                const LocalTileIndex idx_loc_begin, LocalTileSize sz_loc_tiles,
                                KRSender&& k_row, KCSender&& k_col, Matrix<T, D>& mat,
                                Matrix<T, D>& comm_vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const matrix::Distribution& dist = mat.distribution();

  // If the rank doesn't have local matrix tiles, participate in the AllReduce() call by sending tiles
  // from the communication buffer filled with `1`.
  if (sz_loc_tiles.isEmpty()) {
    comm::Index2D this_rank = dist.rankIndex();
    for (SizeType i_tile = i_begin; i_tile < i_end; ++i_tile) {
      if (this_rank.row() == dist.rankGlobalTile<Coord::Row>(i_tile)) {
        const GlobalTileIndex idx_gl_comm(i_tile, 0);
        auto laset_sender =
            di::whenAllLift(blas::Uplo::General, T(1), T(1), comm_vec.readwrite(idx_gl_comm));
        ex::start_detached(tile::laset(di::Policy<DefaultBackend_v<D>>(), std::move(laset_sender)));
        ex::start_detached(
            comm::scheduleAllReduceInPlace(row_task_chain(), MPI_PROD, comm_vec.readwrite(idx_gl_comm)));
      }
    }
    return;
  }

  const GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  const LocalTileSize sz_loc_first_column(sz_loc_tiles.rows(), 1);
  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_loc_first_column)) {
    const GlobalTileIndex idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    const GlobalElementSize sz_subm = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    const GlobalTileIndex idx_gl_comm(idx_gl_tile.row(), 0);

    // set buffer to 1
    ex::start_detached(
        di::whenAllLift(blas::Uplo::General, T(1), T(1), comm_vec.readwrite(idx_gl_comm)) |
        tile::laset(di::Policy<DefaultBackend_v<D>>()));

    // copy the first column of the matrix tile into the column tile of the buffer
    copy1DAsync<D>(k_row, k_col, sz_subm.rows(), sz_subm.cols(), Coord::Col, mat.read(idx_loc_tile),
                   Coord::Col, comm_vec.readwrite(idx_gl_comm));

    ex::start_detached(
        comm::scheduleAllReduceInPlace(row_task_chain(), MPI_PROD, comm_vec.readwrite(idx_gl_comm)));

    // copy the column tile of the buffer into the first column of the matrix tile
    copy1DAsync<D>(k_row, k_col, sz_subm.rows(), sz_subm.cols(), Coord::Col, comm_vec.read(idx_gl_comm),
                   Coord::Col, mat.readwrite(idx_loc_tile));
  }
}

// (All)Reduce-sum column-wise the first local rows of the distributed matrix @p mat of size (n, n).
// The local row on each rank is first offloaded to a local communication buffer @p comm_vec of size (n, 1).
template <class T, Device D, class KRSender, class KCSender>
void reduceSumScalingVector(common::Pipeline<comm::Communicator>& col_task_chain, const SizeType i_begin,
                            const SizeType i_end, const LocalTileIndex idx_loc_begin,
                            LocalTileSize sz_loc_tiles, KRSender&& k_row, KCSender&& k_col,
                            Matrix<T, D>& mat, Matrix<T, D>& comm_vec) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const matrix::Distribution& dist = mat.distribution();

  // If the rank doesn't have local matrix tiles, participate in the AllReduce() call by sending tiles
  // from the communication buffer filled with `0`.
  if (sz_loc_tiles.isEmpty()) {
    comm::Index2D this_rank = dist.rankIndex();
    for (SizeType i_tile = i_begin; i_tile < i_end; ++i_tile) {
      if (this_rank.col() == dist.rankGlobalTile<Coord::Col>(i_tile)) {
        const GlobalTileIndex idx_gl_comm(i_tile, 0);
        ex::start_detached(
            tile::set0(di::Policy<DefaultBackend_v<D>>(), comm_vec.readwrite(idx_gl_comm)));
        ex::start_detached(
            comm::scheduleAllReduceInPlace(col_task_chain(), MPI_SUM, comm_vec.readwrite(idx_gl_comm)));
      }
    }
    return;
  }

  const GlobalTileIndex idx_gl_begin(i_begin, i_begin);
  const LocalTileSize sz_first_local_row(1, sz_loc_tiles.cols());
  for (auto idx_loc_tile : common::iterate_range2d(idx_loc_begin, sz_first_local_row)) {
    const GlobalTileIndex idx_gl_tile = dist.globalTileIndex(idx_loc_tile);
    const GlobalElementSize sz_subm = dist.globalTileElementDistance(idx_gl_begin, idx_gl_tile);
    const GlobalTileIndex idx_gl_comm(idx_gl_tile.col(), 0);

    // set buffer to zero
    ex::start_detached(comm_vec.readwrite(idx_gl_comm) | tile::set0(di::Policy<DefaultBackend_v<D>>()));

    // copy the first row of the matrix tile into the column tile of the buffer
    copy1DAsync<D>(k_row, k_col, sz_subm.rows(), sz_subm.cols(), Coord::Row, mat.read(idx_loc_tile),
                   Coord::Col, comm_vec.readwrite(idx_gl_comm));

    ex::start_detached(
        comm::scheduleAllReduceInPlace(col_task_chain(), MPI_SUM, comm_vec.readwrite(idx_gl_comm)));

    // copy the column tile of the buffer into the first column of the matrix tile
    copy1DAsync<D>(k_row, k_col, sz_subm.rows(), sz_subm.cols(), Coord::Col, comm_vec.read(idx_gl_comm),
                   Coord::Row, mat.readwrite(idx_loc_tile));
  }
}

template <class T, class KSender, class RhoSender>
void solveRank1ProblemDist(const SizeType i_begin, const SizeType i_end,
                           const LocalTileIndex idx_loc_begin, const LocalTileSize sz_loc_tiles,
                           KSender&& k, RhoSender&& rho, Matrix<const T, Device::CPU>& d,
                           Matrix<const T, Device::CPU>& z, Matrix<T, Device::CPU>& evals,
                           Matrix<T, Device::CPU>& delta, Matrix<SizeType, Device::CPU>& i2,
                           Matrix<T, Device::CPU>& evecs) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const matrix::Distribution& dist = evecs.distribution();
  auto rank1_fn = [i_begin, i_end, idx_loc_begin, sz_loc_tiles,
                   dist](const auto& k, const auto& rho, const auto& d_sfut_tile_arr,
                         const auto& z_sfut_tile_arr, const auto& eval_tiles, const auto& delta_tile_arr,
                         const auto& i2_tile_arr, const auto& evec_tile_arr) {
    common::internal::SingleThreadedBlasScope single;

    const SizeType n = problemSize(i_begin, i_end, dist);
    const T* d_ptr = d_sfut_tile_arr[0].get().ptr();
    const T* z_ptr = z_sfut_tile_arr[0].get().ptr();
    T* eval_ptr = eval_tiles[0].ptr();
    T* delta_ptr = delta_tile_arr[0].ptr();
    SizeType* i2_ptr = i2_tile_arr[0].ptr();

    // Iterate over the columns of the local submatrix tile grid
    for (SizeType j_loc_subm_tile = 0; j_loc_subm_tile < sz_loc_tiles.cols(); ++j_loc_subm_tile) {
      // The tile column in the local matrix tile grid
      const SizeType j_loc_tile = idx_loc_begin.col() + j_loc_subm_tile;
      // The tile column in the global matrix tile grid
      const SizeType j_gl_tile = dist.globalTileFromLocalTile<Coord::Col>(j_loc_tile);
      // The element distance between the current tile column and the initial tile column in the global
      // matrix tile grid
      const SizeType j_gl_subm_el = dist.globalTileElementDistance<Coord::Col>(i_begin, j_gl_tile);

      // Skip columns that are in the deflation zone
      if (j_gl_subm_el >= k)
        break;

      // Iterate over the elements of the column tile
      const SizeType ncols = std::min(dist.tileSize<Coord::Col>(j_gl_tile), k - j_gl_subm_el);
      for (SizeType j_tile_el = 0; j_tile_el < ncols; ++j_tile_el) {
        // The global submatrix column
        const SizeType j_gl_el = j_gl_subm_el + j_tile_el;

        // Solve the deflated rank-1 problem
        T& eigenval = eval_ptr[to_sizet(j_gl_el)];
        lapack::laed4(static_cast<int>(k), static_cast<int>(j_gl_el), d_ptr, z_ptr, delta_ptr, rho,
                      &eigenval);

        // Iterate over the rows of the local submatrix tile grid and copy the parts from delta stored on this rank.
        for (SizeType i_loc_subm_tile = 0; i_loc_subm_tile < sz_loc_tiles.rows(); ++i_loc_subm_tile) {
          const SizeType i_subm_evec_arr = i_loc_subm_tile + j_loc_subm_tile * sz_loc_tiles.rows();
          auto& evec_tile = evec_tile_arr[to_sizet(i_subm_evec_arr)];
          // The tile row in the local matrix tile grid
          const SizeType i_loc_tile = idx_loc_begin.row() + i_loc_subm_tile;
          // The tile row in the global matrix tile grid
          const SizeType i_gl_tile = dist.globalTileFromLocalTile<Coord::Row>(i_loc_tile);
          // The element distance between the current tile row and the initial tile row in the
          // global matrix tile grid
          const SizeType i_gl_subm_el = dist.globalTileElementDistance<Coord::Row>(i_begin, i_gl_tile);

          // The tile row in the global submatrix tile grid
          const SizeType i_subm_i2_arr = dist.globalTileFromLocalTile<Coord::Row>(i_loc_tile) - i_begin;
          auto& i2_tile = i2_tile_arr[to_sizet(i_subm_i2_arr)];

          const SizeType nrows = std::min(dist.tileSize<Coord::Row>(i_gl_tile), n - i_gl_subm_el);
          for (int i = 0; i < nrows; ++i) {
            const SizeType ii = i2_tile({i, 0});
            if (ii < k)
              evec_tile({i, j_tile_el}) = delta_ptr[ii];
          }
        }
      }
    }

    // Fill ones for deflated Eigenvectors.
    // Quick return if there is none.
    if (n == k)
      return;

    const GlobalElementIndex origin{i_begin * dist.blockSize().rows(),
                                    i_begin * dist.blockSize().cols()};
    for (SizeType i = 0; i < n; ++i) {
      const SizeType j = i2_ptr[i];
      if (j >= k) {
        const GlobalElementIndex i_g{i + origin.row(), j + origin.col()};
        const GlobalTileIndex i_tile = dist.globalTileIndex(i_g);
        if (dist.rankIndex() == dist.rankGlobalTile(i_tile)) {
          const LocalTileIndex i_tile_l = dist.localTileIndex(i_tile);
          const SizeType i_subm_evec_arr = i_tile_l.row() - idx_loc_begin.row() +
                                           (i_tile_l.col() - idx_loc_begin.col()) * sz_loc_tiles.rows();
          const TileElementIndex i = dist.tileElementIndex(i_g);
          evec_tile_arr[to_sizet(i_subm_evec_arr)](i) = T{1};
        }
      }
    }
  };

  TileCollector tc{i_begin, i_end};

  auto sender =
      ex::when_all(std::forward<KSender>(k), std::forward<RhoSender>(rho),
                   ex::when_all_vector(tc.read(d)), ex::when_all_vector(tc.read(z)),
                   ex::when_all_vector(tc.readwrite(evals)), ex::when_all_vector(tc.readwrite(delta)),
                   ex::when_all_vector(tc.readwrite(i2)), ex::when_all_vector(tc.readwrite(evecs)));

  ex::start_detached(di::transform(di::Policy<Backend::MC>(), std::move(rank1_fn), std::move(sender)));
}

// Assembles the local matrix of eigenvalues @p evals with size (n, 1) by communication tiles row-wise.
template <class T, Device D>
void assembleDistEvalsVec(common::Pipeline<comm::Communicator>& row_task_chain, const SizeType i_begin,
                          const SizeType i_end, const matrix::Distribution& dist_evecs,
                          Matrix<T, D>& evals) {
  namespace ex = pika::execution::experimental;

  comm::Index2D this_rank = dist_evecs.rankIndex();
  for (SizeType i = i_begin; i < i_end; ++i) {
    const GlobalTileIndex evals_idx(i, 0);

    const comm::IndexT_MPI evecs_tile_rank = dist_evecs.rankGlobalTile<Coord::Col>(i);
    if (evecs_tile_rank == this_rank.col()) {
      ex::start_detached(comm::scheduleSendBcast(row_task_chain(), evals.read(evals_idx)));
    }
    else {
      ex::start_detached(
          comm::scheduleRecvBcast(row_task_chain(), evecs_tile_rank, evals.readwrite(evals_idx)));
    }
  }
}

// Distributed version of the tridiagonal solver on CPUs
template <Backend B, class T, Device D, class RhoSender>
void mergeDistSubproblems(comm::CommunicatorGrid grid,
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

  // ---

  // Make sure Isend/Irecv messages don't match between calls by providing a unique `tag`
  //
  // Note: i_split is unique
  const comm::IndexT_MPI tag = to_int(i_split);
  applyGivensRotationsToMatrixColumns(grid.rowCommunicator(), tag, i_begin, i_end, std::move(rots),
                                      ws.e0);
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
  auto k = stablePartitionIndexForDeflation(i_begin, i_end, ws_h.c, ws_hm.i2, ws_h.i3) | ex::split();
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
  solveRank1ProblemDist(i_begin, i_end, idx_loc_begin, sz_loc_tiles, k, std::move(scaled_rho), ws_hm.d1,
                        ws_hm.z1, ws_h.d0, ws_hm.z0, ws_hm.i2, ws_hm.e2);

  assembleDistEvalsVec(row_task_chain, i_begin, i_end, dist_evecs, ws_h.d0);

  copy(idx_loc_begin, sz_loc_tiles, ws_hm.e2, ws.e2);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.z1, ws.z1);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.d1, ws.d1);
  copy(idx_begin_tiles_vec, sz_tiles_vec, ws_hm.i2, ws.i2);
  // ---

  auto n = ex::just(problemSize(i_begin, i_end, dist_evecs));
  // Eigenvector formation: `ws.e2` stores the eigenvectors, `ws.e0` is used as an additional workspace
  // Note ws.z0 is used as continuous buffer workspace to store permuted values of evals and ws.z1
  applyIndex(i_begin, i_end, ws.i2, ws.d1, ws.z0);
  initWeightVector(idx_gl_begin, idx_loc_begin, sz_loc_tiles, n, k, ws.d1, ws.z0, ws.e2, ws.e0);
  reduceMultiplyWeightVector(row_task_chain, i_begin, i_end, idx_loc_begin, sz_loc_tiles, n, k, ws.e0,
                             ws.z0);
  applyIndex(i_begin, i_end, ws.i2, ws.z1, ws.z0);
  formEvecsUsingWeightVec(idx_gl_begin, idx_loc_begin, sz_loc_tiles, n, k, ws.z0, ws.e0, ws.e2);
  sumsqEvecs(idx_gl_begin, idx_loc_begin, sz_loc_tiles, n, k, ws.e2, ws.e0);
  reduceSumScalingVector(col_task_chain, i_begin, i_end, idx_loc_begin, sz_loc_tiles, n, k, ws.e0,
                         ws.z0);
  normalizeEvecs(idx_gl_begin, idx_loc_begin, sz_loc_tiles, n, k, ws.e0, ws.e2);

  // Step #3: Eigenvectors of the tridiagonal system: Q * U
  //
  // The eigenvectors resulting from the multiplication are already in the order of the eigenvalues as
  // prepared for the deflated system.
  //

  dlaf::multiplication::generalSubMatrix<B, D, T>(grid, row_task_chain, col_task_chain, i_begin, i_end,
                                                  T(1), ws.e1, ws.e2, T(0), ws.e0);

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
