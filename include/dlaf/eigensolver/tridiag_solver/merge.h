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

#include <pika/datastructures/tuple.hpp>
#include <pika/future.hpp>
#include <pika/modules/iterator_support.hpp>
#include <pika/parallel/algorithms/merge.hpp>
#include <pika/parallel/algorithms/partition.hpp>
#include <pika/unwrap.hpp>
#include "dlaf/eigensolver/tridiag_solver/index.h"

#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

// The type of a column in the Q matrix
enum class ColType {
  UpperHalf,  // non-zeroes in the upper half only
  LowerHalf,  // non-zeroes in the lower half only
  Dense,      // full column vector
  Deflated    // deflated vectors
};

template <class T>
struct GivensRotation {
  SizeType i;  // the first column index
  SizeType j;  // the second column index
  T c;         // cosine
  T s;         // sine
};

struct ColTypeLens {
  SizeType num_uphalf;
  SizeType num_dense;
  SizeType num_lowhalf;
  SizeType num_deflated;
};

// Auxialiary matrix and vectors used for the D&C algorithm
template <class T>
struct WorkSpace {
  WorkSpace(WorkSpace&&) = default;
  WorkSpace& operator=(WorkSpace&&) = default;

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
  Matrix<T, Device::CPU> mat;

  // Holds the values of the deflated diagonal sorted in ascending order
  Matrix<T, Device::CPU> dtmp;
  // Holds the values of Cuppen's rank-1 vector
  Matrix<T, Device::CPU> z;
  // Holds the values of the rank-1 update vector sorted corresponding to `d_defl`
  Matrix<T, Device::CPU> ztmp;

  // Temporary index map storage
  Matrix<SizeType, Device::CPU> itmp;
  // An index map from next to prevous stage of the algorithm.
  //
  // The index is used to map:
  // 1. original  <--- presorted*
  // 2. presorted <--- deflated**
  // 3. deflated  <--- postsorted***
  //
  // * presorted: the indices of the diagonal `d` of the merged problem sorted in ascending order before
  //              the rank 1 problem is solved.
  // *** postsorted: the indices of the diagonal `d` of the merged problem sorted in ascending order after
  //                 the rank 1 problem is solved. (integrates back the deflated part)
  //
  Matrix<SizeType, Device::CPU> istage;
  // An index map: original <--- matmul
  Matrix<SizeType, Device::CPU> iorig;
  // An index map: <--- deflated
  Matrix<SizeType, Device::CPU> idefl;

  // Assigns a type to each column of Q which is used to calculate the permutation indices for Q and U
  // that bring them in matrix multiplication form.
  Matrix<ColType, Device::CPU> c;
  Matrix<ColType, Device::CPU> ctmp;
};

template <class T>
WorkSpace<T> initWorkSpace(const matrix::Distribution& ev_distr) {
  LocalElementSize vec_size(ev_distr.size().rows(), 1);
  TileElementSize vec_tile_size(ev_distr.blockSize().rows(), 1);
  return WorkSpace<T>{Matrix<T, Device::CPU>(ev_distr),
                      Matrix<T, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<T, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<T, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<ColType, Device::CPU>(vec_size, vec_tile_size),
                      Matrix<ColType, Device::CPU>(vec_size, vec_tile_size)};
}

// Calculates the problem size in the tile range [i_begin, i_end]
inline SizeType problemSize(SizeType i_begin, SizeType i_end, const matrix::Distribution& distr) {
  SizeType nb = distr.blockSize().rows();
  SizeType nbr = distr.tileSize(GlobalTileIndex(i_end, 0)).rows();
  return (i_end - i_begin) * nb + nbr;
}

// Copies and normalizes a row of the `tile` into the column vector tile `col`
//
template <class T>
void copyTileRowAndNormalize(SizeType row, T rho, const matrix::Tile<const T, Device::CPU>& tile,
                             const matrix::Tile<T, Device::CPU>& col) {
  // Negate Q1's last row if rho < 1
  //
  // lapack 3.10.0, dlaed2.f, line 280 and 281
  int sign = (rho < 0) ? -1 : 1;
  for (SizeType i = 0; i < tile.size().rows(); ++i) {
    col(TileElementIndex(i, 0)) = sign * tile(TileElementIndex(row, i)) / std::sqrt(2);
  }
}

DLAF_MAKE_CALLABLE_OBJECT(copyTileRowAndNormalize);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(copyTileRowAndNormalize, copyTileRowAndNormalize_o)

// The bottom row of Q1 and the top row of Q2. The bottom row of Q1 is negated if `rho < 0`.
//
// Note that the norm of `z` is sqrt(2) because it is a concatination of two normalized vectors. Hence to
// normalize `z` we have to divide by sqrt(2). That is handled in `copyTileRowAndNormalize()`
template <class T>
void assembleZVec(SizeType i_begin, SizeType i_middle, SizeType i_end, pika::shared_future<T> rho_fut,
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
    whenAllLift(tile_row, rho_fut, mat_ev.read_sender(mat_ev_idx), z.readwrite_sender(z_idx)) |
        copyTileRowAndNormalize(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

// Multiply by factor 2 to account for the normalization of `z` vector and make sure rho > 0 f
//
template <class T>
pika::future<T> scaleRho(pika::shared_future<T> rho_fut) {
  return pika::dataflow(pika::unwrapping([](T rho) { return 2 * std::abs(rho); }), std::move(rho_fut));
}

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

// Note the range is inclusive: [begin, end]
//
// The tiles are returned in column major order
template <class FutureTile, class T>
std::vector<FutureTile> collectTiles(GlobalTileIndex begin, GlobalTileIndex end,
                                     Matrix<T, Device::CPU>& mat) {
  std::size_t num_tiles = to_sizet(end.row() - begin.row() + 1) * to_sizet(end.col() - begin.col() + 1);
  std::vector<FutureTile> tiles;
  tiles.reserve(num_tiles);

  for (SizeType j = begin.col(); j <= end.col(); ++j) {
    for (SizeType i = begin.row(); i <= end.row(); ++i) {
      GlobalTileIndex idx(i, j);
      if constexpr (std::is_const<T>::value) {
        tiles.push_back(mat.read(idx));
      }
      else {
        tiles.push_back(mat(idx));
      }
    }
  }
  return tiles;
}

struct TileCollector {
  SizeType i_begin;
  SizeType i_end;

  template <class T>
  using ReadFutureTile = pika::shared_future<matrix::Tile<const T, Device::CPU>>;
  template <class T>
  using ReadWriteFutureTile = pika::future<matrix::Tile<T, Device::CPU>>;

  template <class T>
  std::vector<ReadFutureTile<T>> readVec(Matrix<const T, Device::CPU>& vec) {
    return collectTiles<ReadFutureTile<T>, const T>(GlobalTileIndex(i_begin, 0),
                                                    GlobalTileIndex(i_end, 0), vec);
  }

  template <class T>
  std::vector<ReadWriteFutureTile<T>> readwriteVec(Matrix<T, Device::CPU>& vec) {
    return collectTiles<ReadWriteFutureTile<T>, T>(GlobalTileIndex(i_begin, 0),
                                                   GlobalTileIndex(i_end, 0), vec);
  }

  template <class T>
  std::vector<ReadFutureTile<T>> readMat(Matrix<const T, Device::CPU>& mat) {
    return collectTiles<ReadFutureTile<T>, const T>(GlobalTileIndex(i_begin, i_begin),
                                                    GlobalTileIndex(i_end, i_end), mat);
  }

  template <class T>
  std::vector<ReadWriteFutureTile<T>> readwriteMat(Matrix<T, Device::CPU>& mat) {
    return collectTiles<ReadWriteFutureTile<T>, T>(GlobalTileIndex(i_begin, i_begin),
                                                   GlobalTileIndex(i_end, i_end), mat);
  }
};

template <class U>
std::vector<matrix::Tile<const U, Device::CPU>> unwrapConstTile(
    const std::vector<pika::shared_future<matrix::Tile<const U, Device::CPU>>>& tiles_fut) {
  std::vector<matrix::Tile<const U, Device::CPU>> tiles;
  tiles.reserve(tiles_fut.size());
  for (const auto& tile_fut : tiles_fut) {
    const auto& tile = tile_fut.get();
    SizeType len = tile.size().isEmpty() ? 0 : tile.ld() * (tile.size().cols() - 1) + tile.size().rows();
    tiles.emplace_back(tile.size(), MemoryView(tile.ptr(), len), tile.ld());
  }
  return tiles;
}

// Note: not using matrix::copy for Tile<> here because this has to work for U = SizeType too.
template <class U>
void copyVector(SizeType i_begin, SizeType i_end, Matrix<const U, Device::CPU>& in,
                Matrix<U, Device::CPU>& out) {
  auto copy_fn = [](const matrix::Tile<const U, Device::CPU>& in,
                    const matrix::Tile<U, Device::CPU>& out) {
    SizeType rows = in.size().rows();
    for (SizeType i = 0; i < rows; ++i) {
      TileElementIndex idx(i, 0);
      out(idx) = in(idx);
    }
  };
  for (SizeType i = i_begin; i <= i_end; ++i) {
    GlobalTileIndex idx(i, 0);
    pika::dataflow(pika::unwrapping(copy_fn), in.read(idx), out(idx));
  }
}

// Sorts an index `in_index_tiles` based on values in `vals_tiles` in ascending order into the index
// `out_index_tiles` where `vals_tiles` is composed of two pre-sorted ranges in ascending order that are
// merged, the first is [0, k) and the second is [k, n).
//
template <class T>
void sortIndex(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
               Matrix<const T, Device::CPU>& vec, Matrix<const SizeType, Device::CPU>& in_index,
               Matrix<SizeType, Device::CPU>& out_index) {
  SizeType n = problemSize(i_begin, i_end, vec.distribution());
  auto sort_fn = [n](auto k_fut, auto vec, auto in_index, auto out_index) {
    SizeType k = k_fut.get();
    DLAF_ASSERT(k <= n, k, n);

    TileElementIndex zero_idx(0, 0);
    const T* v_ptr = vec[0].get().ptr(zero_idx);
    const SizeType* in_index_ptr = in_index[0].get().ptr(zero_idx);
    // save in variable avoid releasing the tile too soon
    auto out_index_tile = out_index[0].get();
    SizeType* out_index_ptr = out_index_tile.ptr(zero_idx);

    auto begin_it = in_index_ptr;
    auto split_it = in_index_ptr + n;
    auto end_it = in_index_ptr + n;
    pika::merge(pika::execution::par, begin_it, split_it, split_it, end_it, out_index_ptr,
                [v_ptr](SizeType i1, SizeType i2) { return v_ptr[i1] < v_ptr[i2]; });
  };

  TileCollector tc{i_begin, i_end};
  pika::dataflow(std::move(sort_fn), std::move(k_fut), tc.readVec(vec), tc.readVec(in_index),
                 tc.readwriteVec(out_index));
}

// Applies `index` to `in` to get `out`
//
// Note: `pika::unwrapping()` can't be used on this function because std::vector<Tile<const >> is copied
// internally which requires that Tile<const > is copiable which it isn't. As a consequence the API can't
// be simplified unless const is dropped.
template <class T>
void applyIndex(SizeType i_begin, SizeType i_end, Matrix<const SizeType, Device::CPU>& index,
                Matrix<const T, Device::CPU>& in, Matrix<T, Device::CPU>& out) {
  SizeType n = problemSize(i_begin, i_end, index.distribution());
  auto applyIndex_fn = [n](auto index, auto in, auto out) {
    TileElementIndex zero_idx(0, 0);
    const SizeType* i_ptr = index[0].get().ptr(zero_idx);
    const T* in_ptr = in[0].get().ptr(zero_idx);
    // save in variable avoid releasing the tile too soon
    auto out_tile = out[0].get();
    T* out_ptr = out_tile.ptr(zero_idx);

    for (SizeType i = 0; i < n; ++i) {
      out_ptr[i] = in_ptr[i_ptr[i]];
    }
  };

  TileCollector tc{i_begin, i_end};

  pika::dataflow(std::move(applyIndex_fn), tc.readVec(index), tc.readVec(in), tc.readwriteVec(out));
}

inline void composeIndices(
    SizeType n, std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> outer,
    std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> inner,
    std::vector<pika::future<matrix::Tile<SizeType, Device::CPU>>> result) {
  TileElementIndex zero_idx(0, 0);
  const SizeType* inner_ptr = outer[0].get().ptr(zero_idx);
  const SizeType* outer_ptr = inner[0].get().ptr(zero_idx);
  // save in variable avoid releasing the tile too soon
  auto result_tile = result[0].get();
  SizeType* result_ptr = result_tile.ptr(zero_idx);

  for (SizeType i = 0; i < n; ++i) {
    result_ptr[i] = outer_ptr[inner_ptr[i]];
  }
}

inline SizeType stablePartitionIndexForDeflation(
    SizeType n, std::vector<pika::shared_future<matrix::Tile<const ColType, Device::CPU>>> ct_tiles,
    std::vector<pika::future<matrix::Tile<SizeType, Device::CPU>>> index_tiles) {
  TileElementIndex zero_idx(0, 0);
  const ColType* c_ptr = ct_tiles[0].get().ptr(zero_idx);
  // save in variable avoid releasing the tile too soon
  auto index_tile = index_tiles[0].get();
  SizeType* i_ptr = index_tile.ptr(zero_idx);

  // Get the number of non-deflated entries
  SizeType k = 0;
  for (SizeType i = 0; i < n; ++i) {
    if (c_ptr[i] != ColType::Deflated)
      ++k;
  }

  SizeType i1 = 0;  // index of non-deflated values in out
  SizeType i2 = k;  // index of deflated values
  for (SizeType i = 0; i < n; ++i) {
    SizeType& io = (c_ptr[i] == ColType::Deflated) ? i1 : i2;
    i_ptr[io] = i;
    ++io;
  }
  return k;
}

inline void setColTypeTile(const matrix::Tile<ColType, Device::CPU>& tile, ColType val) {
  SizeType len = tile.size().rows();
  for (SizeType i = 0; i < len; ++i) {
    tile(TileElementIndex(i, 0)) = val;
  }
}

DLAF_MAKE_CALLABLE_OBJECT(setColTypeTile);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(setColTypeTile, setColTypeTile_o)

inline void initColTypes(SizeType i_begin, SizeType i_split, SizeType i_end,
                         Matrix<ColType, Device::CPU>& coltypes) {
  using pika::threads::thread_priority;
  using pika::execution::experimental::start_detached;
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;

  for (SizeType i = i_begin; i <= i_end; ++i) {
    ColType val = (i <= i_split) ? ColType::UpperHalf : ColType::LowerHalf;
    whenAllLift(coltypes.readwrite_sender(LocalTileIndex(i, 0)), val) |
        setColTypeTile(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

// Returns true if `d1` is close to `d2`.
//
// Given's deflation condition is the same as the one used in LAPACK's stedc implementation [1].
//
// [1] LAPACK 3.10.0, file dlaed2.f, line 393
template <class T>
bool diagonalValuesNearlyEqual(T tol, T d1, T d2, T z1, T z2) {
  // Note that this is similar to calling `rotg()` but we want to make sure that the condition is
  // satisfied before modifying z1, z2 that is why the function is not used here.
  return std::abs(z1 * z2 * (d1 - d2) / (z1 * z1 + z2 * z2)) < tol;
}

// Assumption 1: The algorithm assumes that the arrays `d_ptr`, `z_ptr` and `c_ptr` are of equal length
// `len` and are sorted in ascending order.
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
    T& d1 = d_ptr[i1];
    T& d2 = d_ptr[i2];
    T& z1 = z_ptr[i1];
    T& z2 = z_ptr[i2];
    ColType& c1 = c_ptr[i1];
    ColType& c2 = c_ptr[i2];

    // if z1 nearly zero deflate the element and move i1 forward to i2
    if (std::abs(rho * z1) < tol) {
      c1 = ColType::Deflated;
      i1 = i2;
      continue;
    }

    // Deflate the second element if z2 nearly zero
    if (std::abs(rho * z2) < tol) {
      c2 = ColType::Deflated;
      continue;
    }

    // If d1 is not nearly equal to d2, move i1 forward to i2
    if (!diagonalValuesNearlyEqual(tol, d1, d2, z1, z2)) {
      i1 = i2;
      continue;
    }

    // When d1 is nearly equal to d2 apply Givens rotation
    T c, s;
    blas::rotg(&z1, &z2, &c, &s);
    d1 = d1 * c * c + d2 * s * s;
    d2 = d1 * s * s + d2 * c * c;

    rots.push_back(GivensRotation<T>{i_ptr[i1], i_ptr[i2], c, s});
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
std::vector<GivensRotation<T>> applyDeflation(
    SizeType n, pika::shared_future<T> rho_fut, pika::shared_future<T> tol_fut,
    std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> i_tiles,
    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> d_tiles,
    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> z_tiles,
    std::vector<pika::future<matrix::Tile<ColType, Device::CPU>>> c_tiles) {
  TileElementIndex zero_idx(0, 0);
  const SizeType* i_ptr = i_tiles[0].get().ptr(zero_idx);
  // save in variable avoid releasing the tile too soon
  auto d_tile = d_tiles[0].get();
  auto z_tile = z_tiles[0].get();
  auto c_tile = c_tiles[0].get();
  T* d_ptr = d_tile.ptr(zero_idx);
  T* z_ptr = z_tile.ptr(zero_idx);
  ColType* c_ptr = c_tile.ptr(zero_idx);

  return applyDeflationToArrays(rho_fut.get(), tol_fut.get(), n, i_ptr, d_ptr, z_ptr, c_ptr);
}

// Partitions `p_ptr` based on a `ctype` in `c_ptr` array.
//
// Returns the number of elements in the second partition.
inline SizeType partitionColType(SizeType len, ColType ctype, const ColType* c_ptr, SizeType* i_ptr) {
  auto it = pika::partition(pika::execution::par, i_ptr, i_ptr + len,
                            [ctype, c_ptr](SizeType i) { return ctype != c_ptr[i]; });
  return len - std::distance(i_ptr, it);
}

// Partition `coltypes` to get the indices of the matrix-multiplication form
inline ColTypeLens partitionIndexForMatrixMultiplication(
    SizeType n, std::vector<pika::shared_future<matrix::Tile<const ColType, Device::CPU>>> c_tiles,
    std::vector<pika::future<matrix::Tile<SizeType, Device::CPU>>> perm_q_tiles) {
  TileElementIndex zero_idx(0, 0);
  const ColType* c_ptr = c_tiles[0].get().ptr(zero_idx);
  // save in variable avoid releasing the tile too soon
  auto perm_q_tile = perm_q_tiles[0].get();
  SizeType* i_ptr = perm_q_tile.ptr(zero_idx);
  ColTypeLens ql;
  ql.num_deflated = partitionColType(n, ColType::Deflated, c_ptr, i_ptr);
  ql.num_lowhalf = partitionColType(n - ql.num_deflated, ColType::LowerHalf, c_ptr, i_ptr);
  ql.num_dense = partitionColType(n - ql.num_deflated - ql.num_lowhalf, ColType::Dense, c_ptr, i_ptr);
  ql.num_uphalf = n - ql.num_deflated - ql.num_lowhalf - ql.num_dense;
  return ql;
};

// Assumption: the memory layout of the matrix from which the tiles are coming is column major.
//
// `tiles`: The tiles of the matrix between tile indices `(i_begin, i_begin)` and `(i_end, i_end)` that
// are potentially affected by the Givens rotations. `n` : column size
//
// Note: a column index may be paired to more than one other index, this may lead to a race condition if
//       parallelized trivially. Current implementation is serial.
//
template <class T>
void applyGivensRotationsToMatrixColumns(SizeType n, SizeType nb, std::vector<GivensRotation<T>> rots,
                                         std::vector<matrix::Tile<T, Device::CPU>> tiles) {
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

    DLAF_ASSERT(i_el != j_el, i_el, j_el);

    // Apply Givens rotations
    blas::rot(n, x, 1, y, 1, rot.c, rot.s);
  }
}

// Copy column `in_col` from tile `in` to column `out_col` in tile `out`.
template <class T>
void copyTileCol(SizeType in_col, matrix::Tile<const T, Device::CPU>& in, SizeType out_col,
                 matrix::Tile<T, Device::CPU>& out) {
  DLAF_ASSERT(in.size().rows() <= out.size().rows(), in.size(), out.size());
  DLAF_ASSERT(in_col <= in.size().cols(), in_col);
  DLAF_ASSERT(out_col <= out.size().cols(), in_col);

  for (SizeType i = 0; i < in.size().rows(); ++i) {
    out(TileElementIndex(i, out_col)) = in(TileElementIndex(i, in_col));
  }
}

// Applies the permutation index `perm_tiles` to tiles from `Q` and saves the result in a workspace
// matrix `mat_q`.
//
// `in_tiles` and `out_tiles` are in column-major order
//
// Note: this is currently not parallelized
// template <class T>
// void applyPermutationIndexToMatrixQ(
//    SizeType i_begin, SizeType i_end,
//    std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> perm_tiles,
//    const matrix::Distribution& distr,
//    std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> in_tiles,
//    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> out_tiles) {
//  SizeType ncol_tiles = i_end - i_begin + 1;
//
//  // Iterate over columns of `in_tiles` and use the permutation index `perm_tiles` to copy columns from
//  for (SizeType j_out_tile = i_begin; j_out_tile <= i_end; ++j_out_tile) {
//    // Get the indices of permutations correspo
//    const auto& perm_tile = perm_tiles[to_sizet(j_out_tile)].get();
//    for (SizeType i_out_tile = i_begin; i_out_tile <= i_end; ++i_out_tile) {
//      auto out_tile = out_tiles[to_sizet(i_out_tile + j_out_tile * ncol_tiles)].get();
//      TileElementSize sz_out_tile = out_tile.size();
//
//      // Iterate over columns of `out_tile`
//      for (SizeType j_out_el = 0; j_out_el < sz_out_tile.cols(); ++j_out_el) {
//        // Get the index of the global column
//        SizeType j_in_gl_el = perm_tile(TileElementIndex(j_out_el, 0));
//        // Get the index of the tile that has column `gl_col`
//        SizeType j_in_tile = distr.globalTileFromGlobalElement<Coord::Col>(j_in_gl_el);
//        const auto& in_tile = in_tiles[to_sizet(i_out_tile + j_in_tile * ncol_tiles)].get();
//        // Get the index of the `gl_col` column within the tile
//        SizeType j_in_el = distr.tileElementFromGlobalElement<Coord::Col>(j_in_gl_el);
//        // Copy a column from `in_tile` into `out_tile`
//        for (SizeType i_out_el = 0; i_out_el < sz_out_tile.rows(); ++i_out_el) {
//          out_tile(TileElementIndex(i_out_el, j_out_el)) = in_tile(TileElementIndex(i_out_el, j_in_el));
//        }
//      }
//    }
//  }
//}

// Inverts `perm_q`
// Initializes `perm_u`
inline void setPermutationsU(
    SizeType n, std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> perm_d,
    std::vector<pika::future<matrix::Tile<SizeType, Device::CPU>>> perm_q,
    std::vector<pika::future<matrix::Tile<SizeType, Device::CPU>>> perm_u) {
  TileElementIndex zero(0, 0);
  const SizeType* d_ptr = perm_d[0].get().ptr(zero);
  SizeType* q_ptr = perm_q[0].get().ptr(zero);
  SizeType* u_ptr = perm_u[0].get().ptr(zero);

  // Invert `perm_q` into `perm_u` temporarily
  for (SizeType i = 0; i < n; ++i) {
    u_ptr[q_ptr[i]] = i;
  }

  // Copy `perm_u` to `perm_q` to save the inverted index back into `perm_q`
  for (SizeType i = 0; i < n; ++i) {
    q_ptr[i] = u_ptr[i];
  }

  // Index map from sorted incices to matrix-multiplication index
  for (SizeType i = 0; i < n; ++i) {
    u_ptr[i] = q_ptr[d_ptr[i]];
  }
}

template <class T>
void solveRank1Problem(SizeType n, SizeType nb, pika::shared_future<SizeType> k_fut,
                       pika::shared_future<T> rho_fut,
                       std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> d_defl,
                       std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> z_defl,
                       std::vector<pika::future<matrix::Tile<T, Device::CPU>>> d,
                       std::vector<pika::future<matrix::Tile<T, Device::CPU>>> mat) {
  SizeType k = k_fut.get();
  T rho = rho_fut.get();

  TileElementIndex zero(0, 0);
  const T* d_defl_ptr = d_defl[0].get().ptr(zero);
  const T* z_ptr = z_defl[0].get().ptr(zero);

  auto d_tile = d[0].get();
  auto mat_tiles = pika::unwrap(std::move(mat));
  T* d_ptr = d_tile.ptr(zero);

  matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

  for (SizeType i = 0; i < k; ++i) {
    SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, i));
    SizeType i_col = distr.tileElementFromGlobalElement<Coord::Col>(i);
    T* delta = mat_tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_col));
    T& eigenval = d_ptr[to_sizet(i)];
    dlaf::internal::laed4_wrapper(static_cast<int>(k), static_cast<int>(i), d_defl_ptr, z_ptr, delta,
                                  rho, &eigenval);

    // TODO: check the eigenvectors formula for `delta`
  }
}

// `mat_[q,u,ev]_tiles` are in column-major order with leading dimension `ncol_tiles`.
//
// template <class T>
// void gemmQU(SizeType i_begin, SizeType i_end, SizeType ncol_tiles, SizeType n1, SizeType n2,
//            pika::shared_future<QLens> qlens_fut,
//            std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> mat_q_tiles,
//            std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> mat_u_tiles,
//            std::vector<pika::future<matrix::Tile<T, Device::CPU>>> mat_ev_tiles) {
//  const QLens& qlens = qlens_fut.get();
//  SizeType l1 = qlens.num_uphalf + qlens.num_dense;
//  SizeType l2 = qlens.num_lowhalf + qlens.num_dense;
//
//  // Iterate over `mat_ev` tiles in column-major order
//  for (SizeType j = i_begin; j <= i_end; ++j) {
//    for (SizeType i = i_begin; i <= i_end; ++i) {
//      auto mat_ev = mat_ev_tiles[i + j * ncol_tiles].get();
//      // Iterate over rows of `mat_q` and columns of `mat_u`
//      for (SizeType k = i_begin; k <= i_end; ++k) {
//        auto const& q_tile = mat_q_tiles[i + k * ncol_tiles].get();
//        auto const& u_tile = mat_u_tiles[j + k * ncol_tiles].get();
//        gemm(blas::Op::NoTrans, blas::Op::Trans, T(1), q_tile, u_tile, T(0), mat_ev);
//      }
//    }
//  }
//}

template <class T>
void mergeSubproblems(SizeType i_begin, SizeType i_split, SizeType i_end, WorkSpace<T>& ws,
                      pika::shared_future<T> rho_fut, Matrix<T, Device::CPU>& d,
                      Matrix<T, Device::CPU>& mat_ev) {
  TileCollector tc{i_begin, i_end};

  // Calculate the merged size of the subproblem
  SizeType n1 = problemSize(i_begin, i_split, mat_ev.distribution());
  SizeType n2 = problemSize(i_split + 1, i_end, mat_ev.distribution());
  SizeType n = n1 + n2;
  SizeType nb = mat_ev.distribution().blockSize().rows();

  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleZVec(i_begin, i_split, i_end, rho_fut, mat_ev, ws.z);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  rho_fut = scaleRho(rho_fut);

  // Calculate the tolerance used for deflation
  pika::future<T> dmax_fut = maxVectorElement(i_begin, i_end, d);
  pika::future<T> zmax_fut = maxVectorElement(i_begin, i_end, ws.z);
  pika::shared_future<T> tol_fut =
      pika::dataflow(pika::unwrapping(calcTolerance<T>), std::move(dmax_fut), std::move(zmax_fut));

  // Initialize the column types vector `c`
  initColTypes(i_begin, i_split, i_end, ws.c);

  // 1. set index `istage` to `original <--- presorted`
  // 2. update index `iorig` to `original <--- presorted`
  // 3. reorder `d -> dtmp`, `z -> ztmp` and `c -> ctmp` using the `istage` such that the diagonal is in
  //    ascending order
  // 4. Deflate the sorted `dtmp`, `ztmp` and `ctmp`
  //
  initIndex(i_begin, i_end, ws.itmp);
  sortIndex(i_begin, i_end, pika::make_ready_future(n1), d, ws.itmp, ws.istage);
  copyVector(i_begin, i_end, ws.iorig, ws.istage);
  applyIndex(i_begin, i_end, ws.istage, d, ws.dtmp);
  applyIndex(i_begin, i_end, ws.istage, ws.z, ws.ztmp);
  applyIndex(i_begin, i_end, ws.istage, ws.c, ws.ctmp);
  pika::future<std::vector<GivensRotation<T>>> rots_fut =
      pika::dataflow(applyDeflation<T>, n, rho_fut, tol_fut, tc.readVec(ws.istage),
                     tc.readwriteVec(ws.dtmp), tc.readwriteVec(ws.ztmp), tc.readwriteVec(ws.ctmp));

  // 1. set index `istage` to `presorted <--- deflated`
  // 2. update index `iorig` to `original <--- deflated`
  // 3. reorder `dtmp -> d`, `ztmp -> z` and `ctmp -> c` using the `istage` such that deflated entries
  //    are at the bottom.
  // 4. Solve the rank-1 problem `d + rho * z * z^T`, save eigenvalues in `dtmp` and eigenvectors in
  // `ws.mat`.
  //
  pika::shared_future<SizeType> k_fut = pika::dataflow(stablePartitionIndexForDeflation, n,
                                                       tc.readVec(ws.ctmp), tc.readwriteVec(ws.istage));
  composeIndices(n, tc.readVec(ws.iorig), tc.readVec(ws.istage), tc.readwriteVec(ws.itmp));
  copyVector(i_begin, i_end, ws.itmp, ws.iorig);
  applyIndex(i_begin, i_end, ws.istage, ws.dtmp, d);
  applyIndex(i_begin, i_end, ws.istage, ws.ztmp, ws.z);
  applyIndex(i_begin, i_end, ws.istage, ws.ctmp, ws.c);
  copyVector(i_begin, i_end, d, ws.dtmp);
  pika::dataflow(solveRank1Problem<T>, n, nb, k_fut, rho_fut, tc.readVec(ws.dtmp), tc.readVec(ws.z),
                 tc.readwriteVec(d), tc.readwriteVec(ws.mat));

  // 1. set index `istage` to `deflated <--- postsorted`
  // 2. set the index `idefl` to `deflated <--- postsorted`
  // 3. update the index `iorig` to `original <--- postsorted`
  // 4. reorder `d -> dtmp` and `c -> ctmp` using the `istage` such that `d` values (eigenvalues and
  //    deflated values) are in ascending order
  // 5. copy `dtmp -> d`
  //
  initIndex(i_begin, i_end, ws.itmp);
  sortIndex(i_begin, i_end, k_fut, ws.dtmp, ws.itmp, ws.istage);
  copyVector(i_begin, i_end, ws.iorig, ws.istage);
  copyVector(i_begin, i_end, ws.istage, ws.idefl);
  composeIndices(n, tc.readVec(ws.iorig), tc.readVec(ws.istage), tc.readwriteVec(ws.itmp));
  copyVector(i_begin, i_end, ws.itmp, ws.iorig);
  applyIndex(i_begin, i_end, ws.istage, d, ws.dtmp);
  applyIndex(i_begin, i_end, ws.istage, ws.c, ws.ctmp);
  copyVector(i_begin, i_end, ws.dtmp, d);

  // 1. set index `istage` to `postsorted <--- matmul`
  // 2. set the index `idefl` to `deflated <--- matmul`
  // 3. update the index `iorig` to `original <--- postsorted`
  pika::shared_future<ColTypeLens> qlens_fut =
      pika::dataflow(partitionIndexForMatrixMultiplication, n, tc.readVec(ws.ctmp),
                     tc.readwriteVec(ws.istage));
  composeIndices(n, tc.readVec(ws.idefl), tc.readVec(ws.istage), tc.readwriteVec(ws.itmp));
  copyVector(i_begin, i_end, ws.itmp, ws.idefl);
  composeIndices(n, tc.readVec(ws.iorig), tc.readVec(ws.istage), tc.readwriteVec(ws.itmp));
  copyVector(i_begin, i_end, ws.itmp, ws.iorig);

  // Apply Givens rotations to `Q` - `mat_ev`
  pika::dataflow(pika::unwrapping(applyGivensRotationsToMatrixColumns<T>), n, nb, std::move(rots_fut),
                 tc.readwriteMat(mat_ev));

  // Use the permutation index `perm_q` on the columns of `mat_ev` and save the result to `mat_q`
  // pika::dataflow(applyPermutationIndexToMatrixQ<T>, i_begin, i_end,
  //               collectReadTiles(col_begin, col_end, ws.imatmul), mat_ev.distribution(),
  //               collectReadTiles(ev_begin, ev_end, mat_ev),
  //               collectReadWriteTiles(ev_begin, ev_end, ws.mat));

  // Invert the `perm_q` index and map sorted indices to matrix-multiplication indices in `perm_u`
  // pika::dataflow(setPermutationsU, n, collectReadTiles(col_begin, col_end, ws.isorted),
  //               collectReadWriteTiles(col_begin, col_end, ws.imatmul),
  //               collectReadWriteTiles(col_begin, col_end, perm_u));

  // GEMM `mat_q` holding Q and `mat_u` holding U^T into `mat_ev`
  //
  // Note: the transpose of `mat_u` is used here to recover U
  // pika::dataflow(gemmQU<T>, qlens_fut, collectReadTiles(ev_begin, ev_end, mat_qws),
  //               collectReadTiles(ev_begin, ev_end, mat_uws),
  //               collectReadWriteTiles(ev_begin, ev_end, mat_ev));
}

}
