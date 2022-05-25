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

#include "dlaf/eigensolver/tridiag_solver/gemm.h"
#include "dlaf/eigensolver/tridiag_solver/index.h"
#include "dlaf/eigensolver/tridiag_solver/permutations.h"
#include "dlaf/lapack/laed4.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include "dlaf/matrix/print_csv.h"

namespace dlaf::eigensolver::internal {

// The type of a column in the Q matrix
enum class ColType {
  UpperHalf,  // non-zeroes in the upper half only
  LowerHalf,  // non-zeroes in the lower half only
  Dense,      // full column vector
  Deflated    // deflated vectors
};

inline std::ostream& operator<<(std::ostream& str, const ColType& ct) {
  if (ct == ColType::Deflated) {
    str << "Deflated";
  }
  else if (ct == ColType::Dense) {
    str << "Dense";
  }
  else if (ct == ColType::UpperHalf) {
    str << "UpperHalf";
  }
  else {
    str << "LowerHalf";
  }
  return str;
}

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

// Auxiliary matrix and vectors used for the D&C algorithm
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
  Matrix<T, Device::CPU> mat1;
  Matrix<T, Device::CPU> mat2;

  // Holds the values of the deflated diagonal sorted in ascending order
  Matrix<T, Device::CPU> dtmp;
  // Holds the values of Cuppen's rank-1 vector
  Matrix<T, Device::CPU> z;
  // Holds the values of the rank-1 update vector sorted corresponding to `d_defl`
  Matrix<T, Device::CPU> ztmp;

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
  Matrix<SizeType, Device::CPU> i1;
  Matrix<SizeType, Device::CPU> i2;
  Matrix<SizeType, Device::CPU> i3;

  // Assigns a type to each column of Q which is used to calculate the permutation indices for Q and U
  // that bring them in matrix multiplication form.
  Matrix<ColType, Device::CPU> c;
};

template <class T>
WorkSpace<T> initWorkSpace(const matrix::Distribution& ev_distr) {
  LocalElementSize vec_size(ev_distr.size().rows(), 1);
  TileElementSize vec_tile_size(ev_distr.blockSize().rows(), 1);
  WorkSpace<T> ws{Matrix<T, Device::CPU>(ev_distr),
                  Matrix<T, Device::CPU>(ev_distr),
                  Matrix<T, Device::CPU>(vec_size, vec_tile_size),
                  Matrix<T, Device::CPU>(vec_size, vec_tile_size),
                  Matrix<T, Device::CPU>(vec_size, vec_tile_size),
                  Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),
                  Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),
                  Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),
                  Matrix<ColType, Device::CPU>(vec_size, vec_tile_size)};
  return ws;
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
void copyTileRowAndNormalize(bool top_tile, T rho, const matrix::Tile<const T, Device::CPU>& tile,
                             const matrix::Tile<T, Device::CPU>& col) {
  // Copy the bottom row of the top tile or the top row of the bottom tile
  SizeType row = (top_tile) ? col.size().rows() - 1 : 0;

  // Negate Q1's last row if rho < 0
  //
  // lapack 3.10.0, dlaed2.f, line 280 and 281
  int sign = (top_tile && rho < 0) ? -1 : 1;

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
void assembleZVec(SizeType i_begin, SizeType i_split, SizeType i_end, pika::shared_future<T> rho_fut,
                  Matrix<const T, Device::CPU>& mat_ev, Matrix<T, Device::CPU>& z) {
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;
  using pika::threads::thread_priority;

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
    whenAllLift(top_tile, rho_fut, mat_ev.read_sender(mat_ev_idx), z.readwrite_sender(z_idx)) |
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
pika::future<T> calcTolerance(SizeType i_begin, SizeType i_end, Matrix<const T, Device::CPU>& d,
                              Matrix<const T, Device::CPU>& z) {
  pika::future<T> dmax_fut = maxVectorElement(i_begin, i_end, d);
  pika::future<T> zmax_fut = maxVectorElement(i_begin, i_end, z);

  auto tol_fn = [](T dmax, T zmax) {
    return 8 * std::numeric_limits<T>::epsilon() * std::max(dmax, zmax);
  };
  return pika::dataflow(pika::unwrapping(std::move(tol_fn)), std::move(dmax_fut), std::move(zmax_fut));
}

// Note the range is inclusive: [begin, end]
//
// The tiles are returned in column major order
template <class FutureTile, class T>
std::vector<FutureTile> collectTiles(SizeType i_begin, SizeType i_end, Matrix<T, Device::CPU>& mat) {
  bool is_col_matrix = mat.distribution().size().cols() == 1;
  SizeType col_begin = (is_col_matrix) ? 0 : i_begin;
  SizeType col_end = (is_col_matrix) ? 0 : i_end;

  std::vector<FutureTile> tiles;
  tiles.reserve(to_sizet(i_end - i_begin + 1) * to_sizet(col_end - col_begin + 1));

  for (SizeType j = col_begin; j <= col_end; ++j) {
    for (SizeType i = i_begin; i <= i_end; ++i) {
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
  std::vector<ReadFutureTile<T>> read(Matrix<const T, Device::CPU>& mat) {
    return collectTiles<ReadFutureTile<T>, const T>(i_begin, i_end, mat);
  }

  template <class T>
  std::vector<ReadWriteFutureTile<T>> readwrite(Matrix<T, Device::CPU>& mat) {
    return collectTiles<ReadWriteFutureTile<T>, T>(i_begin, i_end, mat);
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
    auto split_it = in_index_ptr + k;
    auto end_it = in_index_ptr + n;
    pika::merge(pika::execution::par, begin_it, split_it, split_it, end_it, out_index_ptr,
                [v_ptr](SizeType i1, SizeType i2) { return v_ptr[i1] < v_ptr[i2]; });
  };

  TileCollector tc{i_begin, i_end};
  pika::dataflow(std::move(sort_fn), std::move(k_fut), tc.read(vec), tc.read(in_index),
                 tc.readwrite(out_index));
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
  pika::dataflow(std::move(applyIndex_fn), tc.read(index), tc.read(in), tc.readwrite(out));
}

inline void composeIndices(SizeType i_begin, SizeType i_end, Matrix<const SizeType, Device::CPU>& outer,
                           Matrix<const SizeType, Device::CPU>& inner,
                           Matrix<SizeType, Device::CPU>& result) {
  SizeType n = problemSize(i_begin, i_end, outer.distribution());
  auto compose_fn = [n](auto outer_tiles, auto inner_tiles, auto result_tiles) {
    TileElementIndex zero_idx(0, 0);
    const SizeType* inner_ptr = outer_tiles[0].get().ptr(zero_idx);
    const SizeType* outer_ptr = inner_tiles[0].get().ptr(zero_idx);
    // save in variable avoid releasing the tile too soon
    auto result_tile = result_tiles[0].get();
    SizeType* result_ptr = result_tile.ptr(zero_idx);

    for (SizeType i = 0; i < n; ++i) {
      result_ptr[i] = outer_ptr[inner_ptr[i]];
    }
  };

  TileCollector tc{i_begin, i_end};
  pika::dataflow(compose_fn, tc.read(outer), tc.read(inner), tc.readwrite(result));
}

inline void invertIndex(SizeType i_begin, SizeType i_end, Matrix<const SizeType, Device::CPU>& in,
                        Matrix<SizeType, Device::CPU>& out) {
  SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto inv_fn = [n](auto in_tiles, auto out_tiles) {
    TileElementIndex zero(0, 0);
    const SizeType* in_ptr = in_tiles[0].get().ptr(zero);
    // save in variable avoid releasing the tile too soon
    auto out_tile = out_tiles[0].get();
    SizeType* out_ptr = out_tile.ptr(zero);

    for (SizeType i = 0; i < n; ++i) {
      out_ptr[in_ptr[i]] = i;
    }
  };

  TileCollector tc{i_begin, i_end};
  pika::dataflow(std::move(inv_fn), tc.read(in), tc.readwrite(out));
}

// The index array `out_ptr` holds the indices of elements of `c_ptr` that order it such that ColType::Deflated
// entries are moved to the end. The `c_ptr` array is implicitly ordered according to `in_ptr` on entry.
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

inline pika::future<SizeType> stablePartitionIndexForDeflation(SizeType i_begin, SizeType i_end,
                                                               Matrix<const ColType, Device::CPU>& c,
                                                               Matrix<const SizeType, Device::CPU>& in,
                                                               Matrix<SizeType, Device::CPU>& out) {
  SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto part_fn = [n](auto c_tiles, auto in_tiles, auto out_tiles) {
    TileElementIndex zero_idx(0, 0);
    const ColType* c_ptr = c_tiles[0].get().ptr(zero_idx);
    const SizeType* in_ptr = in_tiles[0].get().ptr(zero_idx);
    // save in variable avoid releasing the tile too soon
    auto out_tile = out_tiles[0].get();
    SizeType* out_ptr = out_tile.ptr(zero_idx);
    return stablePartitionIndexForDeflationArrays(n, c_ptr, in_ptr, out_ptr);
  };

  TileCollector tc{i_begin, i_end};
  return pika::dataflow(part_fn, tc.read(c), tc.read(in), tc.readwrite(out));
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
inline pika::future<ColTypeLens> partitionIndexForMatrixMultiplication(
    SizeType i_begin, SizeType i_end, Matrix<const ColType, Device::CPU>& c,
    Matrix<SizeType, Device::CPU>& index) {
  SizeType n = problemSize(i_begin, i_end, c.distribution());
  auto part_fn = [n](auto c_tiles, auto index_tiles) {
    TileElementIndex zero_idx(0, 0);
    const ColType* c_ptr = c_tiles[0].get().ptr(zero_idx);
    // save in variable avoid releasing the tile too soon
    auto index_tile = index_tiles[0].get();
    SizeType* i_ptr = index_tile.ptr(zero_idx);

    ColTypeLens ql;
    ql.num_deflated = partitionColType(n, ColType::Deflated, c_ptr, i_ptr);
    ql.num_lowhalf = partitionColType(n - ql.num_deflated, ColType::LowerHalf, c_ptr, i_ptr);
    ql.num_dense = partitionColType(n - ql.num_deflated - ql.num_lowhalf, ColType::Dense, c_ptr, i_ptr);
    ql.num_uphalf = n - ql.num_deflated - ql.num_lowhalf - ql.num_dense;
    return ql;
  };

  TileCollector tc{i_begin, i_end};
  return pika::dataflow(std::move(part_fn), tc.read(c), tc.readwrite(index));
};

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
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;
  using pika::threads::thread_priority;

  for (SizeType i = i_begin; i <= i_end; ++i) {
    ColType val = (i <= i_split) ? ColType::UpperHalf : ColType::LowerHalf;
    whenAllLift(coltypes.readwrite_sender(LocalTileIndex(i, 0)), val) |
        setColTypeTile(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
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
    // However, here the second entry is deflated instead of the first (z2/d2 instead of z1/d1), thus `s`
    // is not negated.
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
  SizeType n = problemSize(i_begin, i_end, index.distribution());

  auto deflate_fn = [n](auto rho_fut, auto tol_fut, auto index_tiles, auto d_tiles, auto z_tiles,
                        auto c_tiles) {
    TileElementIndex zero_idx(0, 0);
    const SizeType* i_ptr = index_tiles[0].get().ptr(zero_idx);
    // save in variable avoid releasing the tile too soon
    auto d_tile = d_tiles[0].get();
    auto z_tile = z_tiles[0].get();
    auto c_tile = c_tiles[0].get();
    T* d_ptr = d_tile.ptr(zero_idx);
    T* z_ptr = z_tile.ptr(zero_idx);
    ColType* c_ptr = c_tile.ptr(zero_idx);

    return applyDeflationToArrays(rho_fut.get(), tol_fut.get(), n, i_ptr, d_ptr, z_ptr, c_ptr);
  };

  TileCollector tc{i_begin, i_end};
  return pika::dataflow(std::move(deflate_fn), std::move(rho_fut), std::move(tol_fut), tc.read(index),
                        tc.readwrite(d), tc.readwrite(z), tc.readwrite(c));
}

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

    // Apply Givens rotations
    blas::rot(n, x, 1, y, 1, rot.c, rot.s);
  }
}

template <class T>
void solveRank1Problem(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
                       pika::shared_future<T> rho_fut, Matrix<const T, Device::CPU>& d,
                       Matrix<const T, Device::CPU>& z, Matrix<T, Device::CPU>& evals,
                       Matrix<T, Device::CPU>& evecs) {
  SizeType n = problemSize(i_begin, i_end, evals.distribution());
  SizeType nb = evals.distribution().blockSize().rows();

  auto rank1_fn = [n, nb](auto k_fut, auto rho_fut, auto d_tiles, auto z_tiles, auto eval_tiles,
                          auto evec_tiles_fut) {
    SizeType k = k_fut.get();
    T rho = rho_fut.get();

    TileElementIndex zero(0, 0);
    const T* d_ptr = d_tiles[0].get().ptr(zero);
    const T* z_ptr = z_tiles[0].get().ptr(zero);

    // save in variable avoid releasing the tile too soon
    auto eval_tile = eval_tiles[0].get();
    T* eval_ptr = eval_tile.ptr(zero);

    auto evec_tiles = pika::unwrap(std::move(evec_tiles_fut));
    matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

    std::vector<SizeType> loop_arr(to_sizet(k));
    std::iota(std::begin(loop_arr), std::end(loop_arr), 0);
    pika::for_each(pika::execution::par, std::begin(loop_arr), std::end(loop_arr), [&](SizeType i) {
      T& eigenval = eval_ptr[to_sizet(i)];

      SizeType i_tile = distr.globalTileLinearIndex(GlobalElementIndex(0, i));
      SizeType i_col = distr.tileElementFromGlobalElement<Coord::Col>(i);
      T* delta = evec_tiles[to_sizet(i_tile)].ptr(TileElementIndex(0, i_col));

      dlaf::internal::laed4_wrapper(static_cast<int>(k), static_cast<int>(i), d_ptr, z_ptr, delta, rho,
                                    &eigenval);
    });
  };

  TileCollector tc{i_begin, i_end};
  pika::dataflow(std::move(rank1_fn), std::move(k_fut), std::move(rho_fut), tc.read(d), tc.read(z),
                 tc.readwrite(evals), tc.readwrite(evecs));
}

// References:
// - lapack 3.10.0, dlaed3.f, line 293
// - LAPACK Working Notes: lawn132, Parallelizing the Divide and Conquer Algorithm for the Symmetric
//   Tridiagonal Eigenvalue Problem on Distributed Memory Architectures, 4.2 Orthogonality
template <class T>
void formEvecs(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
               Matrix<const T, Device::CPU>& diag, Matrix<const T, Device::CPU>& z,
               Matrix<T, Device::CPU>& ws, Matrix<T, Device::CPU>& evecs) {
  using ConstTileType = matrix::Tile<const T, Device::CPU>;
  using TileType = matrix::Tile<T, Device::CPU>;

  const matrix::Distribution& distr = evecs.distribution();

  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      auto mul_rows = [i_subm_el, j_subm_el](SizeType k, const ConstTileType& diag_rows,
                                             const ConstTileType& diag_cols,
                                             const ConstTileType& evecs_tile, const TileType& ws_tile) {
        if (i_subm_el >= k || j_subm_el >= k)
          return;

        SizeType nrows = std::min(k - i_subm_el, evecs_tile.size().rows());
        SizeType ncols = std::min(k - j_subm_el, evecs_tile.size().cols());

        for (SizeType j = 0; j < ncols; ++j) {
          for (SizeType i = 0; i < nrows; ++i) {
            T di = diag_rows(TileElementIndex(i, 0));
            T dj = diag_cols(TileElementIndex(j, 0));
            T evec_el = evecs_tile(TileElementIndex(i, j));
            T& ws_el = ws_tile(TileElementIndex(i, 0));

            // Exact comparison is OK because di and dj come from the same vector
            T weight = (di == dj) ? evec_el : evec_el / (di - dj);
            ws_el = (j == 0) ? weight : weight * ws_el;
          }
        }
      };
      pika::dataflow(pika::unwrapping(std::move(mul_rows)), k_fut, diag.read(GlobalTileIndex(i_tile, 0)),
                     diag.read(GlobalTileIndex(j_tile, 0)), evecs.read(GlobalTileIndex(i_tile, j_tile)),
                     ws(GlobalTileIndex(i_tile, j_tile)));
    }
  }

  // Accumulate weights into the first column of ws
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin + 1; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      auto acc_col_ws_fn = [i_subm_el, j_subm_el](SizeType k, const ConstTileType& ws_tile,
                                                  const TileType& ws_tile_col) {
        if (i_subm_el >= k || j_subm_el >= k)
          return;

        SizeType nrows = std::min(k - i_subm_el, ws_tile.size().rows());

        for (SizeType i = 0; i < nrows; ++i) {
          TileElementIndex idx(i, 0);
          ws_tile_col(idx) = ws_tile_col(idx) * ws_tile(idx);
        }
      };
      pika::dataflow(pika::unwrapping(std::move(acc_col_ws_fn)), k_fut,
                     ws.read(GlobalTileIndex(i_tile, j_tile)), ws(GlobalTileIndex(i_tile, i_begin)));
    }
  }

  // Use the first column of `ws` matrix to compute
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      auto weights_vec = [i_subm_el, j_subm_el](SizeType k, const ConstTileType& z_tile,
                                                const ConstTileType& ws_tile,
                                                const TileType& evecs_tile) {
        if (i_subm_el >= k || j_subm_el >= k)
          return;

        SizeType nrows = std::min(k - i_subm_el, evecs_tile.size().rows());
        SizeType ncols = std::min(k - j_subm_el, evecs_tile.size().cols());

        for (SizeType j = 0; j < ncols; ++j) {
          for (SizeType i = 0; i < nrows; ++i) {
            T ws_el = ws_tile(TileElementIndex(i, 0));
            T z_el = z_tile(TileElementIndex(i, 0));
            T& el_evec = evecs_tile(TileElementIndex(i, j));
            el_evec = std::copysign(std::sqrt(std::abs(ws_el)), z_el) / el_evec;
          }
        }
      };
      pika::dataflow(pika::unwrapping(std::move(weights_vec)), k_fut, z.read(GlobalTileIndex(i_tile, 0)),
                     ws.read(GlobalTileIndex(i_tile, i_begin)), evecs(GlobalTileIndex(i_tile, j_tile)));
    }
  }

  // Calculate the sum of square for each column in each tile
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      auto locnorm_fn = [i_subm_el, j_subm_el](SizeType k, const ConstTileType& evecs_tile,
                                               const TileType& ws_tile) {
        if (i_subm_el >= k || j_subm_el >= k)
          return;

        SizeType nrows = std::min(k - i_subm_el, evecs_tile.size().rows());
        SizeType ncols = std::min(k - j_subm_el, evecs_tile.size().cols());

        for (SizeType j = 0; j < ncols; ++j) {
          T loc_norm = 0;
          for (SizeType i = 0; i < nrows; ++i) {
            T el = evecs_tile(TileElementIndex(i, j));
            loc_norm += el * el;
          }
          ws_tile(TileElementIndex(0, j)) = loc_norm;
        }
      };

      GlobalTileIndex mat_idx(i_tile, j_tile);
      pika::dataflow(pika::unwrapping(std::move(locnorm_fn)), k_fut, evecs.read(mat_idx), ws(mat_idx));
    }
  }

  // Sum the sum of squares into the first row of `ws` submatrix
  for (SizeType i_tile = i_begin + 1; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      auto sum_first_tile_cols_fn = [i_subm_el, j_subm_el](SizeType k, const ConstTileType& ws_tile,
                                                           const TileType& ws_row_tile) {
        if (i_subm_el >= k || j_subm_el >= k)
          return;

        SizeType ncols = std::min(k - j_subm_el, ws_tile.size().cols());
        for (SizeType j = 0; j < ncols; ++j) {
          ws_row_tile(TileElementIndex(0, j)) += ws_tile(TileElementIndex(0, j));
        }
      };

      pika::dataflow(pika::unwrapping(std::move(sum_first_tile_cols_fn)), k_fut,
                     ws.read(GlobalTileIndex(i_tile, j_tile)), ws(GlobalTileIndex(i_begin, j_tile)));
    }
  }

  // Normalize column vectors
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType i_subm_el = distr.globalTileElementDistance<Coord::Row>(i_begin, i_tile);
    for (SizeType j_tile = i_begin; j_tile <= i_end; ++j_tile) {
      SizeType j_subm_el = distr.globalTileElementDistance<Coord::Col>(i_begin, j_tile);
      auto scale_fn = [i_subm_el, j_subm_el](SizeType k, const ConstTileType& ws_tile,
                                             const TileType& evecs_tile) {
        if (i_subm_el >= k || j_subm_el >= k)
          return;

        SizeType nrows = std::min(k - i_subm_el, evecs_tile.size().rows());
        SizeType ncols = std::min(k - j_subm_el, evecs_tile.size().cols());

        for (SizeType j = 0; j < ncols; ++j) {
          for (SizeType i = 0; i < nrows; ++i) {
            T& evecs_el = evecs_tile(TileElementIndex(i, j));
            evecs_el = evecs_el / std::sqrt(ws_tile(TileElementIndex(0, j)));
          }
        }
      };

      pika::dataflow(pika::unwrapping(std::move(scale_fn)), k_fut,
                     ws.read(GlobalTileIndex(i_begin, j_tile)), evecs(GlobalTileIndex(i_tile, j_tile)));
    }
  }
}

template <class T>
void permutateQ(SizeType i_begin, SizeType i_split, SizeType i_end,
                pika::future<ColTypeLens> ct_lens_fut, Matrix<const SizeType, Device::CPU>& index,
                Matrix<T, Device::CPU>& mat_ev, Matrix<T, Device::CPU>& mat_q) {
  SizeType n = problemSize(i_begin, i_end, mat_ev.distribution());
  SizeType n1 = problemSize(i_begin, i_split, mat_ev.distribution());
  SizeType nb = mat_ev.distribution().blockSize().rows();

  auto permutate_fn = [n, n1, nb](auto ct_lens_fut, auto index_fut_tiles, auto mat_ev_fut_tiles,
                                  auto mat_q_fut_tiles) {
    ColTypeLens ct_lens = ct_lens_fut.get();
    TileElementIndex zero(0, 0);
    const SizeType* i_ptr = index_fut_tiles[0].get().ptr(zero);

    auto mat_ev_tiles = pika::unwrap(mat_ev_fut_tiles);
    auto mat_q_tiles = pika::unwrap(mat_q_fut_tiles);

    matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

    // Q1'
    applyPermutations<T, Coord::Col>(GlobalElementIndex(0, 0),
                                     GlobalElementSize(n1, ct_lens.num_uphalf + ct_lens.num_dense), 0,
                                     distr, i_ptr, mat_ev_tiles, mat_q_tiles);

    // Q2'
    applyPermutations<T, Coord::Col>(GlobalElementIndex(n1, 0),
                                     GlobalElementSize(n - n1, ct_lens.num_dense + ct_lens.num_lowhalf),
                                     n1, distr, i_ptr + ct_lens.num_uphalf, mat_ev_tiles, mat_q_tiles);

    // Deflated
    applyPermutations<T, Coord::Col>(GlobalElementIndex(0, n - ct_lens.num_deflated),
                                     GlobalElementSize(n, ct_lens.num_deflated), 0, distr,
                                     i_ptr + n - ct_lens.num_deflated, mat_ev_tiles, mat_q_tiles);
  };

  TileCollector tc{i_begin, i_end};
  pika::dataflow(std::move(permutate_fn), std::move(ct_lens_fut), tc.read(index), tc.readwrite(mat_ev),
                 tc.readwrite(mat_q));
}

template <class T>
void permutateU(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
                Matrix<const SizeType, Device::CPU>& index, Matrix<T, Device::CPU>& mat_in,
                Matrix<T, Device::CPU>& mat_out) {
  SizeType n = problemSize(i_begin, i_end, mat_in.distribution());
  SizeType nb = mat_in.distribution().blockSize().rows();
  auto permute_fn = [n, nb](auto k_fut, auto index_tiles, auto mat_in_tiles_fut,
                            auto mat_out_tiles_fut) {
    SizeType k = k_fut.get();
    TileElementIndex zero(0, 0);
    const SizeType* i_ptr = index_tiles[0].get().ptr(zero);
    auto mat_in_tiles = pika::unwrap(mat_in_tiles_fut);
    auto mat_out_tiles = pika::unwrap(mat_out_tiles_fut);

    matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

    applyPermutations<T, Coord::Row>(GlobalElementIndex(0, 0), GlobalElementSize(k, k), 0, distr, i_ptr,
                                     mat_in_tiles, mat_out_tiles);
  };
  TileCollector tc{i_begin, i_end};
  pika::dataflow(std::move(permute_fn), std::move(k_fut), tc.read(index), tc.readwrite(mat_in),
                 tc.readwrite(mat_out));
}

template <class T, Coord coord>
void permutateQU(SizeType i_begin, SizeType i_end, Matrix<const SizeType, Device::CPU>& index,
                 Matrix<T, Device::CPU>& mat_in, Matrix<T, Device::CPU>& mat_out) {
  SizeType n = problemSize(i_begin, i_end, mat_in.distribution());
  SizeType nb = mat_in.distribution().blockSize().rows();
  auto permute_fn = [n, nb](auto index_tiles, auto mat_in_tiles_fut, auto mat_out_tiles_fut) {
    TileElementIndex zero(0, 0);
    const SizeType* i_ptr = index_tiles[0].get().ptr(zero);
    auto mat_in_tiles = pika::unwrap(mat_in_tiles_fut);
    auto mat_out_tiles = pika::unwrap(mat_out_tiles_fut);

    matrix::Distribution distr(LocalElementSize(n, n), TileElementSize(nb, nb));

    applyPermutations<T, coord>(GlobalElementIndex(0, 0), GlobalElementSize(n, n), 0, distr, i_ptr,
                                mat_in_tiles, mat_out_tiles);
  };
  TileCollector tc{i_begin, i_end};
  pika::dataflow(std::move(permute_fn), tc.read(index), tc.readwrite(mat_in), tc.readwrite(mat_out));
}

// Assumption: Matrices are set to zero.
//
template <class T>
void gemmQU(SizeType i_begin, SizeType i_end, Matrix<const T, Device::CPU>& mat_a,
            Matrix<const T, Device::CPU>& mat_b, Matrix<T, Device::CPU>& mat_c) {
  // Iterate over columns of `c`
  for (SizeType j = i_begin; j <= i_end; ++j) {
    // Iterate over rows of `c`
    for (SizeType i = i_begin; i <= i_end; ++i) {
      // Iterate over columns of `a` and rows of `b`
      for (SizeType k = i_begin; k <= i_end; ++k) {
        // C = alpha * A * B + beta * C
        T beta = 1;
        if (k == i_begin)
          beta = 0;

        auto tile_a = mat_a.read(GlobalTileIndex(i, k));
        auto tile_b = mat_b.read(GlobalTileIndex(k, j));
        auto tile_c = mat_c(GlobalTileIndex(i, j));
        pika::dataflow(pika::unwrapping(tile::internal::gemm_o), blas::Op::NoTrans, blas::Op::NoTrans,
                       T(1), std::move(tile_a), std::move(tile_b), beta, std::move(tile_c));
      }
    }
  }
}

template <class T, Device Source, Device Destination>
void copySubMatrix(SizeType i_begin, SizeType i_end, Matrix<const T, Source>& source,
                   Matrix<T, Destination>& dest) {
  namespace ex = pika::execution::experimental;

  for (SizeType j = i_begin; j <= i_end; ++j) {
    for (SizeType i = i_begin; i <= i_end; ++i) {
      ex::when_all(source.read_sender(LocalTileIndex(i, j)),
                   dest.readwrite_sender(LocalTileIndex(i, j))) |
          matrix::copy(dlaf::internal::Policy<matrix::internal::CopyBackend_v<Source, Destination>>{}) |
          ex::start_detached();
    }
  }
}

template <class T>
void setUnitDiag(SizeType i_begin, SizeType i_end, pika::shared_future<SizeType> k_fut,
                 Matrix<T, Device::CPU>& mat) {
  auto diag_f = [](SizeType k, SizeType tile_begin, matrix::Tile<T, Device::CPU> tile) {
    // If all elements of the tile are after the `k` index reset the offset
    SizeType tile_offset = k - tile_begin;
    if (tile_offset < 0)
      tile_offset = 0;

    // Set all diagonal elements of the tile to 1.
    for (SizeType i = tile_offset; i < tile.size().rows(); ++i) {
      tile(TileElementIndex(i, i)) = 1;
    }
  };

  // Iterate over diagonal tiles
  const matrix::Distribution& distr = mat.distribution();
  for (SizeType i_tile = i_begin; i_tile <= i_end; ++i_tile) {
    SizeType tile_begin = distr.globalElementFromGlobalTileAndTileElement<Coord::Row>(i_tile, 0) -
                          distr.globalElementFromGlobalTileAndTileElement<Coord::Row>(i_begin, 0);
    pika::dataflow(pika::unwrapping(diag_f), k_fut, tile_begin, mat(GlobalTileIndex(i_tile, i_tile)));
  }
}

// Set submatrix to zero
template <class T>
void resetSubMatrix(SizeType i_begin, SizeType i_end, Matrix<T, Device::CPU>& mat) {
  using dlaf::internal::Policy;
  using pika::execution::experimental::start_detached;
  using pika::threads::thread_priority;

  for (SizeType j = i_begin; j <= i_end; ++j) {
    for (SizeType i = i_begin; i <= i_end; ++i) {
      mat.readwrite_sender(GlobalTileIndex(i, j)) |
          tile::set0(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
    }
  }
}

template <class T>
void mergeSubproblems(SizeType i_begin, SizeType i_split, SizeType i_end, WorkSpace<T>& ws,
                      pika::shared_future<T> rho_fut, Matrix<T, Device::CPU>& d,
                      Matrix<T, Device::CPU>& mat_ev) {
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
  pika::shared_future<T> tol_fut = calcTolerance(i_begin, i_end, d, ws.z);

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
  sortIndex(i_begin, i_end, pika::make_ready_future(n1), d, ws.i1, ws.i2);
  pika::future<std::vector<GivensRotation<T>>> rots_fut =
      applyDeflation(i_begin, i_end, rho_fut, tol_fut, ws.i2, d, ws.z, ws.c);

  TileCollector tc{i_begin, i_end};
  pika::dataflow(pika::unwrapping(applyGivensRotationsToMatrixColumns<T>), n, nb, std::move(rots_fut),
                 tc.readwrite(mat_ev));

  // Step #2
  //
  //    i2 (in)  : initial <--- pre_sorted
  //    i3 (out) : initial <--- deflated
  //
  // - reorder `d -> dtmp`, `z -> ztmp`, using `i3` such that deflated entries are at the bottom.
  // - solve the rank-1 problem and save eigenvalues in `dtmp` and eigenvectors in `ws.mat1`.
  // - set deflated diagonal entries of `U` to 1 (temporary solution until optimized GEMM is implemented)
  //
  pika::shared_future<SizeType> k_fut =
      stablePartitionIndexForDeflation(i_begin, i_end, ws.c, ws.i2, ws.i3);
  applyIndex(i_begin, i_end, ws.i3, d, ws.dtmp);
  applyIndex(i_begin, i_end, ws.i3, ws.z, ws.ztmp);
  copyVector(i_begin, i_end, ws.dtmp, d);
  resetSubMatrix(i_begin, i_end, ws.mat1);
  solveRank1Problem(i_begin, i_end, k_fut, rho_fut, d, ws.ztmp, ws.dtmp, ws.mat1);
  formEvecs(i_begin, i_end, k_fut, d, ws.ztmp, ws.mat2, ws.mat1);
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
  permutateQU<T, Coord::Row>(i_begin, i_end, ws.i2, ws.mat1, ws.mat2);  // permutate U to match Q
  gemmQU(i_begin, i_end, mat_ev, ws.mat2, ws.mat1);

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
  applyIndex(i_begin, i_end, ws.i2, ws.dtmp, d);
  permutateQU<T, Coord::Col>(i_begin, i_end, ws.i2, ws.mat1, mat_ev);

  // ----------- ! Optimized approach

  //  // Step #4
  //  //
  //  //    i1 (in)  : initial  <--- post_sorted
  //  //       (out) : initial  <--- matmul
  //  //    i2 (in)  : deflated <--- post_sorted
  //  //       (out) : deflated <--- matmul
  //  //
  //  auto ctlens_fut = partitionIndexForMatrixMultiplication(i_begin, i_end, ws.ctmp, ws.i2);
  //  partitionIndexForMatrixMultiplication(i_begin, i_end, ws.c, ws.i1);
  //
  //  // Permutate columns of the `Q` matrix to arrange in multiplication form
  //  resetSubMatrix(i_begin, i_end, ws.mat2);
  //  permutateQ(i_begin, i_split, i_end, std::move(ctlens_fut), ws.i1, mat_ev, ws.mat2);
  //
  //  // Permutate rows of the `U` matrix to arrange in multiplication form
  //  resetSubMatrix(i_begin, i_end, mat_ev);
  //  permutateU(i_begin, i_end, k_fut, ws.i2, ws.mat1, mat_ev);

  //  // Matrix multiply Q' * U' to get the eigenvectors of the merged system
  //  gemmQU(i_begin, i_end, ws.mat2, mat_ev, ws.mat1);
  //
  //  // Copy back into `mat_ev`
  //  copySubMatrix(i_begin, i_end, ws.mat1, mat_ev);
  //
  //  matrix::print(format::csv{}, "fin evecs", mat_ev);
}
}
