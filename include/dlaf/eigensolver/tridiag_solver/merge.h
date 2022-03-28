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
#include <pika/parallel/algorithms/partition.hpp>
#include <pika/parallel/algorithms/sort.hpp>
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
  Matrix<T, Device::CPU> d_defl;
  // Holds the values of Cuppen's rank-1 vector
  Matrix<T, Device::CPU> z;
  // Holds the values of the rank-1 update vector sorted corresponding to `d_defl`
  Matrix<T, Device::CPU> z_defl;

  // An index map from sorted (in ascending order) indices to initial indices of D
  Matrix<SizeType, Device::CPU> isorted;
  // An index map from sorted (in ascending order) indices of D to sorted indices of the diagonal after
  // the deflated system is solved and the deflated part is incorporated Holds indices/permutations of
  // the rows of U that bring it in Q-U matrix multiplication form
  Matrix<SizeType, Device::CPU> ideflated;
  // An index map from Q-U matrix multiplication form `ideflated`
  Matrix<SizeType, Device::CPU> imatmul;
  // Assigns a type to each column of Q which is used to calculate the permutation indices for Q and U
  // that bring them in matrix multiplication form.
  Matrix<internal::ColType, Device::CPU> coltypes;
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
                      Matrix<internal::ColType, Device::CPU>(vec_size, vec_tile_size)};
}

// Calculates the combined problem size of the two subproblems that are being merged
inline SizeType combinedProblemSize(SizeType i_begin, SizeType i_end,
                                    const matrix::Distribution& distr) {
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

template <class T>
std::vector<pika::future<matrix::Tile<T, Device::CPU>>> collectReadWriteTiles(
    GlobalTileIndex begin, GlobalTileIndex end, Matrix<T, Device::CPU>& mat) {
  using FutureTileType = pika::future<matrix::Tile<T, Device::CPU>>;
  return collectTiles<FutureTileType>(begin, end, mat);
}

template <class T>
std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> collectReadTiles(
    GlobalTileIndex begin, GlobalTileIndex end, Matrix<const T, Device::CPU>& mat) {
  using FutureTileType = pika::shared_future<matrix::Tile<const T, Device::CPU>>;
  return collectTiles<FutureTileType, const T>(begin, end, mat);
}

// Save the index of the elements (perm_d) in ascending order of the diagonal (d)
//
// NOTE: `d_tiles` should be `const T` but there seems to be a compile-time issue
template <class T>
void sortAscendingBasedOnDiagonal(
    SizeType n, std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> d_tiles,
    std::vector<pika::future<matrix::Tile<SizeType, Device::CPU>>> perm_tiles) {
  TileElementIndex zero_idx(0, 0);
  const T* d_ptr = d_tiles[0].get().ptr(zero_idx);
  SizeType* perm_ptr = perm_tiles[0].get().ptr(zero_idx);

  pika::sort(pika::execution::par, perm_ptr, perm_ptr + n,
             [d_ptr](SizeType i1, SizeType i2) { return d_ptr[i1] < d_ptr[i2]; });
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

// @param zt tile from the `z`-vector
// @param ct tile from the `coltypes` vector
template <class T>
void updateTileColumnTypesBasedOnZvecNearlyZero(T tol, T rho,
                                                const matrix::Tile<const T, Device::CPU>& zt,
                                                const matrix::Tile<ColType, Device::CPU>& ct) {
  SizeType len = zt.size().rows();
  for (SizeType i = 0; i < len; ++i) {
    TileElementIndex idx(i, 0);
    if (std::abs(rho * zt(idx)) < tol) {
      ct(idx) = ColType::Deflated;
    }
  }
}

DLAF_MAKE_CALLABLE_OBJECT(updateTileColumnTypesBasedOnZvecNearlyZero);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(updateTileColumnTypesBasedOnZvecNearlyZero,
                                     updateTileColumnTypesBasedOnZvecNearlyZero_o)

template <class T>
void updateVectorColumnTypesBasedOnZvecNearlyZero(SizeType i_begin, SizeType i_end,
                                                  pika::shared_future<T> tol_fut,
                                                  pika::shared_future<T> rho_fut,
                                                  Matrix<const T, Device::CPU>& z,
                                                  Matrix<ColType, Device::CPU>& coltypes) {
  using pika::threads::thread_priority;
  using pika::execution::experimental::start_detached;
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;

  for (SizeType i = i_begin; i <= i_end; ++i) {
    LocalTileIndex idx(i, 0);
    whenAllLift(std::move(tol_fut), std::move(rho_fut), z.read_sender(idx),
                coltypes.readwrite_sender(idx)) |
        updateTileColumnTypesBasedOnZvecNearlyZero(Policy<Backend::MC>(thread_priority::normal)) |
        start_detached();
  }
}

template <class T>
struct GivensRotation {
  // GivensRotation() = default;
  // GivensRotation(const GivensRotation&) = default;
  // GivensRotation& operator=(const GivensRotation&) = default;
  // GivensRotation(SizeType i, SizeType j, T c, T s) : i{i}, j{j}, c{c}, s{s} {}
  SizeType i;  // the first column index
  SizeType j;  // the second column index
  T c;         // cosine
  T s;         // sine
};

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

template <class T>
void updateDiagValuesWithGivensCoeff(T c, T s, T& d1, T& d2) {
  d1 = d1 * c * c + d2 * s * s;
  d2 = d1 * s * s + d2 * c * c;
}

// Assumption 1: The algorithm assumes that the diagonal `dptr` is sorted in ascending order with
// corresponding `zptr` and `coltyps` arrays.
//
// Assumption 2: The `coltyps` array stores as `ColType::Deflated` at all indices corresponding to zeroes
// in `zptr`.
//
// Note: parallelizing this algorithm is non-trivial because the deflation regions due to Givens
// rotations can cross over tiles and are of unknown length. However such algorithm is unlikely to
// benefit much from parallelization anyway as it is quite light on flops and it appears memory bound.
//
// Returns an array of Given's rotations used to update the colunmns of the eigenvector matrix Q
template <class T>
std::vector<GivensRotation<T>> applyDeflationWithGivensRotation(T tol, SizeType len, T* d_ptr,
                                                                const SizeType* i_ptr, T* z_ptr,
                                                                ColType* c_ptr) {
  std::vector<GivensRotation<T>> rots;
  rots.reserve(to_sizet(len));

  SizeType i1 = 0;  // index of 1st element in the Givens rotation

  // Iterate over the indices of the sorted elements in pair (i1, i2) where i1 < i2 for every iteration
  for (SizeType i2 = 1; i2 < len; ++i2) {
    // Indices of elements sorted in ascending order
    SizeType i1s = i_ptr[i1];
    SizeType i2s = i_ptr[i2];

    T& d1 = d_ptr[i1s];
    T& d2 = d_ptr[i2s];
    T& z1 = z_ptr[i1s];
    T& z2 = z_ptr[i2s];
    ColType& c1 = c_ptr[i1s];
    ColType& c2 = c_ptr[i2s];

    // if z2 == 0 go to the next iteration
    if (c2 == ColType::Deflated)
      continue;

    if (c1 != ColType::Deflated && diagonalValuesNearlyEqual(tol, d1, d2, z1, z2)) {
      // if z1 != 0 and z2 != 0 and d1 = d2 apply Givens rotation
      T c, s;
      blas::rotg(&z1, &z2, &c, &s);
      updateDiagValuesWithGivensCoeff(c, s, d1, d2);

      rots.push_back(GivensRotation<T>{i1s, i2s, c, s});
      //  Set the the `i1` column as "Dense" if the `i2` column has opposite non-zero structure (i.e if
      //  one comes from Q1 and the other from Q2 or vice-versa)
      if ((c1 == ColType::UpperHalf && c2 == ColType::LowerHalf) ||
          (c1 == ColType::LowerHalf && c2 == ColType::UpperHalf)) {
        c1 = ColType::Dense;
      }
      c2 = ColType::Deflated;
    }
    else {
      // if z2 != 0 but z1 == 0 or d1 != d2 then use the index of i2 as the new 1st element in the Givens rotation
      i1 = i2;
    }
  }

  return rots;
}

template <class T>
std::vector<GivensRotation<T>> applyGivensDeflation(
    SizeType n, pika::shared_future<T> tol,
    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> d_tiles,
    std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> perm_d_tiles,
    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> z_tiles,
    std::vector<pika::future<matrix::Tile<ColType, Device::CPU>>> coltypes_tiles) {
  TileElementIndex zero_idx(0, 0);
  T* d_ptr = d_tiles[0].get().ptr(zero_idx);
  T* z_ptr = z_tiles[0].get().ptr(zero_idx);
  ColType* c_ptr = coltypes_tiles[0].get().ptr(zero_idx);
  const SizeType* i_ptr = perm_d_tiles[0].get().ptr(zero_idx);

  return applyDeflationWithGivensRotation(tol.get(), n, d_ptr, i_ptr, z_ptr, c_ptr);
}

struct QLens {
  SizeType num_uphalf;
  SizeType num_dense;
  SizeType num_lowhalf;
  SizeType num_deflated;
};

// Partitions `p_ptr` based on a `ctype` in `c_ptr` array.
//
// Returns the number of elements in the second partition.
inline SizeType partitionColType(SizeType len, ColType ctype, const ColType* c_ptr, SizeType* i_ptr) {
  auto it = pika::partition(pika::execution::par, i_ptr, i_ptr + len,
                            [ctype, c_ptr](SizeType i) { return ctype != c_ptr[i]; });
  return len - std::distance(i_ptr, it);
}

// Partition `coltypes` to get the indices of the matrix-multiplication form
inline QLens setMatrixMultiplicationIndex(
    SizeType n, std::vector<pika::shared_future<matrix::Tile<const ColType, Device::CPU>>> c_tiles,
    std::vector<pika::future<matrix::Tile<SizeType, Device::CPU>>> perm_q_tiles) {
  TileElementIndex zero_idx(0, 0);
  const ColType* c_ptr = c_tiles[0].get().ptr(zero_idx);
  SizeType* i_ptr = perm_q_tiles[0].get().ptr(zero_idx);
  QLens ql;
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
void applyGivensRotationsToMatrixColumns(std::vector<GivensRotation<T>> rots, SizeType n,
                                         SizeType ncol_tiles, const matrix::Distribution& distr,
                                         std::vector<matrix::Tile<T, Device::CPU>> tiles) {
  for (const GivensRotation<T>& rot : rots) {
    // Get the index of the tile that has column `rot.i`
    SizeType i_tile = distr.globalTileFromGlobalElement<Coord::Col>(rot.i);
    // Get the index of the `rot.i` column within the tile
    SizeType i_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.i);
    // Get the pointer to the first element of the `rot.i` column
    //
    // Note: this works because `tiles` come from a matrix with column-major layout
    T* x = tiles[to_sizet(i_tile * ncol_tiles)].ptr(TileElementIndex(0, i_el));

    // Get the index of the tile that has column `rot.j`
    SizeType j_tile = distr.globalTileFromGlobalElement<Coord::Col>(rot.j);
    // Get the index of the `rot.j` column within the tile
    SizeType j_el = distr.tileElementFromGlobalElement<Coord::Col>(rot.j);
    // Get the pointer to the first element of the `rot.j` column
    //
    // Note: this works because `tiles` come from a matrix with column-major layout
    T* y = tiles[to_sizet(j_tile * ncol_tiles)].ptr(TileElementIndex(0, j_el));

    DLAF_ASSERT(i_el != j_el, "");

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
template <class T>
void applyPermutationIndexToMatrixQ(
    SizeType i_begin, SizeType i_end,
    std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> perm_tiles,
    const matrix::Distribution& distr,
    std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> in_tiles,
    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> out_tiles) {
  SizeType ncol_tiles = i_end - i_begin + 1;

  // Iterate over columns of `in_tiles` and use the permutation index `perm_tiles` to copy columns from
  for (SizeType j_out_tile = i_begin; j_out_tile <= i_end; ++j_out_tile) {
    // Get the indices of permutations correspo
    const auto& perm_tile = perm_tiles[to_sizet(j_out_tile)].get();
    for (SizeType i_out_tile = i_begin; i_out_tile <= i_end; ++i_out_tile) {
      auto out_tile = out_tiles[to_sizet(i_out_tile + j_out_tile * ncol_tiles)].get();
      TileElementSize sz_out_tile = out_tile.size();

      // Iterate over columns of `out_tile`
      for (SizeType j_out_el = 0; j_out_el < sz_out_tile.cols(); ++j_out_el) {
        // Get the index of the global column
        SizeType j_in_gl_el = perm_tile(TileElementIndex(j_out_el, 0));
        // Get the index of the tile that has column `gl_col`
        SizeType j_in_tile = distr.globalTileFromGlobalElement<Coord::Col>(j_in_gl_el);
        const auto& in_tile = in_tiles[to_sizet(i_out_tile + j_in_tile * ncol_tiles)].get();
        // Get the index of the `gl_col` column within the tile
        SizeType j_in_el = distr.tileElementFromGlobalElement<Coord::Col>(j_in_gl_el);
        // Copy a column from `in_tile` into `out_tile`
        for (SizeType i_out_el = 0; i_out_el < sz_out_tile.rows(); ++i_out_el) {
          out_tile(TileElementIndex(i_out_el, j_out_el)) = in_tile(TileElementIndex(i_out_el, j_in_el));
        }
      }
    }
  }
}

template <class T>
void offloadInAscendingOrder(
    SizeType n, std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> index,
    std::vector<pika::shared_future<matrix::Tile<const ColType, Device::CPU>>> coltypes,
    std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> in_tiles,
    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> out_tiles) {
  // const QLens& qlens = qlens_fut.get();

  TileElementIndex zero(0, 0);
  const SizeType* i_ptr = index[0].get().ptr(zero);
  const ColType* c_ptr = coltypes[0].get().ptr(zero);
  const T* in_ptr = in_tiles[0].get().ptr(zero);
  T* out_ptr = out_tiles[0].get().ptr(zero);

  SizeType k = 0;  // index of non-deflated entry of the output pointer
  // Iterates over all elements of `in_tiles` in ascending order and saves non-deflated values in `i_ptr`
  for (SizeType i = 0; i < n; ++i) {
    SizeType is = i_ptr[i];  // map the sorted index `i` to the original index `is`

    // skip deflated entries
    if (c_ptr[is] == ColType::Deflated)
      continue;

    out_ptr[k] = in_ptr[is];
    ++k;
  }
}

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
void buildRank1EigVecMatrix(
    SizeType n, pika::shared_future<QLens> qlens_fut,
    std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> d_defl,
    pika::shared_future<T> rho_fut,
    std::vector<pika::shared_future<matrix::Tile<const T, Device::CPU>>> z_defl,
    std::vector<pika::shared_future<matrix::Tile<const ColType, Device::CPU>>> coltypes,
    std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> perm_d,
    std::vector<pika::future<matrix::Tile<T, Device::CPU>>> d,
    std::vector<pika::shared_future<matrix::Tile<const SizeType, Device::CPU>>> perm_u,
    const matrix::Distribution& distr, std::vector<pika::future<matrix::Tile<T, Device::CPU>>> mat_uws) {
  TileElementIndex zero(0, 0);
  const QLens& qlens = qlens_fut.get();
  SizeType k = qlens.num_dense + qlens.num_lowhalf + qlens.num_uphalf;
  const T* d_defl_ptr = d_defl[0].get().ptr(zero);
  const SizeType* s_ptr = perm_d[0].get().ptr(zero);
  T rho = rho_fut.get();
  T* d_ptr = d[0].get().ptr(zero);
  const SizeType* u_ptr = perm_u[0].get().ptr(zero);
  const T* z_ptr = z_defl[0].get().ptr(zero);
  const ColType* c_ptr = coltypes[0].get().ptr(zero);

  for (SizeType i = 0; i < n; ++i) {
    SizeType is = s_ptr[i];  // original index
    if (c_ptr[is] == ColType::Deflated)
      continue;

    SizeType iu = u_ptr[i];  // matrix-multiplication index
    SizeType i_tile = distr.globalTileFromGlobalElement<Coord::Col>(iu);
    SizeType i_col = distr.tileElementFromGlobalElement<Coord::Col>(iu);

    T* delta = mat_uws[to_sizet(i_tile)].get().ptr(TileElementIndex(0, i_col));
    T& eigenval = d_ptr[to_sizet(is)];
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
  // Assemble the rank-1 update vector `z` from the last row of Q1 and the first row of Q2
  assembleZVec(i_begin, i_split, i_end, rho_fut, mat_ev, ws.z);

  // Double `rho` to account for the normalization of `z` and make sure `rho > 0` for the root solver laed4
  rho_fut = scaleRho(rho_fut);

  // Calculate the tolerance used for deflation
  pika::future<T> dmax_fut = maxVectorElement(i_begin, i_end, d);
  pika::future<T> zmax_fut = maxVectorElement(i_begin, i_end, ws.z);
  pika::shared_future<T> tol_fut =
      pika::dataflow(pika::unwrapping(calcTolerance<T>), std::move(dmax_fut), std::move(zmax_fut));

  // Initialize permutation indices
  initIndex(i_begin, i_end, ws.isorted);
  initIndex(i_begin, i_end, ws.ideflated);
  initIndex(i_begin, i_end, ws.imatmul);

  // Initialize coltypes
  initColTypes(i_begin, i_split, i_end, ws.coltypes);

  // Calculate the merged size of the subproblem
  SizeType n = combinedProblemSize(i_begin, i_end, mat_ev.distribution());

  // The tile indices of column vectors `d`, `z`, `coltypes`, `d_defl` and `z_defl`
  GlobalTileIndex col_begin(i_begin, 0);
  GlobalTileIndex col_end(i_end, 0);

  // Sort the diagonal in ascending order and initialize the corresponding permutation index
  //
  // Note: `pika::unwrapping()` is not used because it requires that `Tile<>` is copiable at compile-time.
  pika::dataflow(sortAscendingBasedOnDiagonal<T>, n, collectReadTiles(col_begin, col_end, d),
                 collectReadWriteTiles(col_begin, col_end, ws.isorted));

  // Apply deflation with Givens rotations on `d`, `z` and `coltypes` using the sorted index `perm_d` and
  // return metadata used for applying the rotations on `Q`
  //
  // Note: `pika::unwrapping()` is not used because it requires that `Tile<>` is copiable at compile-time.
  pika::shared_future<std::vector<GivensRotation<T>>> rots_fut =
      pika::dataflow(applyGivensDeflation<T>, n, tol_fut, collectReadWriteTiles(col_begin, col_end, d),
                     collectReadTiles(col_begin, col_end, ws.isorted),
                     collectReadWriteTiles(col_begin, col_end, ws.z),
                     collectReadWriteTiles(col_begin, col_end, ws.coltypes));

  // Build the permutaion index for the `Q`-`U` multiplication and return the sizes of each segment.
  //
  // Note: `pika::unwrapping()` is not used because it requires that `Tile<>` is copiable at
  // compile-time.
  pika::shared_future<QLens> qlens_fut =
      pika::dataflow(setMatrixMultiplicationIndex, n, collectReadTiles(col_begin, col_end, ws.coltypes),
                     collectReadWriteTiles(col_begin, col_end, ws.imatmul));

  // Apply Givens rotations to `Q`
  SizeType ncol_tiles = i_end - i_begin + 1;
  GlobalTileIndex ev_begin(i_begin, i_begin);
  GlobalTileIndex ev_end(i_end, i_end);
  pika::dataflow(pika::unwrapping(applyGivensRotationsToMatrixColumns<T>), rots_fut, n, ncol_tiles,
                 mat_ev.distribution(), collectReadWriteTiles(ev_begin, ev_end, mat_ev));

  // Use the permutation index `perm_q` on the columns of `mat_ev` and save the result to `mat_q`
  pika::dataflow(applyPermutationIndexToMatrixQ<T>, i_begin, i_end,
                 collectReadTiles(col_begin, col_end, ws.imatmul), mat_ev.distribution(),
                 collectReadTiles(ev_begin, ev_end, mat_ev),
                 collectReadWriteTiles(ev_begin, ev_end, ws.mat));

  // Invert the `perm_q` index and map sorted indices to matrix-multiplication indices in `perm_u`
  // pika::dataflow(setPermutationsU, n, collectReadTiles(col_begin, col_end, ws.isorted),
  //               collectReadWriteTiles(col_begin, col_end, ws.imatmul),
  //               collectReadWriteTiles(col_begin, col_end, perm_u));

  // Offload the non-deflated diagonal values from `d` in ascedning order to `d_defl` using the index
  // `perm_d` and `coltypes`.
  pika::dataflow(offloadInAscendingOrder<T>, n, collectReadTiles(col_begin, col_end, ws.isorted),
                 collectReadTiles(col_begin, col_end, ws.coltypes),
                 collectReadTiles(col_begin, col_end, d),
                 collectReadWriteTiles(col_begin, col_end, ws.d_defl));

  // Offload the non-zero rank-1 values from `z` to `z_defl` using the permutations index `perm_d` and `coltypes`.
  pika::dataflow(offloadInAscendingOrder<T>, n, collectReadTiles(col_begin, col_end, ws.isorted),
                 collectReadTiles(col_begin, col_end, ws.coltypes),
                 collectReadTiles(col_begin, col_end, ws.z),
                 collectReadWriteTiles(col_begin, col_end, ws.z_defl));

  // Build the matrix of eigenvectors `U^T` of the deflated rank-1 problem `d_defl + rho * z_defl *
  // z_defl^T` using the root solver `laed4`.
  // pika::dataflow(buildRank1EigVecMatrix<T>, n, qlens_fut, collectReadTiles(col_begin, col_end, ws.d_defl),
  //               rho_fut, collectReadTiles(col_begin, col_end, ws.z_defl),
  //               collectReadTiles(col_begin, col_end, ws.coltypes),
  //               collectReadTiles(col_begin, col_end, ws.isorted),
  //               collectReadWriteTiles(col_begin, col_end, d),
  //               collectReadTiles(col_begin, col_end, ws.ideflated), ws.mat.distribution(),
  //               collectReadWriteTiles(ev_begin, ev_end, mat_uws));

  // GEMM `mat_q` holding Q and `mat_u` holding U^T into `mat_ev`
  //
  // Note: the transpose of `mat_u` is used here to recover U
  // pika::dataflow(gemmQU<T>, qlens_fut, collectReadTiles(ev_begin, ev_end, mat_qws),
  //               collectReadTiles(ev_begin, ev_end, mat_uws),
  //               collectReadWriteTiles(ev_begin, ev_end, mat_ev));
}

}
