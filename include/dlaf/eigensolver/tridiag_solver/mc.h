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

#include <algorithm>

#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include "dlaf/common/callable_object.h"
#include "dlaf/eigensolver/tridiag_solver/api.h"
#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

#include "dlaf/matrix/print_csv.h"

namespace dlaf::eigensolver::internal {

template <class T>
struct TridiagSolver<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_trd, Matrix<T, Device::CPU>& d,
                   Matrix<T, Device::CPU>& mat_ev);
};

/// Splits [i_begin, i_end] in the middle and waits for all splits on [i_begin, i_middle] and [i_middle +
/// 1, i_end] before saving the triad <i_begin, i_middle, i_end> into `indices`.
///
/// The recursive calls span a binary tree which is traversed in depth first left-right-root order. That
/// is also the order of triads in `indices`.
///
/// Note: the intervals are all closed!
///
inline void splitIntervalInTheMiddleRecursively(
    SizeType i_begin, SizeType i_end, std::vector<std::tuple<SizeType, SizeType, SizeType>>& indices) {
  if (i_begin == i_end)
    return;
  SizeType i_middle = (i_begin + i_end) / 2;
  splitIntervalInTheMiddleRecursively(i_begin, i_middle, indices);
  splitIntervalInTheMiddleRecursively(i_middle + 1, i_end, indices);
  indices.emplace_back(i_begin, i_middle, i_end);
}

/// Generates an array of triad indices. Each triad is composed of begin <= middle < end indices and
/// represents two intervals [begin, middle] and [middle + 1, end]. The two intervals are the subproblems
/// that have to be merged to arrive at [begin, end].
///
/// Note: the intervals are all closed!
///
inline std::vector<std::tuple<SizeType, SizeType, SizeType>> generateSubproblemIndices(SizeType n) {
  DLAF_ASSERT(n > 0, n);
  std::vector<std::tuple<SizeType, SizeType, SizeType>> indices;
  indices.reserve(to_sizet(n));
  splitIntervalInTheMiddleRecursively(0, n - 1, indices);
  return indices;
}

// Cuppen's decomposition
//
// Substracts the offdiagonal element at the split from the top and bottom diagonal elements and returns
// the offdiagonal element. The split is between the last row of the top tile and the first row of the
// bottom tile.
//
template <class T>
T cuppensTileDecomposition(const matrix::Tile<T, Device::CPU>& top,
                           const matrix::Tile<T, Device::CPU>& bottom) {
  (void) top;
  (void) bottom;

  T offdiag_val = top(TileElementIndex{top.size().rows() - 1, 1});
  T& top_diag_val = top(TileElementIndex{top.size().rows() - 1, 0});
  T& bottom_diag_val = bottom(TileElementIndex{0, 0});

  // Refence: Lapack working notes: LAWN 69, Serial Cuppen algorithm, Chapter 3
  //
  top_diag_val -= std::abs(offdiag_val);
  bottom_diag_val -= std::abs(offdiag_val);
  return offdiag_val;
}

DLAF_MAKE_CALLABLE_OBJECT(cuppensTileDecomposition);

template <class T>
std::vector<pika::shared_future<T>> cuppensDecomposition(Matrix<T, Device::CPU>& mat_trd) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  SizeType i_end = mat_trd.distribution().nrTiles().rows() - 1;
  std::vector<pika::shared_future<T>> offdiag_vals;
  offdiag_vals.reserve(to_sizet(i_end));

  for (SizeType i_split = 0; i_split < i_end; ++i_split) {
    // Cuppen's tridiagonal decomposition
    offdiag_vals.push_back(
        ex::when_all(mat_trd(LocalTileIndex(i_split, 0)), mat_trd(LocalTileIndex(i_split + 1, 0))) |
        di::transform(di::Policy<Backend::MC>(), cuppensTileDecomposition_o) | ex::make_future());
  }
  return offdiag_vals;
}

// Solve leaf eigensystem with stedc
template <class T>
void solveLeaf(Matrix<T, Device::CPU>& mat_trd, Matrix<T, Device::CPU>& mat_ev) {
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;
  using pika::threads::thread_priority;

  SizeType ntiles = mat_trd.distribution().nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    whenAllLift(mat_trd.readwrite_sender(LocalTileIndex(i, 0)),
                mat_ev.readwrite_sender(LocalTileIndex(i, i))) |
        tile::stedc(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
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
void offloadDiagonal(Matrix<const T, Device::CPU>& mat_trd, Matrix<T, Device::CPU>& d) {
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;
  using pika::execution::experimental::start_detached;
  using pika::threads::thread_priority;

  for (SizeType i = 0; i < d.distribution().nrTiles().rows(); ++i) {
    whenAllLift(mat_trd.read_sender(GlobalTileIndex(i, 0)), d.readwrite_sender(GlobalTileIndex(i, 0))) |
        copyDiagTile(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

/// Notation:
///
/// nb - the block/tile size of all matrices and vectors
/// n1 - the size of the top subproblem
/// n2 - the size of the bottom subproblem
/// Q1 - (n1 x n1) the orthogonal matrix of the top subproblem
/// Q2 - (n2 x n2) the orthogonal matrix of the bottom subproblem
/// n := n1 + n2, the size of the merged problem
///
///      ┌───┬───┐
///      │Q1 │   │
/// Q := ├───┼───┤ , (n x n) orthogonal matrix composed of the top and bottom subproblems
///      │   │Q2 │
///      └───┴───┘
/// D                 := diag(Q), (n x 1) the diagonal of Q
/// z                 := (n x 1) rank 1 update vector
/// rho               := rank 1 update scaling factor
/// D + rho*z*z^T     := rank 1 update problem
/// U                 := (n x n) matrix of eigenvectors of the rank 1 update problem:
///
/// k                 := the size of the deflated rank 1 update problem (k <= n)
/// D'                := (k x 1), deflated D
/// z'                := (k x 1), deflated z
/// D' + rho*z'*z'^T  := deflated rank 1 update problem
/// U'                := (k x k) matrix of eigenvectors of the deflated rank 1 update problem
///
/// l1  := number of columns of the top subproblem after deflation, l1 <= n1
/// l2  := number of columns of the bottom subproblem after deflation, l2 <= n2
/// Q1' := (n1 x l1) the non-deflated part of Q1 (l1 <= n1)
/// Q2' := (n2 x l2) the non-deflated part of Q2 (l2 <= n2)
/// Qd  := (n-k x n) the deflated parts of Q1 and Q2
/// U1' := (l1 x k) is the first l1 rows of U'
/// U2' := (l2 x k) is the last l2 rows of U'
/// I   := (n-k x n-k) identity matrix
/// P   := (n x n) permutation matrix used to bring Q and U into multiplication form
///
/// Q-U multiplication form to arrive at the eigenvectors of the merged problem:
///
/// ┌────────┬──────┬────┐       ┌───────────────┬────┐       ┌───────────────┬────┐
/// │        │      │    │       │               │    │       │               │    │
/// │  Q1'   │      │    │       │               │    │       │    Q1'xU1'    │    │
/// │        │      │    │       │   U1'         │    │       │               │    │
/// │        │      │    │       │         ──────┤    │       │               │    │
/// ├──────┬─┴──────┤ Qd │       ├───────        │    │       ├───────────────┤ Qd │
/// │      │        │    │   X   │          U2'  │    │   =   │               │    │
/// │      │        │    │       │               │    │       │    Q2'xU2'    │    │
/// │      │  Q2'   │    │       ├───────────────┼────┤       │               │    │
/// │      │        │    │       │               │ I  │       │               │    │
/// │      │        │    │       │               │    │       │               │    │
/// └──────┴────────┴────┘       └───────────────┴────┘       └───────────────┴────┘
///
/// Note:
/// 1. U1' and U2' may overlap (in practice they almost always do)
/// 2. The overlap between U1' and U2' matches the number of shared columns between Q1' and Q2'
/// 3. The overlap region is due to deflation via Givens rotations of a column vector from Q1 with a
///    column vector of Q2.
template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_trd,
                                                      Matrix<T, Device::CPU>& d,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  // Set `mat_ev` to `zero` (needed for Given's rotation to make sure no random values are picked up)
  matrix::util::set0<Backend::MC, T, Device::CPU>(pika::threads::thread_priority::normal, mat_ev);

  // Cuppen's decomposition
  std::vector<pika::shared_future<T>> offdiag_vals = cuppensDecomposition(mat_trd);

  // Solve with stedc for each tile of `mat_trd` (nb x 2) and save eigenvectors in diagonal tiles of
  // `mat_ev` (nb x nb)
  solveLeaf(mat_trd, mat_ev);

  // Offload the diagonal from `mat_trd` to `d`
  offloadDiagonal(mat_trd, d);

  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& distr = mat_ev.distribution();
  WorkSpace<T> ws = initWorkSpace<T>(distr);

  // Each triad represents two subproblems to be merged
  for (auto [i_begin, i_split, i_end] : generateSubproblemIndices(distr.nrTiles().rows())) {
    mergeSubproblems(i_begin, i_split, i_end, ws, offdiag_vals[to_sizet(i_split)], d, mat_ev);
  }
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
