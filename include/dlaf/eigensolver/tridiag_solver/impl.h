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

#include <pika/execution.hpp>
#include <pika/thread.hpp>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include "dlaf/common/callable_object.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/eigensolver/tridiag_solver/api.h"
#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/eigensolver/tridiag_solver/merge.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/types.h"

#include "dlaf/matrix/print_csv.h"

namespace dlaf::eigensolver::internal {

// Splits [i_begin, i_end] in the middle and waits for all splits on [i_begin, i_middle] and [i_middle +
// 1, i_end] before saving the triad <i_begin, i_middle, i_end> into `indices`.
//
// The recursive calls span a binary tree which is traversed in depth first left-right-root order. That
// is also the order of triads in `indices`.
//
// Note: the intervals are all closed!
//
inline void splitIntervalInTheMiddleRecursively(
    SizeType i_begin, SizeType i_end, std::vector<std::tuple<SizeType, SizeType, SizeType>>& indices) {
  if (i_begin == i_end)
    return;
  SizeType i_middle = (i_begin + i_end) / 2;
  splitIntervalInTheMiddleRecursively(i_begin, i_middle, indices);
  splitIntervalInTheMiddleRecursively(i_middle + 1, i_end, indices);
  indices.emplace_back(i_begin, i_middle, i_end);
}

// Generates an array of triad indices. Each triad is composed of begin <= middle < end indices and
// represents two intervals [begin, middle] and [middle + 1, end]. The two intervals are the subproblems
// that have to be merged to arrive at [begin, end].
//
// Note: the intervals are all closed!
//
inline std::vector<std::tuple<SizeType, SizeType, SizeType>> generateSubproblemIndices(SizeType n) {
  if (n == 0)
    return {};

  std::vector<std::tuple<SizeType, SizeType, SizeType>> indices;
  indices.reserve(to_sizet(n));
  splitIntervalInTheMiddleRecursively(0, n - 1, indices);
  return indices;
}

template <class T, Device D>
auto cuppensDecomposition(Matrix<T, D>& tridiag) {
  namespace ex = pika::execution::experimental;
  using sender_type = decltype(ex::split(
      cuppensDecompAsync<T, D>(tridiag.readwrite_sender_tile(std::declval<LocalTileIndex>()),
                               tridiag.readwrite_sender_tile(std::declval<LocalTileIndex>()))));
  using vector_type = std::vector<sender_type>;

  if (tridiag.nrTiles().rows() == 0)
    return vector_type{};

  const SizeType i_end = tridiag.nrTiles().rows() - 1;
  vector_type offdiag_vals;
  offdiag_vals.reserve(to_sizet(i_end));

  for (SizeType i_split = 0; i_split < i_end; ++i_split) {
    offdiag_vals.push_back(ex::split(
        cuppensDecompAsync<T, D>(tridiag.readwrite_sender_tile(LocalTileIndex(i_split, 0)),
                                 tridiag.readwrite_sender_tile(LocalTileIndex(i_split + 1, 0)))));
  }
  return offdiag_vals;
}

// Solve leaf eigensystem with stedc
template <class T, Device D>
void solveLeaf(Matrix<T, D>& tridiag, Matrix<T, D>& evecs) {
  SizeType ntiles = tridiag.distribution().nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    stedcAsync<D>(tridiag.readwrite_sender_tile(LocalTileIndex(i, 0)),
                  evecs.readwrite_sender_tile(LocalTileIndex(i, i)));
  }
}

// Copy the first column of @p tridiag (n x 2) into the column matrix @p evals (n x 1).
template <class T, Device D>
void offloadDiagonal(Matrix<const T, D>& tridiag, Matrix<T, D>& evals) {
  for (SizeType i = 0; i < evals.distribution().nrTiles().rows(); ++i) {
    copyDiagonalFromCompactTridiagonalAsync<D>(tridiag.read_sender2(GlobalTileIndex(i, 0)),
                                               evals.readwrite_sender_tile(GlobalTileIndex(i, 0)));
  }
}

// Notation:
//
// nb - the block/tile size of all matrices and vectors
// n1 - the size of the top subproblem
// n2 - the size of the bottom subproblem
// Q1 - (n1 x n1) the orthogonal matrix of the top subproblem
// Q2 - (n2 x n2) the orthogonal matrix of the bottom subproblem
// n := n1 + n2, the size of the merged problem
//
//      ┌───┬───┐
//      │Q1 │   │
// Q := ├───┼───┤ , (n x n) orthogonal matrix composed of the top and bottom subproblems
//      │   │Q2 │
//      └───┴───┘
// D                 := diag(Q), (n x 1) the diagonal of Q
// z                 := (n x 1) rank 1 update vector
// rho               := rank 1 update scaling factor
// D + rho*z*z^T     := rank 1 update problem
// U                 := (n x n) matrix of eigenvectors of the rank 1 update problem:
//
// k                 := the size of the deflated rank 1 update problem (k <= n)
// D'                := (k x 1), deflated D
// z'                := (k x 1), deflated z
// D' + rho*z'*z'^T  := deflated rank 1 update problem
// U'                := (k x k) matrix of eigenvectors of the deflated rank 1 update problem
//
// l1  := number of columns of the top subproblem after deflation, l1 <= n1
// l2  := number of columns of the bottom subproblem after deflation, l2 <= n2
// Q1' := (n1 x l1) the non-deflated part of Q1 (l1 <= n1)
// Q2' := (n2 x l2) the non-deflated part of Q2 (l2 <= n2)
// Qd  := (n-k x n) the deflated parts of Q1 and Q2
// U1' := (l1 x k) is the first l1 rows of U'
// U2' := (l2 x k) is the last l2 rows of U'
// I   := (n-k x n-k) identity matrix
// P   := (n x n) permutation matrix used to bring Q and U into multiplication form
//
// Q-U multiplication form to arrive at the eigenvectors of the merged problem:
//
// ┌────────┬──────┬────┐       ┌───────────────┬────┐       ┌───────────────┬────┐
// │        │      │    │       │               │    │       │               │    │
// │  Q1'   │      │    │       │               │    │       │    Q1'xU1'    │    │
// │        │      │    │       │   U1'         │    │       │               │    │
// │        │      │    │       │         ──────┤    │       │               │    │
// ├──────┬─┴──────┤ Qd │       ├───────        │    │       ├───────────────┤ Qd │
// │      │        │    │   X   │          U2'  │    │   =   │               │    │
// │      │        │    │       │               │    │       │    Q2'xU2'    │    │
// │      │  Q2'   │    │       ├───────────────┼────┤       │               │    │
// │      │        │    │       │               │ I  │       │               │    │
// │      │        │    │       │               │    │       │               │    │
// └──────┴────────┴────┘       └───────────────┴────┘       └───────────────┴────┘
//
// Note:
// 1. U1' and U2' may overlap (in practice they almost always do)
// 2. The overlap between U1' and U2' matches the number of shared columns between Q1' and Q2'
// 3. The overlap region is due to deflation via Givens rotations of a column vector from Q1 with a
//    column vector of Q2.
//
template <Backend B, Device D, class T>
void TridiagSolver<B, D, T>::call(Matrix<T, D>& tridiag, Matrix<T, D>& evals, Matrix<T, D>& evecs) {
  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& distr = evecs.distribution();
  LocalElementSize vec_size(distr.size().rows(), 1);
  TileElementSize vec_tile_size(distr.blockSize().rows(), 1);
  WorkSpace<T, D> ws{Matrix<T, D>(distr),                           // mat1
                     Matrix<T, D>(distr),                           // mat2
                     Matrix<T, D>(vec_size, vec_tile_size),         // dtmp
                     Matrix<T, D>(vec_size, vec_tile_size),         // z
                     Matrix<T, D>(vec_size, vec_tile_size),         // ztmp
                     Matrix<SizeType, D>(vec_size, vec_tile_size),  // i1
                     Matrix<SizeType, D>(vec_size, vec_tile_size),  // i2
                     Matrix<SizeType, D>(vec_size, vec_tile_size),  // i3
                     Matrix<ColType, D>(vec_size, vec_tile_size)};  // c

  // Mirror workspace on host memory for CPU-only kernels
  WorkSpaceHostMirror<T, D> ws_h{initMirrorMatrix(evals), initMirrorMatrix(ws.mat1),
                                 initMirrorMatrix(ws.dtmp), initMirrorMatrix(ws.z),
                                 initMirrorMatrix(ws.ztmp), initMirrorMatrix(ws.i2),
                                 initMirrorMatrix(ws.c),
                                 // TODO: Not needed: for local version (appease warning)
                                 initMirrorMatrix(evecs), initMirrorMatrix(ws.mat2)

  };

  // Set `evecs` to `zero` (needed for Given's rotation to make sure no random values are picked up)
  matrix::util::set0<B, T, D>(pika::execution::thread_priority::normal, evecs);

  // Cuppen's decomposition
  auto offdiag_vals = cuppensDecomposition(tridiag);

  // Solve with stedc for each tile of `tridiag` (nb x 2) and save eigenvectors in diagonal tiles of
  // `evecs` (nb x nb)
  solveLeaf(tridiag, evecs);

  // Offload the diagonal from `tridiag` to `evals`
  offloadDiagonal(tridiag, evals);

  // Each triad represents two subproblems to be merged
  for (auto [i_begin, i_split, i_end] : generateSubproblemIndices(distr.nrTiles().rows())) {
    mergeSubproblems<B>(i_begin, i_split, i_end, offdiag_vals[to_sizet(i_split)], ws, ws_h, evals,
                        evecs);
  }
}

// Overload which provides the eigenvector matrix as complex values where the imaginery part is set to zero.
template <Backend B, Device D, class T>
void TridiagSolver<B, D, T>::call(Matrix<T, D>& tridiag, Matrix<T, D>& evals,
                                  Matrix<std::complex<T>, D>& evecs) {
  Matrix<T, D> real_evecs(evecs.distribution());
  TridiagSolver<B, D, T>::call(tridiag, evals, real_evecs);

  // Convert real to complex numbers
  const matrix::Distribution& dist = evecs.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    castToComplexAsync<D>(real_evecs.read_sender2(tile_wrt_local),
                          evecs.readwrite_sender_tile(tile_wrt_local));
  }
}

// Solve for each tile of the local matrix @p tridiag (n x 2) with `stedc()` and save the result in the
// corresponding diagonal tile of the distribtued matrix @p evecs (n x n)
//
// @p tridiag is a local matrix of size (n x 2)
// @p evecs is a distributed matrix of size (n x n)
//
template <class T, Device D>
void solveDistLeaf(comm::CommunicatorGrid grid, common::Pipeline<comm::Communicator>& full_task_chain,
                   Matrix<T, D>& tridiag, Matrix<T, D>& evecs) {
  const matrix::Distribution& dist = evecs.distribution();
  namespace ex = pika::execution::experimental;

  comm::Index2D this_rank = dist.rankIndex();
  SizeType ntiles = dist.nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    GlobalTileIndex ii_tile(i, i);
    comm::Index2D ii_rank = dist.rankGlobalTile(ii_tile);
    GlobalTileIndex idx_trd(i, 0);
    if (ii_rank == this_rank) {
      stedcAsync<D>(tridiag.readwrite_sender_tile(idx_trd), evecs.readwrite_sender_tile(ii_tile));
      ex::start_detached(
          comm::scheduleSendBcast(ex::make_unique_any_sender(full_task_chain()),
                                  // TODO: any_sender to unique_any_sender conversion. Should not be necessary.
                                  ex::make_unique_any_sender(tridiag.read_sender2(idx_trd))));
    }
    else {
      comm::IndexT_MPI root_rank = grid.rankFullCommunicator(ii_rank);
      ex::start_detached(
          comm::scheduleRecvBcast(ex::make_unique_any_sender(full_task_chain()), root_rank,
                                  tridiag.readwrite_sender_tile(idx_trd)));
    }
  }
}

// Distributed tridiagonal eigensolver
//
template <Backend B, Device D, class T>
void TridiagSolver<B, D, T>::call(comm::CommunicatorGrid grid, Matrix<T, D>& tridiag,
                                  Matrix<T, D>& evals, Matrix<T, D>& evecs) {
  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& dist_evecs = evecs.distribution();
  const matrix::Distribution& dist_evals = evals.distribution();

  WorkSpace<T, D> ws{Matrix<T, D>(dist_evecs),         // mat1
                     Matrix<T, D>(dist_evecs),         // mat2
                     Matrix<T, D>(dist_evals),         // dtmp
                     Matrix<T, D>(dist_evals),         // z
                     Matrix<T, D>(dist_evals),         // ztmp
                     Matrix<SizeType, D>(dist_evals),  // i1
                     Matrix<SizeType, D>(dist_evals),  // i2
                     Matrix<SizeType, D>(dist_evals),  // i3
                     Matrix<ColType, D>(dist_evals)};  // c

  // Mirror workspace on host memory for CPU-only kernels
  WorkSpaceHostMirror<T, D> ws_h{initMirrorMatrix(evals),   initMirrorMatrix(ws.mat1),
                                 initMirrorMatrix(ws.dtmp), initMirrorMatrix(ws.z),
                                 initMirrorMatrix(ws.ztmp), initMirrorMatrix(ws.i2),
                                 initMirrorMatrix(ws.c),    initMirrorMatrix(evecs),
                                 initMirrorMatrix(ws.mat2)};

  // Set `evecs` to `zero` (needed for Given's rotation to make sure no random values are picked up)
  matrix::util::set0<B, T, D>(pika::execution::thread_priority::normal, evecs);

  // Cuppen's decomposition
  auto offdiag_vals = cuppensDecomposition(tridiag);

  common::Pipeline<comm::Communicator> full_task_chain(grid.fullCommunicator().clone());
  common::Pipeline<comm::Communicator> row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> col_task_chain(grid.colCommunicator().clone());

  // Solve with stedc for each tile of `tridiag` (nb x 2) and save eigenvectors in diagonal tiles of
  // `evecs` (nb x nb)
  solveDistLeaf(grid, full_task_chain, tridiag, evecs);

  // Offload the diagonal from `tridiag` to `evals`
  offloadDiagonal(tridiag, evals);

  // Each triad represents two subproblems to be merged
  SizeType nrtiles = dist_evecs.nrTiles().rows();
  for (auto [i_begin, i_split, i_end] : generateSubproblemIndices(nrtiles)) {
    mergeDistSubproblems<B>(grid, full_task_chain, row_task_chain, col_task_chain, i_begin, i_split,
                            i_end, offdiag_vals[to_sizet(i_split)], ws, ws_h, evals, evecs);
  }
}

// \overload TridiagSolver<B, D, T>::call()
//
// This overload of the distributed tridiagonal version of the algorithm provides the eigenvector matrix
// as complex values where the imaginery part is set to zero.
//
template <Backend B, Device D, class T>
void TridiagSolver<B, D, T>::call(comm::CommunicatorGrid grid, Matrix<T, D>& tridiag,
                                  Matrix<T, D>& evals, Matrix<std::complex<T>, D>& evecs) {
  Matrix<T, D> real_evecs(evecs.distribution());
  TridiagSolver<B, D, T>::call(grid, tridiag, evals, real_evecs);

  // Convert real to complex numbers
  const matrix::Distribution& dist = evecs.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    castToComplexAsync<D>(real_evecs.read_sender2(tile_wrt_local),
                          evecs.readwrite_sender_tile(tile_wrt_local));
  }
}

}
