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
#include <atomic>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <pika/execution.hpp>
#include <pika/thread.hpp>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include <dlaf/common/callable_object.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/eigensolver/tridiag_solver/api.h>
#include <dlaf/eigensolver/tridiag_solver/kernels_async.h>
#include <dlaf/eigensolver/tridiag_solver/merge.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/hdf5.h>
#include <dlaf/permutations/general.h>
#include <dlaf/permutations/general/impl.h>
#include <dlaf/sender/make_sender_algorithm_overloads.h>
#include <dlaf/sender/policy.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal {

// Splits [i_begin, i_end) in the middle and waits for all splits on [i_begin, i_split] and [i_split,
// i_end) before saving the triad <i_begin, i_split, i_end> into `indices`.
//
// The recursive calls span a binary tree which is traversed in depth first left-right-root order. That
// is also the order of triads in `indices`.
//
inline void splitIntervalInTheMiddleRecursively(
    const SizeType i_begin, const SizeType i_end,
    std::vector<std::tuple<SizeType, SizeType, SizeType>>& indices) {
  if (i_begin + 1 == i_end)
    return;
  const SizeType i_split = util::ceilDiv<SizeType>(i_begin + i_end, 2);
  splitIntervalInTheMiddleRecursively(i_begin, i_split, indices);
  splitIntervalInTheMiddleRecursively(i_split, i_end, indices);
  indices.emplace_back(i_begin, i_split, i_end);
}

// Generates an array of triad indices. Each triad is composed of begin <= middle < end indices and
// represents two intervals [begin, middle] and [middle + 1, end]. The two intervals are the subproblems
// that have to be merged to arrive at [begin, end].
//
// Note: the intervals are all closed!
//
inline std::vector<std::tuple<SizeType, SizeType, SizeType>> generateSubproblemIndices(const SizeType n) {
  if (n == 0)
    return {};

  std::vector<std::tuple<SizeType, SizeType, SizeType>> indices;
  indices.reserve(to_sizet(n));
  splitIntervalInTheMiddleRecursively(0, n, indices);
  return indices;
}

template <class T>
auto cuppensDecomposition(Matrix<T, Device::CPU>& tridiag) {
  namespace ex = pika::execution::experimental;
  using sender_type = decltype(ex::split(
      ex::ensure_started(cuppensDecompAsync<T>(tridiag.readwrite(std::declval<LocalTileIndex>()),
                                               tridiag.readwrite(std::declval<LocalTileIndex>())))));
  using vector_type = std::vector<sender_type>;

  if (tridiag.nrTiles().rows() == 0)
    return vector_type{};

  const SizeType i_end = tridiag.nrTiles().rows();
  vector_type offdiag_vals;
  offdiag_vals.reserve(to_sizet(i_end - 1));

  for (SizeType i_split = 1; i_split < i_end; ++i_split) {
    offdiag_vals.push_back(ex::split(
        ex::ensure_started(cuppensDecompAsync<T>(tridiag.readwrite(LocalTileIndex(i_split - 1, 0)),
                                                 tridiag.readwrite(LocalTileIndex(i_split, 0))))));
  }
  return offdiag_vals;
}

// Solve leaf eigensystem with stedc
template <class T>
void solveLeaf(Matrix<T, Device::CPU>& tridiag, Matrix<T, Device::CPU>& evecs) {
  const SizeType ntiles = tridiag.distribution().nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    stedcAsync(tridiag.readwrite(LocalTileIndex(i, 0)), evecs.readwrite(LocalTileIndex(i, i)));
  }
}

#ifdef DLAF_WITH_GPU
template <class T>
void solveLeaf(Matrix<T, Device::CPU>& tridiag, Matrix<T, Device::GPU>& evecs,
               Matrix<T, Device::CPU>& h_evecs) {
  namespace ex = pika::execution::experimental;
  using matrix::copy;
  using pika::execution::thread_stacksize;
  const auto cp_policy =
      dlaf::internal::Policy<matrix::internal::CopyBackend_v<Device::GPU, Device::CPU>>{
          thread_stacksize::nostack};

  const SizeType ntiles = tridiag.distribution().nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    const auto id_tr = LocalTileIndex(i, 0);
    const auto id_ev = LocalTileIndex(i, i);

    stedcAsync(tridiag.readwrite(id_tr), h_evecs.readwrite(id_ev));
    ex::start_detached(ex::when_all(h_evecs.read(id_ev), evecs.readwrite(id_ev)) | copy(cp_policy));
  }
}
#endif

// Copy the first column of @p tridiag (n x 2) into the column matrix @p evals (n x 1).
template <class T, Device D>
void offloadDiagonal(Matrix<const T, Device::CPU>& tridiag, Matrix<T, D>& evals) {
  for (SizeType i = 0; i < evals.distribution().nrTiles().rows(); ++i) {
    copyDiagonalFromCompactTridiagonalAsync<D>(tridiag.read(GlobalTileIndex(i, 0)),
                                               evals.readwrite(GlobalTileIndex(i, 0)));
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
void TridiagSolver<B, D, T>::call(Matrix<T, Device::CPU>& tridiag, Matrix<T, D>& evals,
                                  Matrix<T, D>& evecs) {
  using pika::execution::thread_priority;

  // Quick return for empty matrix
  if (evecs.size().isEmpty())
    return;

  // If the matrix is composed by a single tile simply call stedc.
  if (evecs.nrTiles().linear_size() == 1) {
    if constexpr (D == Device::CPU) {
      solveLeaf(tridiag, evecs);
    }
    else {
      Matrix<T, Device::CPU> h_evecs{evecs.distribution()};
      solveLeaf(tridiag, evecs, h_evecs);
    }
    offloadDiagonal(tridiag, evals);
    return;
  }

  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& distr = evecs.distribution();
  const LocalElementSize vec_size(distr.size().rows(), 1);
  const TileElementSize vec_tile_size(distr.blockSize().rows(), 1);
  WorkSpace<T, D> ws{Matrix<T, D>(distr),                            // e0
                     Matrix<T, D>(distr),                            // e1
                     evecs,                                          // e2
                     evals,                                          // d1
                     Matrix<T, D>(vec_size, vec_tile_size),          // z0
                     Matrix<T, D>(vec_size, vec_tile_size),          // z1
                     Matrix<SizeType, D>(vec_size, vec_tile_size),   // i2
                     Matrix<SizeType, D>(vec_size, vec_tile_size),   // i5
                     Matrix<SizeType, D>(vec_size, vec_tile_size),   // i5b
                     Matrix<SizeType, D>(vec_size, vec_tile_size)};  // i6

  WorkSpaceHost<T> ws_h{Matrix<T, Device::CPU>(vec_size, vec_tile_size),          // d0
                        Matrix<ColType, Device::CPU>(vec_size, vec_tile_size),    // c
                        Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),   // i1
                        Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size),   // i3
                        Matrix<SizeType, Device::CPU>(vec_size, vec_tile_size)};  // i4

  // Mirror workspace on host memory for CPU-only kernels
  WorkSpaceHostMirror<T, D> ws_hm{initMirrorMatrix(ws.e2), initMirrorMatrix(ws.d1),
                                  initMirrorMatrix(ws.z0), initMirrorMatrix(ws.z1),
                                  initMirrorMatrix(ws.i2), initMirrorMatrix(ws.i5)};

  // Set `ws.e0` to `zero` (needed for Given's rotation to make sure no random values are picked up)
  matrix::util::set0<B, T, D>(thread_priority::normal, ws.e0);

  // Cuppen's decomposition
  auto offdiag_vals = cuppensDecomposition(tridiag);

  // Solve with stedc for each tile of `tridiag` (nb x 2) and save eigenvectors in diagonal tiles of
  // `evecs` (nb x nb)
  if constexpr (D == Device::CPU) {
    solveLeaf(tridiag, ws.e0);
  }
  else {
    solveLeaf(tridiag, ws.e0, ws_hm.e2);
  }

  // Offload the diagonal from `tridiag` to `d0`
  offloadDiagonal(tridiag, ws_h.d0);

  // Each triad represents two subproblems to be merged
  for (auto [i_begin, i_split, i_end] : generateSubproblemIndices(distr.nrTiles().rows())) {
    mergeSubproblems<B>(i_begin, i_split, i_end, offdiag_vals[to_sizet(i_split - 1)], ws, ws_h, ws_hm);
  }

  const SizeType n = evecs.nrTiles().rows();
  copy(ws_hm.i2, ws.i2);

  // Note: ws_hm.d1 is the mirror of ws.d1 which is evals
  applyIndex(0, n, ws_hm.i2, ws_h.d0, ws_hm.d1);
  copy(ws_hm.d1, evals);

  dlaf::permutations::permute<B, D, T, Coord::Col>(0, n, ws.i2, ws.e0, evecs);
}

// Overload which provides the eigenvector matrix as complex values where the imaginery part is set to zero.
template <Backend B, Device D, class T>
void TridiagSolver<B, D, T>::call(Matrix<T, Device::CPU>& tridiag, Matrix<T, D>& evals,
                                  Matrix<std::complex<T>, D>& evecs) {
  Matrix<T, D> real_evecs(evecs.distribution());
  TridiagSolver<B, D, T>::call(tridiag, evals, real_evecs);

  // Convert real to complex numbers
  const matrix::Distribution& dist = evecs.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    castToComplexAsync<D>(real_evecs.read(tile_wrt_local), evecs.readwrite(tile_wrt_local));
  }
}

// Solve for each tile of the local matrix @p tridiag (n x 2) with `stedc()` and save the result in the
// corresponding diagonal tile of the distribtued matrix @p evecs (n x n)
//
// @p tridiag is a local matrix of size (n x 2)
// @p evecs is a distributed matrix of size (n x n)
//
template <class T>
void solveDistLeaf(comm::CommunicatorPipeline<comm::CommunicatorType::Full>& full_task_chain,
                   Matrix<T, Device::CPU>& tridiag, Matrix<T, Device::CPU>& evecs) {
  const matrix::Distribution& dist = evecs.distribution();
  namespace ex = pika::execution::experimental;

  const comm::Index2D this_rank = dist.rankIndex();
  const SizeType ntiles = dist.nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    const GlobalTileIndex ii_tile(i, i);
    const comm::Index2D ii_rank = dist.rankGlobalTile(ii_tile);
    const GlobalTileIndex id_tr(i, 0);
    if (ii_rank == this_rank) {
      stedcAsync(tridiag.readwrite(id_tr), evecs.readwrite(ii_tile));
      if (full_task_chain.size() > 1) {
        ex::start_detached(comm::schedule_bcast_send(full_task_chain.exclusive(), tridiag.read(id_tr)));
      }
    }
    else {
      const comm::IndexT_MPI root_rank = full_task_chain.rank_full_communicator(ii_rank);
      ex::start_detached(comm::schedule_bcast_recv(full_task_chain.exclusive(), root_rank,
                                                   tridiag.readwrite(id_tr)));
    }
  }
}

#ifdef DLAF_WITH_GPU
template <class T>
void solveDistLeaf(comm::CommunicatorPipeline<comm::CommunicatorType::Full>& full_task_chain,
                   Matrix<T, Device::CPU>& tridiag, Matrix<T, Device::GPU>& evecs,
                   Matrix<T, Device::CPU>& h_evecs) {
  const matrix::Distribution& dist = evecs.distribution();
  namespace ex = pika::execution::experimental;
  using matrix::copy;
  using pika::execution::thread_stacksize;
  const auto cp_policy =
      dlaf::internal::Policy<matrix::internal::CopyBackend_v<Device::GPU, Device::CPU>>{
          thread_stacksize::nostack};

  const comm::Index2D this_rank = dist.rankIndex();
  const SizeType ntiles = dist.nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    const GlobalTileIndex ii_tile(i, i);
    const comm::Index2D ii_rank = dist.rankGlobalTile(ii_tile);
    const GlobalTileIndex id_tr(i, 0);
    if (ii_rank == this_rank) {
      stedcAsync(tridiag.readwrite(id_tr), h_evecs.readwrite(ii_tile));
      ex::start_detached(ex::when_all(h_evecs.read(ii_tile), evecs.readwrite(ii_tile)) |
                         copy(cp_policy));
      if (full_task_chain.size() > 1) {
        ex::start_detached(comm::schedule_bcast_send(full_task_chain.exclusive(), tridiag.read(id_tr)));
      }
    }
    else {
      const comm::IndexT_MPI root_rank = full_task_chain.rank_full_communicator(ii_rank);
      ex::start_detached(comm::schedule_bcast_recv(full_task_chain.exclusive(), root_rank,
                                                   tridiag.readwrite(id_tr)));
    }
  }
}
#endif

// Distributed tridiagonal eigensolver
//
template <Backend B, Device D, class T>
void TridiagSolver<B, D, T>::call(comm::CommunicatorGrid& grid, Matrix<T, Device::CPU>& tridiag,
                                  Matrix<T, D>& evals, Matrix<T, D>& evecs) {
  using pika::execution::thread_priority;

  auto full_task_chain = grid.full_communicator_pipeline();

  // Quick return for empty matrix
  if (evecs.size().isEmpty())
    return;

#ifdef DLAF_WITH_HDF5
  static std::atomic<size_t> num_tridiag_solver_calls = 0;
  std::stringstream fname;
  fname << "tridiag_solver-"
        << matrix::internal::TypeToString_v<T> << std::to_string(num_tridiag_solver_calls) << ".h5";
  std::optional<matrix::internal::FileHDF5> file;

  if (getTuneParameters().debug_dump_tridiag_solver_data) {
    file = matrix::internal::FileHDF5(grid.fullCommunicator(), fname.str());
    file->write(tridiag, "/input");
  }
#endif

  // If the matrix is composed by a single tile simply call stedc.
  if (evecs.nrTiles().linear_size() == 1) {
    if constexpr (D == Device::CPU) {
      solveDistLeaf(full_task_chain, tridiag, evecs);
    }
    else {
      Matrix<T, Device::CPU> h_evecs{evecs.distribution()};
      solveDistLeaf(full_task_chain, tridiag, evecs, h_evecs);
    }
    offloadDiagonal(tridiag, evals);
    return;
  }

  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& dist_evecs = evecs.distribution();
  const matrix::Distribution& dist_evals = evals.distribution();
  const matrix::Distribution dist_local({dist_evecs.local_size().cols(), 1}, dist_evecs.tile_size());

  WorkSpace<T, D> ws{Matrix<T, D>(dist_evecs),          // e0
                     Matrix<T, D>(dist_evecs),          // e1
                     evecs,                             // e2
                     evals,                             // d1
                     Matrix<T, D>(dist_evals),          // z0
                     Matrix<T, D>(dist_evals),          // z1
                     Matrix<SizeType, D>(dist_evals),   // i2
                     Matrix<SizeType, D>(dist_evals),   // i5
                     Matrix<SizeType, D>(dist_local),   // i5b
                     Matrix<SizeType, D>(dist_evals)};  // i6

  WorkSpaceHost<T> ws_h{Matrix<T, Device::CPU>(dist_evals),          // d0
                        Matrix<ColType, Device::CPU>(dist_evals),    // c
                        Matrix<SizeType, Device::CPU>(dist_evals),   // i1
                        Matrix<SizeType, Device::CPU>(dist_evals),   // i3
                        Matrix<SizeType, Device::CPU>(dist_evals)};  // i4

  // Mirror workspace on host memory for CPU-only kernels
  DistWorkSpaceHostMirror<T, D> ws_hm{initMirrorMatrix(ws.e0), initMirrorMatrix(ws.e2),
                                      initMirrorMatrix(ws.d1), initMirrorMatrix(ws.z0),
                                      initMirrorMatrix(ws.z1), initMirrorMatrix(ws.i2),
                                      initMirrorMatrix(ws.i5), initMirrorMatrix(ws.i5b),
                                      initMirrorMatrix(ws.i6)};

  // Set `ws.e0` to `zero` (needed for Given's rotation to make sure no random values are picked up)
  matrix::util::set0<B, T, D>(thread_priority::normal, ws.e0);

  // Cuppen's decomposition
  auto offdiag_vals = cuppensDecomposition(tridiag);

  auto row_task_chain = grid.row_communicator_pipeline();
  auto col_task_chain = grid.col_communicator_pipeline();

  // Solve with stedc for each tile of `tridiag` (nb x 2) and save eigenvectors in diagonal tiles of
  // `evecs` (nb x nb)
  if constexpr (D == Device::CPU) {
    solveDistLeaf(full_task_chain, tridiag, ws.e0);
  }
  else {
    solveDistLeaf(full_task_chain, tridiag, ws.e0, ws_hm.e0);
  }

  // Offload the diagonal from `tridiag` to `evals`
  offloadDiagonal(tridiag, ws_h.d0);

  // Each triad represents two subproblems to be merged
  SizeType nrtiles = dist_evecs.nrTiles().rows();
  for (auto [i_begin, i_split, i_end] : generateSubproblemIndices(nrtiles)) {
    mergeDistSubproblems<B>(full_task_chain, row_task_chain, col_task_chain, i_begin, i_split, i_end,
                            offdiag_vals[to_sizet(i_split - 1)], ws, ws_h, ws_hm);
  }

  const SizeType n = evecs.nrTiles().rows();
  copy(ws.e0, ws_hm.e0);

  // Note: ws_hm.d1 is the mirror of ws.d1 which is evals
  applyIndex(0, n, ws_h.i1, ws_h.d0, ws_hm.d1);
  copy(ws_hm.d1, evals);

  // Note: ws_hm.e2 is the mirror of ws.e2 which is evecs
  dlaf::permutations::permute<Backend::MC, Device::CPU, T, Coord::Col>(row_task_chain, 0, n, ws_h.i1,
                                                                       ws_hm.e0, ws_hm.e2);
  copy(ws_hm.e2, evecs);

#ifdef DLAF_WITH_HDF5
  if (getTuneParameters().debug_dump_tridiag_solver_data) {
    file->write(evecs, "/evecs");
    file->write(evals, "/evals");
  }

  num_tridiag_solver_calls++;
#endif
}

// \overload TridiagSolver<B, D, T>::call()
//
// This overload of the distributed tridiagonal version of the algorithm provides the eigenvector matrix
// as complex values where the imaginery part is set to zero.
//
template <Backend B, Device D, class T>
void TridiagSolver<B, D, T>::call(comm::CommunicatorGrid& grid, Matrix<T, Device::CPU>& tridiag,
                                  Matrix<T, D>& evals, Matrix<std::complex<T>, D>& evecs) {
  Matrix<T, D> real_evecs(evecs.distribution());
  TridiagSolver<B, D, T>::call(grid, tridiag, evals, real_evecs);

  // Convert real to complex numbers
  const matrix::Distribution& dist = evecs.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    castToComplexAsync<D>(real_evecs.read(tile_wrt_local), evecs.readwrite(tile_wrt_local));
  }
}

}
