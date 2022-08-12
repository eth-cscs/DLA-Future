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
#include "dlaf/eigensolver/tridiag_solver/misc_gpu_kernels.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

#include "dlaf/matrix/print_csv.h"

namespace dlaf::eigensolver::internal {

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
  using pika::execution::thread_priority;
  using pika::execution::experimental::start_detached;

  SizeType ntiles = mat_trd.distribution().nrTiles().rows();
  for (SizeType i = 0; i < ntiles; ++i) {
    whenAllLift(mat_trd.readwrite_sender(LocalTileIndex(i, 0)),
                mat_ev.readwrite_sender(LocalTileIndex(i, i))) |
        tile::stedc(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

template <class T, Device D>
void offloadDiagonal(Matrix<const T, D>& mat_trd, Matrix<T, D>& evals) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  for (SizeType i = 0; i < evals.distribution().nrTiles().rows(); ++i) {
    auto diag_fn = [](const auto& tridiag_tile, const auto& diag_tile, [[maybe_unused]] auto&&... ts) {
      if constexpr (D == Device::CPU) {
        for (SizeType i = 0; i < tridiag_tile.size().rows(); ++i) {
          diag_tile(TileElementIndex(i, 0)) = tridiag_tile(TileElementIndex(i, 0));
        }
      }
      else {
        copyDiagTileFromTridiagTile(tridiag_tile.size().rows(), tridiag_tile.ptr(), diag_tile.ptr(),
                                    ts...);
      }
    };
    auto sender = ex::when_all(mat_trd.read_sender(GlobalTileIndex(i, 0)),
                               evals.readwrite_sender(GlobalTileIndex(i, 0)));
    ex::start_detached(
        di::transform<di::TransformDispatchType::Plain>(di::Policy<DefaultBackend<D>::value>(),
                                                        std::move(diag_fn), std::move(sender)));
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
template <Backend backend, Device device, class T>
void TridiagSolver<backend, device, T>::call(Matrix<T, device>& tridiag, Matrix<T, device>& evals,
                                             Matrix<T, device>& evecs) {
  // Auxiliary matrix used for the D&C algorithm
  const matrix::Distribution& distr = evecs.distribution();
  LocalElementSize vec_size(distr.size().rows(), 1);
  TileElementSize vec_tile_size(distr.blockSize().rows(), 1);
  WorkSpace<T, device> ws{Matrix<T, device>(distr),                           // mat1
                          Matrix<T, device>(distr),                           // mat2
                          Matrix<T, device>(vec_size, vec_tile_size),         // dtmp
                          Matrix<T, device>(vec_size, vec_tile_size),         // z
                          Matrix<T, device>(vec_size, vec_tile_size),         // ztmp
                          Matrix<SizeType, device>(vec_size, vec_tile_size),  // i1
                          Matrix<SizeType, device>(vec_size, vec_tile_size),  // i2
                          Matrix<SizeType, device>(vec_size, vec_tile_size),  // i3
                          Matrix<ColType, device>(vec_size, vec_tile_size)};  // c

  matrix::MatrixMirror<T, Device::CPU, device> tridiag_h(tridiag);

  // Set `evecs` to `zero` (needed for Given's rotation to make sure no random values are picked up)
  matrix::util::set0<backend, T, device>(pika::execution::thread_priority::normal, evecs);

  // Mirror workspace on host memory for CPU-only kernels
  WorkSpaceHostMirror<T, device> ws_h{
      matrix::MatrixMirror<T, Device::CPU, device>(evals),         // evals
      matrix::MatrixMirror<T, Device::CPU, device>(evecs),         // evecs
      matrix::MatrixMirror<T, Device::CPU, device>(ws.mat1),       // mat1
      matrix::MatrixMirror<T, Device::CPU, device>(ws.mat2),       // mat2
      matrix::MatrixMirror<T, Device::CPU, device>(ws.dtmp),       // dtmp
      matrix::MatrixMirror<T, Device::CPU, device>(ws.z),          // z
      matrix::MatrixMirror<T, Device::CPU, device>(ws.ztmp),       // ztmp
      matrix::MatrixMirror<SizeType, Device::CPU, device>(ws.i1),  // i1
      matrix::MatrixMirror<SizeType, Device::CPU, device>(ws.i2),  // i2
      matrix::MatrixMirror<SizeType, Device::CPU, device>(ws.i3),  // i3
      matrix::MatrixMirror<ColType, Device::CPU, device>(ws.c)     // c
  };

  // Cuppen's decomposition
  std::vector<pika::shared_future<T>> offdiag_vals = cuppensDecomposition(tridiag_h.get());

  // Solve with stedc for each tile of `mat_trd` (nb x 2) and save eigenvectors in diagonal tiles of
  // `evecs` (nb x nb)
  solveLeaf(tridiag_h.get(), ws_h.evecs.get());

  // Offload the diagonal from `mat_trd` to `evals`
  tridiag_h.copyTargetToSource();
  offloadDiagonal(tridiag, evals);

  ws_h.evals.copySourceToTarget();
  ws_h.evecs.copyTargetToSource();

  // Each triad represents two subproblems to be merged
  for (auto [i_begin, i_split, i_end] : generateSubproblemIndices(distr.nrTiles().rows())) {
    mergeSubproblems<backend>(i_begin, i_split, i_end, offdiag_vals[to_sizet(i_split)], ws, ws_h, evals,
                              evecs);
  }
  ws_h.evecs.copySourceToTarget();
}

// Overload which provides the eigenvector matrix as complex values where the imaginery part is set to zero.
template <Backend backend, Device device, class T>
void TridiagSolver<backend, device, T>::call(Matrix<T, device>& tridiag, Matrix<T, device>& evals,
                                             Matrix<std::complex<T>, device>& evecs) {
  Matrix<T, device> real_evecs(evecs.distribution());
  TridiagSolver<backend, device, T>::call(tridiag, evals, real_evecs);

  // Convert real to complex numbers
  const matrix::Distribution& dist = evecs.distribution();
  for (auto tile_wrt_local : iterate_range2d(dist.localNrTiles())) {
    auto sender = pika::execution::experimental::when_all(real_evecs.read_sender(tile_wrt_local),
                                                          evecs.readwrite_sender(tile_wrt_local));
    if constexpr (device == Device::CPU) {
      dlaf::internal::transformDetach(
          dlaf::internal::Policy<Backend::MC>(),
          [](const matrix::Tile<const T, Device::CPU>& in,
             const matrix::Tile<std::complex<T>, Device::CPU>& out) {
            for (auto el_idx : iterate_range2d(out.size())) {
              out(el_idx) = std::complex<T>(in(el_idx), 0);
            }
          },
          std::move(sender));
    }
    else {
#ifdef DLAF_WITH_GPU
      namespace ex = pika::execution::experimental;
      namespace di = dlaf::internal;
      ex::start_detached(di::transform<di::TransformDispatchType::Plain>(
          dlaf::internal::Policy<Backend::GPU>(),
          [](const matrix::Tile<const T, Device::GPU>& in,
             const matrix::Tile<std::complex<T>, Device::GPU>& out, cudaStream_t stream) {
            castTileToComplex(in.size().rows(), in.size().cols(), in.ld(), in.ptr(), out.ptr(), stream);
          },
          std::move(sender)));
#endif
    }
  }
}
}
