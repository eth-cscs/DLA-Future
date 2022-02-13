//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/common/callable_object.h"
#include "dlaf/eigensolver/tridiag_solver/api.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/sender/make_sender_algorithm_overloads.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
struct TridiagSolver<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_a, SizeType i_begin, SizeType i_end,
                   Matrix<T, Device::CPU>& d, Matrix<T, Device::CPU>& z, Matrix<T, Device::CPU>& mat_ws,
                   Matrix<T, Device::CPU>& mat_ev);
  static void call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a,
                   Matrix<T, Device::CPU>& mat_ev);
};

template <class T>
void cuppensDecomposition(const matrix::Tile<T, Device::CPU>& top,
                          const matrix::Tile<T, Device::CPU>& bottom) {
  (void) top;
  (void) bottom;

  T offdiag_val = top(TileElementIndex{top.size().rows() - 1, top.size().cols() - 1});
  T& top_diag_val = top(TileElementIndex{top.size().rows() - 1, top.size().cols() - 2});
  T& bottom_diag_val = bottom(TileElementIndex{0, 0});

  top_diag_val -= offdiag_val;
  bottom_diag_val -= offdiag_val;
}

DLAF_MAKE_CALLABLE_OBJECT(cuppensDecomposition);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(cuppensDecomposition, cuppensDecomposition_o)

template <class T>
void copyTileRow(SizeType row, const matrix::Tile<const T, Device::CPU>& tile,
                 const matrix::Tile<T, Device::CPU>& col) {
  for (SizeType i = 0; i < tile.size().rows(); ++i) {
    col(TileElementIndex(i, 0)) = tile(TileElementIndex(row, i));
  }
}

DLAF_MAKE_CALLABLE_OBJECT(copyTileRow);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(copyTileRow, copyTileRow_o)

// The bottom row of Q1 and the top row of Q2
template <class T>
void assembleZVec(SizeType i_begin, SizeType i_middle, SizeType i_end,
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
    whenAllLift(tile_row, mat_ev.read_sender(mat_ev_idx), z.readwrite_sender(z_idx)) |
        copyTileRow(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
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
void assembleDiag(SizeType i_begin, SizeType i_end, Matrix<const T, Device::CPU>& mat_a,
                  Matrix<T, Device::CPU>& d) {
  using pika::threads::thread_priority;
  using pika::execution::experimental::start_detached;
  using dlaf::internal::Policy;
  using dlaf::internal::whenAllLift;

  for (SizeType i = i_begin; i <= i_end; ++i) {
    whenAllLift(mat_a.read_sender(GlobalTileIndex(i, 0)), d.readwrite_sender(GlobalTileIndex(i, 0))) |
        copyDiagTile(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
  }
}

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_a, SizeType i_begin,
                                                      SizeType i_end, Matrix<T, Device::CPU>& d,
                                                      Matrix<T, Device::CPU>& z,
                                                      Matrix<T, Device::CPU>& mat_ws,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  using pika::threads::thread_priority;
  using dlaf::internal::Policy;
  using pika::execution::experimental::start_detached;

  if (i_begin == i_end) {
    // Solve leaf eigensystem with stedc
    dlaf::internal::whenAllLift(mat_a.readwrite_sender(LocalTileIndex(i_begin, 0)),
                                mat_ev.readwrite_sender(LocalTileIndex(i_begin, i_begin))) |
        tile::stedc(Policy<Backend::MC>(thread_priority::normal)) | start_detached();
    return;
  }
  SizeType i_midpoint = (i_begin + i_end) / 2;

  // Cuppen's tridiagonal decomposition
  dlaf::internal::whenAllLift(mat_a.readwrite_sender(LocalTileIndex(i_midpoint, 0)),
                              mat_a.readwrite_sender(LocalTileIndex(i_midpoint + 1, 0))) |
      cuppensDecomposition(Policy<Backend::MC>(thread_priority::normal)) | start_detached();

  // Left leaf
  TridiagSolver<Backend::MC, Device::CPU, T>::call(mat_a, i_begin, i_midpoint, d, z, mat_ws, mat_ev);
  // Right leaf
  TridiagSolver<Backend::MC, Device::CPU, T>::call(mat_a, i_midpoint + 1, i_end, d, z, mat_ws, mat_ev);

  // Form D + rzz^T from `mat_a` and `mat_ev`
  assembleZVec(i_begin, i_midpoint, i_end, mat_ev, z);
  assembleDiag(i_begin, i_end, mat_a, d);

  // The norm of `z` is sqrt(2) because it is a concatination of two normalized vectors

  // multiply the parameter `rho` by 2 to account for the normalization of `z`

  // Deflate D + rzz^T
  // Find evals of D + rzz^T with laed4 (root solver)
  // Form evecs
  // Gemm
}

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(comm::CommunicatorGrid grid,
                                                      Matrix<T, Device::CPU>& mat_a,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  (void) grid;
  (void) mat_a;
  (void) mat_ev;
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
}
}
