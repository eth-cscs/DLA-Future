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
  static void call(Matrix<BaseType<T>, Device::CPU>& mat_a, SizeType i_begin, SizeType i_end,
                   Matrix<T, Device::CPU>& mat_ev);
  static void call(comm::CommunicatorGrid grid, Matrix<BaseType<T>, Device::CPU>& mat_a,
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

// The bottom row of Q1 and the top row of Q2
template <class T>
void assembleZVec(SizeType i_begin, SizeType i_middle, SizeType i_end,
                  Matrix<const T, Device::CPU>& mat_ev, Matrix<T, Device::CPU>& z) {
  using hpx::threads::thread_priority;
  using hpx::execution::experimental::detach;
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
    // Transpose the column vector `z` and copy rows from Q1 or Q2
    whenAllLift(common::transposed(z.distribution().tileSize(z_idx)), TileElementIndex(tile_row, 0),
                mat_ev.read_sender(mat_ev_idx), TileElementIndex(0, 0), z.readwrite_sender(z_idx)) |
        matrix::copy(Policy<Backend::MC>(thread_priority::normal)) | detach();
  }
}

DLAF_MAKE_CALLABLE_OBJECT(cuppensDecomposition);
DLAF_MAKE_SENDER_ALGORITHM_OVERLOADS(cuppensDecomposition, cuppensDecomposition_o)

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(Matrix<BaseType<T>, Device::CPU>& mat_a,
                                                      SizeType i_begin, SizeType i_end,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  using hpx::threads::thread_priority;
  using dlaf::internal::Policy;
  using hpx::execution::experimental::detach;

  if (i_begin == i_end) {
    // Solve leaf eigensystem with stedc
    dlaf::internal::whenAllLift(mat_a.readwrite_sender(LocalTileIndex(i_begin, 0)),
                                mat_ev.readwrite_sender(LocalTileIndex(i_begin, i_begin))) |
        tile::stedc(Policy<Backend::MC>(thread_priority::normal)) | detach();
    return;
  }
  SizeType i_midpoint = (i_begin + i_end) / 2;

  // Cuppen's tridiagonal decomposition
  dlaf::internal::whenAllLift(mat_a.readwrite_sender(LocalTileIndex(i_midpoint, 0)),
                              mat_a.readwrite_sender(LocalTileIndex(i_midpoint + 1, 0))) |
      cuppensDecomposition(Policy<Backend::MC>(thread_priority::normal)) | detach();

  TridiagSolver<Backend::MC, Device::CPU, T>::call(mat_a, i_begin, i_midpoint, mat_ev);    // left
  TridiagSolver<Backend::MC, Device::CPU, T>::call(mat_a, i_midpoint + 1, i_end, mat_ev);  // right
  // TODO: form D + rzz^T from `mat_a` and `mat_ev`
  // TODO: Deflate D + rzz^T
  // TODO: Find evals of D + rzz^T with laed4 (root solver)
  // TODO: Form evecs
  // TODO: Gemm
}

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(comm::CommunicatorGrid grid,
                                                      Matrix<BaseType<T>, Device::CPU>& mat_a,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  (void) grid;
  (void) mat_a;
  (void) mat_ev;
}

/// ---- ETI
#define DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct TridiagSolver<BACKEND, DEVICE, DATATYPE>;

DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, float)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, double)
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
