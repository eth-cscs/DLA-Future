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

#include "dlaf/eigensolver/tridiag_solver/api.h"
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
void cuppensTridiagTileUpdate(const matrix::Tile<T, Device::CPU>& top,
                              const matrix::Tile<T, Device::CPU>& bottom) {
  (void) top;
  (void) bottom;

  T offdiag_val = top(TileElementIndex{top.size().rows() - 1, top.size().cols() - 1});
  T& top_diag_val = top(TileElementIndex{top.size().rows() - 1, top.size().cols() - 2});
  T& bottom_diag_val = bottom(TileElementIndex{0, 0});

  top_diag_val -= offdiag_val;
  bottom_diag_val -= offdiag_val;
}

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(Matrix<BaseType<T>, Device::CPU>& mat_a,
                                                      SizeType i_begin, SizeType i_end,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  if (i_begin == i_end) {
    // TODO: solve with stedc
    return;
  }
  SizeType i_midpoint = (i_begin + i_end) / 2;
  // TODO: call Cuppen's tridiag decomposition
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
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_CUDA
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, float)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, double)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
// DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif

}
}
}
