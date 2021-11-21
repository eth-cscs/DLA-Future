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

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
struct TridiagSolver<Backend::MC, Device::CPU, T> {
  static void call(Matrix<T, Device::CPU>& mat_a, Matrix<T, Device::CPU>& mat_ev);
  static void call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a,
                   Matrix<T, Device::CPU>& mat_ev);
};

template <class T>
void TridiagSolver<Backend::MC, Device::CPU, T>::call(Matrix<T, Device::CPU>& mat_a,
                                                      Matrix<T, Device::CPU>& mat_ev) {
  (void) mat_a;
  (void) mat_ev;
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
