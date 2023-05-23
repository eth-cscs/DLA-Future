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

#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal {

template <Backend backend, Device device, class T>
struct TridiagSolver {
  static void call(Matrix<T, Device::CPU>& tridiag, Matrix<T, device>& evals, Matrix<T, device>& evecs);
  static void call(Matrix<T, Device::CPU>& tridiag, Matrix<T, device>& evals,
                   Matrix<std::complex<T>, device>& evecs);
  static void call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& tridiag,
                   Matrix<T, device>& evals, Matrix<T, device>& evecs);
  static void call(comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& tridiag,
                   Matrix<T, device>& evals, Matrix<std::complex<T>, device>& evecs);
};

// ETI
#define DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct TridiagSolver<BACKEND, DEVICE, DATATYPE>;

DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, double)

#ifdef DLAF_WITH_GPU
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, double)
#endif

}
