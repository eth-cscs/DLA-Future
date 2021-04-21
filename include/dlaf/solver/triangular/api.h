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

#include "dlaf/types.h"

namespace dlaf {
namespace solver {
namespace internal {
template <Backend backend, Device device, class T>
struct Triangular {
  static void call_LLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_LLT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_LUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_LUT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_RLN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_RLT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_RUN(blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_RUT(blas::Op op, blas::Diag diag, T alpha, Matrix<const T, device>& mat_a,
                       Matrix<T, device>& mat_b);
  static void call_LLN(comm::CommunicatorGrid grid, blas::Diag diag, T alpha,
                       Matrix<const T, device>& mat_a, Matrix<T, device>& mat_b);
};

/// ---- ETI
#define DLAF_SOLVER_TRIANGULAR_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct Triangular<BACKEND, DEVICE, DATATYPE>;

DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_CUDA
DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_SOLVER_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
}
}
