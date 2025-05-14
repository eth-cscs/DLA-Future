//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <complex>

#include <blas.hh>

#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>

namespace dlaf::inverse::internal {
template <Backend backend, Device device, class T>
struct Triangular {
  static void call_L(blas::Diag diag, Matrix<T, device>& mat_a);
  static void call_U(blas::Diag diag, Matrix<T, device>& mat_a);
  static void call_L(comm::CommunicatorGrid& grid, blas::Diag diag, Matrix<T, device>& mat_a);
  static void call_U(comm::CommunicatorGrid& grid, blas::Diag diag, Matrix<T, device>& mat_a);
};

// ETI
#define DLAF_INVERSE_TRIANGULAR_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct Triangular<BACKEND, DEVICE, DATATYPE>;

DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_INVERSE_TRIANGULAR_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
