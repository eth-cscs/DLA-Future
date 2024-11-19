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

#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/types.h>

namespace dlaf::multiplication::internal {

using matrix::internal::MatrixRef;

template <Backend B, Device D, class T>
struct Hermitian {
  static void call_LL(const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b, const T beta,
                      Matrix<T, D>& mat_c);

  static void call_LL(comm::CommunicatorGrid& grid, const T alpha, Matrix<const T, D>& mat_a,
                      MatrixRef<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c);
};

// ETI
#define DLAF_MULTIPLICATION_HERMITIAN_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct Hermitian<BACKEND, DEVICE, DATATYPE>;

DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_MULTIPLICATION_HERMITIAN_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
