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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
struct BackTransformationT2B {
  static void call(const SizeType band_size, Matrix<T, D>& mat_e, Matrix<const T, Device::CPU>& mat_hh);
};

template <Backend B, Device D, class T>
struct BackTransformationT2B_D {
  static void call(comm::CommunicatorGrid grid, const SizeType band_size, Matrix<T, D>& mat_e, Matrix<const T, Device::CPU>& mat_hh);
};

/// ---- ETI
#define DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_LOCAL_ETI(KWORD, BACKEND, DEVICE, T) \
  KWORD template struct BackTransformationT2B<BACKEND, DEVICE, T>;

#define DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_DISTR_ETI(KWORD, BACKEND, DEVICE, T) \
  KWORD template struct BackTransformationT2B_D<BACKEND, DEVICE, T>;

#define DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(KWORD, BACKEND, DEVICE, T) \
  DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_LOCAL_ETI(KWORD, BACKEND, DEVICE, T) \
  DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_DISTR_ETI(KWORD, BACKEND, DEVICE, T)

DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_LOCAL_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_LOCAL_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_LOCAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_BT_BAND_TO_TRIDIAGONAL_LOCAL_ETI(extern, Backend::GPU, Device::GPU,
                                                  std::complex<double>)
#endif
}
