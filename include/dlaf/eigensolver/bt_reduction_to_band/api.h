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

#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

template <Backend backend, Device device, class T>
struct BackTransformationReductionToBand {
  static void call(SizeType b, Matrix<T, device>& mat_c, Matrix<const T, device>& mat_v,
                   Matrix<const T, Device::CPU>& mat_taus);

  static void call(comm::CommunicatorGrid grid, const SizeType b, Matrix<T, device>& mat_c,
                   Matrix<const T, device>& mat_v, Matrix<const T, Device::CPU>& mat_taus);
};

// ETI
#define DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct BackTransformationReductionToBand<BACKEND, DEVICE, DATATYPE>;

DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_BT_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
