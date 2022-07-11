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

#include "dlaf/types.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <Backend backend, Device device, class T>
struct GenToStd {
  static void call_L(Matrix<T, device>& mat_a, Matrix<T, device>& mat_l);
  static void call_L(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a, Matrix<T, device>& mat_l);
  static void call_U(Matrix<T, device>& mat_a, Matrix<T, device>& mat_u);
  static void call_U(comm::CommunicatorGrid grid, Matrix<T, device>& mat_a, Matrix<T, device>& mat_u);
};

/// ---- ETI
#define DLAF_GENTOSTD_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct GenToStd<BACKEND, DEVICE, DATATYPE>;

DLAF_GENTOSTD_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_GENTOSTD_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_GENTOSTD_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_GENTOSTD_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_GENTOSTD_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_GENTOSTD_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_GENTOSTD_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_GENTOSTD_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
}
}
