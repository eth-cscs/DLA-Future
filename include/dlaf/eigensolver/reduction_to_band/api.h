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

#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

template <Backend B, Device D, class T>
struct ReductionToBand {
  static common::internal::vector<pika::shared_future<common::internal::vector<T>>> call(
      Matrix<T, D>& mat_a, const SizeType band_size);
  static common::internal::vector<pika::shared_future<common::internal::vector<T>>> call(
      comm::CommunicatorGrid grid, Matrix<T, D>& mat_a, const SizeType band_size);
};

/// ---- ETI
#define DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct ReductionToBand<BACKEND, DEVICE, DATATYPE>;

DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_REDUCTION_TO_BAND_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
