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

#include <pika/execution.hpp>

#include <dlaf/matrix/matrix.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal {

template <class T>
struct TridiagResult1Stage {
  Matrix<T, Device::CPU> taus;
  Matrix<BaseType<T>, Device::CPU> tridiagonal;
};

template <Backend B, Device D, class T>
struct ReductionToTrid {
  static TridiagResult1Stage<T> call(Matrix<T, D>& mat_a);
};

// ETI
#define DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct ReductionToTrid<BACKEND, DEVICE, DATATYPE>;

DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_REDUCTION_TO_TRID_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
