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

#include "dlaf/eigensolver/eigensolver/api.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf {
namespace eigensolver {
namespace internal {
template <Backend backend, Device device, class T>
struct GenEigensolver {
  static EigensolverResult<T, device> call(blas::Uplo uplo, Matrix<T, device>& mat_a,
                                           Matrix<T, device>& mat_b);
};

/// ---- ETI
#define DLAF_EIGENSOLVER_GEN_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct GenEigensolver<BACKEND, DEVICE, DATATYPE>;

DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_GEN_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

}
}
}
