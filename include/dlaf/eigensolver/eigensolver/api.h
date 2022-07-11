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

#include "dlaf/blas/tile.h"
#include "dlaf/common/vector.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver {

template <class T, Device D>
struct EigensolverResult {
  common::internal::vector<BaseType<T>> eigenvalues;
  Matrix<T, D> eigenvectors;
};

namespace internal {

template <Backend B, Device D, class T>
struct Eigensolver {
  static EigensolverResult<T, D> call(blas::Uplo uplo, Matrix<T, D>& mat_a);
};

/// ---- ETI
#define DLAF_EIGENSOLVER_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct Eigensolver<BACKEND, DEVICE, DATATYPE>;

DLAF_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
}
