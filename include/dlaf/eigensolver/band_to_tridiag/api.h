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

#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver {

template <class T, Device D>
struct TridiagResult {
  Matrix<BaseType<T>, D> tridiagonal;
  Matrix<T, D> hh_reflectors;
};

namespace internal {

template <class T>
SizeType nrSweeps(SizeType size) noexcept {
  // Complex needs an extra sweep to have a real tridiagonal matrix.
  return isComplex_v<T> ? size - 1 : size - 2;
}

inline SizeType nrStepsForSweep(SizeType sweep, SizeType size, SizeType band) noexcept {
  // Sweep size-2 should be handled differently
  // as it is executed only for complex types (see nrSweeps)
  return sweep == size - 2 ? 1 : util::ceilDiv(size - sweep - 2, band);
}

template <Backend B, Device D, class T>
struct BandToTridiag;

template <Device D, class T>
struct BandToTridiag<Backend::MC, D, T> {
  static TridiagResult<T, Device::CPU> call_L(const SizeType b, Matrix<const T, D>& mat_a) noexcept;
};

/// ---- ETI
#define DLAF_EIGENSOLVER_B2T_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct BandToTridiag<BACKEND, DEVICE, DATATYPE>;

DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_CUDA
DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::GPU, float)
DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::GPU, double)
DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_B2T_ETI(extern, Backend::MC, Device::GPU, std::complex<double>)
#endif
}
}
