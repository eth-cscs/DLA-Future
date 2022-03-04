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

namespace dlaf {
namespace eigensolver {

template <class T, Device device>
struct ReturnTridiagType {
  Matrix<BaseType<T>, device> tridiagonal;
  Matrix<T, device> hh_reflectors;
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

template <Backend backend, Device device, class T>
struct BandToTridiag {};

}
}
}
