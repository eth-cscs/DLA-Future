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

#include <blas.hh>

#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::multiplication {
namespace internal {

template <Backend B, Device D, class T>
struct GeneralSub {
  static void callNN(const SizeType i_tile_from, const SizeType i_tile_to, const blas::Op opA,
                     const blas::Op opB, const T alpha, Matrix<const T, D>& mat_a,
                     Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c);
};

/// ---- ETI
#define DLAF_MULTIPLICATION_GENERAL_ETI(KWORD, BACKEND, DEVICE, DATATYPE) \
  KWORD template struct GeneralSub<BACKEND, DEVICE, DATATYPE>;

DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, float)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, double)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, std::complex<float>)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, float)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, double)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_MULTIPLICATION_GENERAL_ETI(extern, Backend::GPU, Device::GPU, std::complex<double>)
#endif
}
}
