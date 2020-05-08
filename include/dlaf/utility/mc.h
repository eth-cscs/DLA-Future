//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/utility/internal.h"
#include "dlaf/matrix.h"

namespace dlaf {

template <>
struct Utility<Backend::MC> {
  /// Compute max norm of the distribtued Matrix @param mat_a (https://en.wikipedia.org/wiki/Matrix_norm#Max_norm)
  template <class T>
  static T norm_max(comm::CommunicatorGrid grid, blas::Uplo uplo, Matrix<const T, Device::CPU>& mat_a);
};

}

#include "dlaf/utility/norm_max/mc.tpp"

/// ---- ETI
namespace dlaf {

#define DLAF_NORM_MAX_ETI(KWORD, DATATYPE)                                     \
  KWORD template DATATYPE                                                      \
  Utility<Backend::MC>::norm_max<DATATYPE>(comm::CommunicatorGrid, blas::Uplo, \
                                           Matrix<const DATATYPE, Device::CPU>&);

DLAF_NORM_MAX_ETI(extern, float)
DLAF_NORM_MAX_ETI(extern, double)
// DLAF_NORM_MAX_ETI(extern, std::complex<float>)
// DLAF_NORM_MAX_ETI(extern, std::complex<double>)

}
