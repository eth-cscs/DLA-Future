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

#include "dlaf/communication/communicator_grid.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::auxiliary::internal {

template <Backend backend, Device device, class T>
struct Norm {};

template <class T>
struct Norm<Backend::MC, Device::CPU, T> {
  static dlaf::BaseType<T> max_L(comm::CommunicatorGrid comm_grid, comm::Index2D rank,
                                 Matrix<const T, Device::CPU>& matrix);

  static dlaf::BaseType<T> max_G(comm::CommunicatorGrid comm_grid, comm::Index2D rank,
                                 Matrix<const T, Device::CPU>& matrix);
};

// ETI
#define DLAF_NORM_ETI(KWORD, DATATYPE) KWORD template struct Norm<Backend::MC, Device::CPU, DATATYPE>;

DLAF_NORM_ETI(extern, float)
DLAF_NORM_ETI(extern, double)
DLAF_NORM_ETI(extern, std::complex<float>)
DLAF_NORM_ETI(extern, std::complex<double>)
}
