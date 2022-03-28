//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/eigensolver/impl.h"

namespace dlaf::eigensolver::internal {

DLAF_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_ETI(, Backend::MC, Device::CPU, std::complex<double>)

}
