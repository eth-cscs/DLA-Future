//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/mc.h"

namespace dlaf::eigensolver::internal {

DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(, Backend::GPU, Device::GPU, float)
DLAF_TRIDIAGONAL_EIGENSOLVER_ETI(, Backend::GPU, Device::GPU, double)

}
