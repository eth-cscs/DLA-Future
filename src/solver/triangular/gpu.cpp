//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/solver/triangular/impl.h"

namespace dlaf {
namespace solver {
namespace internal {

DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, float)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, double)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_SOLVER_TRIANGULAR_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
}
}
