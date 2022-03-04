//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#include "dlaf/eigensolver/band_to_tridiag/mc.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

DLAF_EIGENSOLVER_B2T_MC_ETI(, float)
DLAF_EIGENSOLVER_B2T_MC_ETI(, double)
DLAF_EIGENSOLVER_B2T_MC_ETI(, std::complex<float>)
DLAF_EIGENSOLVER_B2T_MC_ETI(, std::complex<double>)

}
}
}
