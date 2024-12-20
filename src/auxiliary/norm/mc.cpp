//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>

#include <dlaf/auxiliary/norm/mc.h>

namespace dlaf::auxiliary::internal {

DLAF_NORM_ETI(, float)
DLAF_NORM_ETI(, double)
DLAF_NORM_ETI(, std::complex<float>)
DLAF_NORM_ETI(, std::complex<double>)
}
