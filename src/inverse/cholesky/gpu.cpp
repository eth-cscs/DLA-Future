//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>

#include <dlaf/inverse/cholesky/impl.h>

namespace dlaf::inverse::internal {

DLAF_INVERSE_CHOLESKY_ETI(, Backend::GPU, Device::GPU, float)
DLAF_INVERSE_CHOLESKY_ETI(, Backend::GPU, Device::GPU, double)
DLAF_INVERSE_CHOLESKY_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_INVERSE_CHOLESKY_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
