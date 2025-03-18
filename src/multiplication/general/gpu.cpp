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

#include <dlaf/multiplication/general/impl.h>

namespace dlaf::multiplication::internal {

DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::GPU, Device::GPU, float)
DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::GPU, Device::GPU, double)
DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::GPU, Device::GPU, std::complex<double>)

}
