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

DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::MC, Device::CPU, float)
DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::MC, Device::CPU, double)
DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_MULTIPLICATION_GENERAL_ETI(, Backend::MC, Device::CPU, std::complex<double>)

}
