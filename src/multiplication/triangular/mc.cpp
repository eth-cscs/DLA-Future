//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/multiplication/triangular/impl.h>

namespace dlaf {
namespace multiplication {
namespace internal {

DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, float)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, double)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_MULTIPLICATION_TRIANGULAR_ETI(, Backend::MC, Device::CPU, std::complex<double>)
}
}
}
