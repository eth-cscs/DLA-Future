//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/gen_to_std/impl.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, float)
DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, double)
DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_GENTOSTD_ETI(, Backend::MC, Device::CPU, std::complex<double>)

}
}
}
