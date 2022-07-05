//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/permutations/general/impl.h"

namespace dlaf::permutations::internal {

DLAF_PERMUTATIONS_GENERAL_ETI(, Backend::GPU, Device::GPU, float)
DLAF_PERMUTATIONS_GENERAL_ETI(, Backend::GPU, Device::GPU, double)
DLAF_PERMUTATIONS_GENERAL_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_PERMUTATIONS_GENERAL_ETI(, Backend::GPU, Device::GPU, std::complex<double>)

}
