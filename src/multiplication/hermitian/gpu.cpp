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

#include <dlaf/multiplication/hermitian/impl.h>

namespace dlaf::multiplication::internal {

DLAF_MULTIPLICATION_HERMITIAN_ETI(, Backend::GPU, Device::GPU, float)
DLAF_MULTIPLICATION_HERMITIAN_ETI(, Backend::GPU, Device::GPU, double)
DLAF_MULTIPLICATION_HERMITIAN_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_MULTIPLICATION_HERMITIAN_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
