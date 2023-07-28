//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/matrix/matrix_ref.h>

namespace dlaf::matrix::internal {

DLAF_MATRIX_REF_ETI(, float, Device::CPU)
DLAF_MATRIX_REF_ETI(, double, Device::CPU)
DLAF_MATRIX_REF_ETI(, std::complex<float>, Device::CPU)
DLAF_MATRIX_REF_ETI(, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_GPU)
DLAF_MATRIX_REF_ETI(, float, Device::GPU)
DLAF_MATRIX_REF_ETI(, double, Device::GPU)
DLAF_MATRIX_REF_ETI(, std::complex<float>, Device::GPU)
DLAF_MATRIX_REF_ETI(, std::complex<double>, Device::GPU)
#endif
}
