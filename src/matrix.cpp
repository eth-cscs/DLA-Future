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

#include <dlaf/matrix/matrix.h>

namespace dlaf {
namespace matrix {

DLAF_MATRIX_ETI(, float, Device::CPU)
DLAF_MATRIX_ETI(, double, Device::CPU)
DLAF_MATRIX_ETI(, std::complex<float>, Device::CPU)
DLAF_MATRIX_ETI(, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_GPU)
DLAF_MATRIX_ETI(, float, Device::GPU)
DLAF_MATRIX_ETI(, double, Device::GPU)
DLAF_MATRIX_ETI(, std::complex<float>, Device::GPU)
DLAF_MATRIX_ETI(, std::complex<double>, Device::GPU)
#endif
}
}
