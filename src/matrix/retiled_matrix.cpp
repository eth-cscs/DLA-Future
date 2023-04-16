//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/matrix/retiled_matrix.h>

namespace dlaf {
namespace matrix {

DLAF_RETILED_MATRIX_ETI(, float, Device::CPU)
DLAF_RETILED_MATRIX_ETI(, double, Device::CPU)
DLAF_RETILED_MATRIX_ETI(, std::complex<float>, Device::CPU)
DLAF_RETILED_MATRIX_ETI(, std::complex<double>, Device::CPU)

#if defined(DLAF_WITH_GPU)
DLAF_RETILED_MATRIX_ETI(, float, Device::GPU)
DLAF_RETILED_MATRIX_ETI(, double, Device::GPU)
DLAF_RETILED_MATRIX_ETI(, std::complex<float>, Device::GPU)
DLAF_RETILED_MATRIX_ETI(, std::complex<double>, Device::GPU)
#endif
}
}
