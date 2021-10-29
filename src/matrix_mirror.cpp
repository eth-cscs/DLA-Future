//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/matrix/matrix_mirror.h>

namespace dlaf {
namespace matrix {

DLAF_MATRIX_MIRROR_ETI(, float, Device::CPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(, double, Device::CPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<float>, Device::CPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<double>, Device::CPU, Device::CPU)

#ifdef DLAF_WITH_GPU
DLAF_MATRIX_MIRROR_ETI(, float, Device::CPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(, double, Device::CPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<float>, Device::CPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<double>, Device::CPU, Device::GPU)

DLAF_MATRIX_MIRROR_ETI(, float, Device::GPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(, double, Device::GPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<float>, Device::GPU, Device::CPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<double>, Device::GPU, Device::CPU)

DLAF_MATRIX_MIRROR_ETI(, float, Device::GPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(, double, Device::GPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<float>, Device::GPU, Device::GPU)
DLAF_MATRIX_MIRROR_ETI(, std::complex<double>, Device::GPU, Device::GPU)
#endif

}
}
