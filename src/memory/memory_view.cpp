//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/memory/memory_view.h>

namespace dlaf {
namespace memory {

DLAF_MEMVIEW_ETI(, float, Device::CPU)
DLAF_MEMVIEW_ETI(, double, Device::CPU)
DLAF_MEMVIEW_ETI(, std::complex<float>, Device::CPU)
DLAF_MEMVIEW_ETI(, std::complex<double>, Device::CPU)

#ifdef DLAF_WITH_GPU
DLAF_MEMVIEW_ETI(, float, Device::GPU)
DLAF_MEMVIEW_ETI(, double, Device::GPU)
DLAF_MEMVIEW_ETI(, std::complex<float>, Device::GPU)
DLAF_MEMVIEW_ETI(, std::complex<double>, Device::GPU)
#endif

}
}
