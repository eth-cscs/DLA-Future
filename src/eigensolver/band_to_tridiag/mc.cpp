//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/band_to_tridiag/mc.h"

namespace dlaf::eigensolver::internal {

DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::CPU, float)
DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::CPU, double)
DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::CPU, std::complex<double>)

#ifdef DLAF_WITH_GPU
DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::GPU, float)
DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::GPU, double)
DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::GPU, std::complex<float>)
DLAF_EIGENSOLVER_B2T_ETI(, Backend::MC, Device::GPU, std::complex<double>)
#endif
}
