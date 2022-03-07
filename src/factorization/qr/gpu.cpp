//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/factorization/qr.h"

namespace dlaf::factorization::internal {

DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_LOCAL_ETI(, Backend::GPU, Device::GPU, std::complex<double>)
}
