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

#include <dlaf/factorization/qr/t_factor_impl.h>

namespace dlaf::factorization::internal {

DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, float)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, double)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, std::complex<float>)
DLAF_FACTORIZATION_QR_TFACTOR_ETI(, Backend::MC, Device::CPU, std::complex<double>)
}
