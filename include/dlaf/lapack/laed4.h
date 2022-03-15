//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

namespace dlaf {
namespace internal {

void laed4_wrapper(int n, int i, float const* d, float const* z, float* delta, float rho, float* lambda);

void laed4_wrapper(int n, int i, double const* d, double const* z, double* delta, double rho,
                   double* lambda);

}
}
