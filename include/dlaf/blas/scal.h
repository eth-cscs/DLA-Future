//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <complex>
#include <cstdint>

/// @file
/// Provides overloads for mixed real complex variants of scal missing in blaspp.
/// - csscal
/// - zdscal

namespace blas {

void scal(std::int64_t n, float a, std::complex<float>* x, std::int64_t incx) noexcept;
void scal(std::int64_t n, double a, std::complex<double>* x, std::int64_t incx) noexcept;

}
