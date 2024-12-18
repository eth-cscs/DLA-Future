//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <cstdint>

#include <blas/config.h>
#include <blas/mangling.h>

#include <dlaf/blas/scal.h>
#include <dlaf/common/assert.h>
#include <dlaf/types.h>

extern "C" {
#define BLAS_csscal BLAS_FORTRAN_NAME(csscal, CSSCAL)
#define BLAS_zdscal BLAS_FORTRAN_NAME(zdscal, ZDSCAL)

void BLAS_csscal(...);
void BLAS_zdscal(...);
}

namespace blas {

void scal(std::int64_t n, float a, std::complex<float>* x, std::int64_t incx) noexcept {
  auto n_ = dlaf::to_signed<blas_int>(n);
  auto incx_ = dlaf::to_signed<blas_int>(incx);
  BLAS_csscal(&n_, &a, x, &incx_);
}

void scal(std::int64_t n, double a, std::complex<double>* x, std::int64_t incx) noexcept {
  auto n_ = dlaf::to_signed<blas_int>(n);
  auto incx_ = dlaf::to_signed<blas_int>(incx);
  BLAS_zdscal(&n_, &a, x, &incx_);
}

}
