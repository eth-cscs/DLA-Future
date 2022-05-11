//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <lapack/mangling.h>
#include <dlaf/common/assert.h>
#include <dlaf/lapack/laed4.h>

extern "C" {
#define LAPACK_slaed4 LAPACK_GLOBAL(slaed4, SLAED4)
#define LAPACK_dlaed4 LAPACK_GLOBAL(dlaed4, DLAED4)

void LAPACK_slaed4(...);
void LAPACK_dlaed4(...);
}

namespace dlaf {
namespace internal {

void laed4_wrapper(int n, int i, float const* d, float const* z, float* delta, float rho,
                   float* lambda) noexcept {
  ++i;  // Fortran indexing starts from 1
  int info = 0;
  LAPACK_slaed4(&n, &i, d, z, delta, &rho, lambda, &info);
  DLAF_ASSERT(info == 0, info);
}

void laed4_wrapper(int n, int i, double const* d, double const* z, double* delta, double rho,
                   double* lambda) noexcept {
  ++i;  // Fortran indexing starts from 1
  int info = 0;
  LAPACK_dlaed4(&n, &i, d, z, delta, &rho, lambda, &info);
  DLAF_ASSERT(info == 0, info);
}

}
}
