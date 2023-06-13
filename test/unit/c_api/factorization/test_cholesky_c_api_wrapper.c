//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_cholesky_c_api_wrapper.h"
#include <dlaf_c/factorization/cholesky.h>

void C_dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info) {
  dlaf_pdpotrf(uplo, n, a, ia, ja, desca, info);
}
