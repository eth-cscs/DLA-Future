//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_eigensolver_c_api_wrapper.h"

#include <dlaf_c/eigensolver/eigensolver.h>

void C_dlaf_pdsyevd(char uplo, int m, double* a, int* desca, double* w, double* z, int* descz,
                    int* info) {
  dlaf_pdsyevd(uplo, m, a, desca, w, z, descz, info);
}
