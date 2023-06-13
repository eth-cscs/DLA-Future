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
#include <dlaf_c/desc.h>

void C_dlaf_pdsyevd(char uplo, int m, double* a, int* desca, double* w, double* z, int* descz,
                    int* info) {
  dlaf_pdsyevd(uplo, m, a, desca, w, z, descz, info);
}

void C_dlaf_eigensolver_d(int dlaf_context, char uplo, double* a, struct DLAF_descriptor desca, double* w, double* z, struct DLAF_descriptor descz) {
  dlaf_eigensolver_d(dlaf_context, uplo, a, desca, w, z, descz);
}

void C_dlaf_eigensolver_s(int dlaf_context, char uplo, float* a, struct DLAF_descriptor desca, float* w, float* z, struct DLAF_descriptor descz) {
  dlaf_eigensolver_s(dlaf_context, uplo, a, desca, w, z, descz);
}
