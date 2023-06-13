//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "eigensolver.h"

#include <dlaf_c/eigensolver/eigensolver.h>

void dlaf_pssyevd(char uplo, int m, float* a, int* desca, float* w, float* z, int* descz, int* info) {
  pxsyevd<float>(uplo, m, a, desca, w, z, descz, *info);
}

void dlaf_pdsyevd(char uplo, int m, double* a, int* desca, double* w, double* z, int* descz, int* info) {
  pxsyevd<double>(uplo, m, a, desca, w, z, descz, *info);
}

void dlaf_eigensolver_s(int dlaf_context, char uplo, float* a, struct DLAF_descriptor dlaf_desca,
                        float* w, float* z, struct DLAF_descriptor dlaf_descz) {
  eigensolver<float>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}

void dlaf_eigensolver_d(int dlaf_context, char uplo, double* a, struct DLAF_descriptor dlaf_desca,
                        double* w, double* z, struct DLAF_descriptor dlaf_descz) {
  eigensolver<double>(dlaf_context, uplo, a, dlaf_desca, w, z, dlaf_descz);
}
