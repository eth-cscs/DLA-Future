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

#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

DLAF_EXTERN_C void C_dlaf_pdsyevd(char uplo, int m, double* a, int* desca, double* w, double* z,
                                  int* descz, int* info);

DLAF_EXTERN_C void C_dlaf_pssyevd(char uplo, int m, float* a, int* desca, float* w, float* z, int* descz,
                                  int* info);

DLAF_EXTERN_C void C_dlaf_eigensolver_d(int dlaf_context, char uplo, double* a,
                                        struct DLAF_descriptor desca, double* w, double* z,
                                        struct DLAF_descriptor descz);

DLAF_EXTERN_C void C_dlaf_eigensolver_s(int dlaf_context, char uplo, float* a,
                                        struct DLAF_descriptor desca, float* w, float* z,
                                        struct DLAF_descriptor descz);
