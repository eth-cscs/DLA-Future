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

DLAF_EXTERN_C void dlaf_pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int* info);

DLAF_EXTERN_C void dlaf_pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int* info);

DLAF_EXTERN_C void dlaf_cholesky_d(int dlaf_context, char uplo, double* a, struct DLAF_descriptor dlaf_desca);

DLAF_EXTERN_C void dlaf_cholesky_s(int dlaf_context, char uplo, float* a, struct DLAF_descriptor dlaf_desca);
