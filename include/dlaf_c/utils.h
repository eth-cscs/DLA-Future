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

/// @file

#ifdef __cplusplus
#define DLAF_EXTERN_C extern "C"
#else
#define DLAF_EXTERN_C
#endif

#ifdef __cplusplus
#include <complex>
using dlaf_complex_c = std::complex<float>;   ///< Single precision complex number
using dlaf_complex_z = std::complex<double>;  ///< Double precision complex number
#else
#include <complex.h>
typedef float complex dlaf_complex_c;   ///< Single precision complex number
typedef double complex dlaf_complex_z;  ///< Double precision complex number
#endif

#include <dlaf_c/desc.h>

DLAF_EXTERN_C struct DLAF_descriptor make_dlaf_descriptor(int m, int n, int i, int j, int* desc);
