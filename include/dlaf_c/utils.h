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

#ifdef __cplusplus
#define DLAF_EXTERN_C extern "C"
#else
#define DLAF_EXTERN_C
#endif

#ifdef __cplusplus
#include <complex>
using complex_c = std::complex<float>;
using complex_z = std::complex<double>;
#else
#include <complex.h>
typedef float complex complex_c;
typedef double complex complex_z;
#endif
