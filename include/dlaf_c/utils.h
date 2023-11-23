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
#define DLAF_NOEXCEPT noexcept
#else
#define DLAF_EXTERN_C
#define DLAF_NOEXCEPT
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

/// Make a DLA-Future descriptor from a ScaLAPACK descriptor
///
/// @param m Number of rows to be operated on (number of rows in the distributed submatrix)
/// @param n Number of columns to be operated on (number of columns in the distributed submatrix)
/// @param i Row index in the global matrix indicating the first row of the submatrix
/// @param j Column index in the global matrix indicating the first colum index of the submatrix
/// @param desc ScaLAPACK descriptor
/// @return DLA-Future descriptor
DLAF_EXTERN_C struct DLAF_descriptor make_dlaf_descriptor(const int m, const int n, const int i,
                                                          const int j, const int desc[9]) DLAF_NOEXCEPT;
