//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "dlaf/types.h"

namespace dlaf::gpulapack {

template <class T>
void larft0(cublasHandle_t handle, const SizeType n, SizeType k, const T* v, const SizeType ldv,
            const T* tau, T* t, const SizeType ldt);

#define DLAF_CUBLAS_LARFT_ETI(kword, Type)                                                       \
  kword template void larft0(cublasHandle_t handle, const SizeType n, SizeType k, const Type* v, \
                             const SizeType ldv, const Type* tau, Type* t, const SizeType ldt)

DLAF_CUBLAS_LARFT_ETI(extern, float);
DLAF_CUBLAS_LARFT_ETI(extern, double);
DLAF_CUBLAS_LARFT_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LARFT_ETI(extern, std::complex<double>);
}

#endif
