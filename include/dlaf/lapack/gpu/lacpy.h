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
#include <blas.hh>

#include "dlaf/types.h"

namespace dlaf::gpulapack {

template <class T>
void lacpy(blas::Uplo uplo, SizeType m, SizeType n, const T* a, SizeType lda, T* b, SizeType ldb,
           cudaStream_t stream);

#define DLAF_CUBLAS_LACPY_ETI(kword, Type)                                                        \
  kword template void lacpy(blas::Uplo uplo, SizeType m, SizeType n, const Type* a, SizeType lda, \
                            Type* b, SizeType ldb, cudaStream_t stream)

DLAF_CUBLAS_LACPY_ETI(extern, float);
DLAF_CUBLAS_LACPY_ETI(extern, double);
DLAF_CUBLAS_LACPY_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LACPY_ETI(extern, std::complex<double>);
}

#endif
