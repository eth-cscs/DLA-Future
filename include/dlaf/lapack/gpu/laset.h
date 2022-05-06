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

#ifdef DLAF_WITH_GPU

#include "dlaf/gpu/api.h"
#include "dlaf/gpu/blas/api.h"
#include "dlaf/types.h"

namespace dlaf::gpulapack {

template <class T>
void laset(blas::Uplo uplo, SizeType m, SizeType n, T alpha, T beta, T* a, SizeType lda,
           cudaStream_t stream);

#if defined DLAF_WITH_CUDA
#define DLAF_CUBLAS_LASET_ETI(kword, Type)                                                           \
  kword template void laset(blas::Uplo uplo, SizeType m, SizeType n, Type alpha, Type beta, Type* a, \
                            SizeType lda, cudaStream_t stream)

DLAF_CUBLAS_LASET_ETI(extern, float);
DLAF_CUBLAS_LASET_ETI(extern, double);
DLAF_CUBLAS_LASET_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LASET_ETI(extern, std::complex<double>);
#elif defined DLAF_WITH_HIP
template <class T>
void laset(blas::Uplo uplo, SizeType m, SizeType n, T alpha, T beta, T* a, SizeType lda,
           cudaStream_t stream) {}
#endif
}

#endif
