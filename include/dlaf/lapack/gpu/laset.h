//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include <blas.hh>
#include <whip.hpp>

#include <dlaf/gpu/blas/api.h>
#include <dlaf/types.h>

namespace dlaf::gpulapack {

template <class T>
void laset(blas::Uplo uplo, SizeType m, SizeType n, T alpha, T beta, T* a, SizeType lda,
           whip::stream_t stream);

#define DLAF_CUBLAS_LASET_ETI(kword, Type)                                                           \
  kword template void laset(blas::Uplo uplo, SizeType m, SizeType n, Type alpha, Type beta, Type* a, \
                            SizeType lda, whip::stream_t stream)

DLAF_CUBLAS_LASET_ETI(extern, float);
DLAF_CUBLAS_LASET_ETI(extern, double);
DLAF_CUBLAS_LASET_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LASET_ETI(extern, std::complex<double>);

template <class T>
void gemv_conj_gpu(int m, int n, const T alpha, const T* A, int lda, const T* x, const T beta, T* y,
                   whip::stream_t stream);

#define DLAF_CUSTOM_GEMV_ETI(kword, Type)                                                   \
  kword template void gemv_conj_gpu(int m, int n, const Type alpha, const Type* A, int lda, \
                                    const Type* x, const Type beta, Type* y, whip::stream_t stream)

DLAF_CUSTOM_GEMV_ETI(extern, float);
DLAF_CUSTOM_GEMV_ETI(extern, double);
DLAF_CUSTOM_GEMV_ETI(extern, std::complex<float>);
DLAF_CUSTOM_GEMV_ETI(extern, std::complex<double>);

}

#endif
