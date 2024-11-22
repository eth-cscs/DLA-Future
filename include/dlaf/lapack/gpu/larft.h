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
void larft_gemv0(cublasHandle_t handle, const SizeType m, SizeType k, const T* v, const SizeType ldv,
                 const T* tau, T* t, const SizeType ldt);

template <class T>
void larft_gemv1_notau(cublasHandle_t handle, const SizeType m, const SizeType k, const T* v,
                       const SizeType ldv, T* t, const SizeType ldt);

template <class T>
void larft_gemv1_fixtau(const SizeType k, const T* tau, const SizeType inctau, T* t, const SizeType ldt,
                        whip::stream_t stream);

#define DLAF_CUBLAS_LARFT_GEMV_ETI(kword, Type)                                                       \
  kword template void larft_gemv0(cublasHandle_t handle, const SizeType n, SizeType k, const Type* v, \
                                  const SizeType ldv, const Type* tau, Type* t, const SizeType ldt);  \
  kword template void larft_gemv1_notau(cublasHandle_t handle, const SizeType m, const SizeType k,    \
                                        const Type* v, const SizeType ldv, Type* t,                   \
                                        const SizeType ldt);                                          \
  kword template void larft_gemv1_fixtau(const SizeType k, const Type* tau, const SizeType inctau,    \
                                         Type* t, const SizeType ldt, whip::stream_t stream)

DLAF_CUBLAS_LARFT_GEMV_ETI(extern, float);
DLAF_CUBLAS_LARFT_GEMV_ETI(extern, double);
DLAF_CUBLAS_LARFT_GEMV_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LARFT_GEMV_ETI(extern, std::complex<double>);

}

#endif
