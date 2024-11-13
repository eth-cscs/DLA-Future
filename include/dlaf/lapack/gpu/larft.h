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
void larft_gemv0(cublasHandle_t handle, const SizeType n, SizeType k, const T* v, const SizeType ldv,
                 const T* tau, T* t, const SizeType ldt);

#define DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, id)                                              \
  kword void larft_gemv##id(const SizeType n, SizeType k, const Type* v, const SizeType ldv, Type* t, \
                            const SizeType ldt, whip::stream_t stream)

#define DLAF_CUBLAS_LARFT_GEMV_INTERNAL_SET(kword, Type) \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 100);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 101);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 102);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 103);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 110);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 111);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 120);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 121);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 122);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 200);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 201);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 202);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 203);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 210);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 211);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 212);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 213);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 214);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 220);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 221);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 222);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 223);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 224);     \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL(kword, Type, 225)

DLAF_CUBLAS_LARFT_GEMV_INTERNAL_SET(template <class T>, T);

#define DLAF_CUBLAS_LARFT_GEMV_ETI(kword, Type)                                                       \
  kword template void larft_gemv0(cublasHandle_t handle, const SizeType n, SizeType k, const Type* v, \
                                  const SizeType ldv, const Type* tau, Type* t, const SizeType ldt);  \
  DLAF_CUBLAS_LARFT_GEMV_INTERNAL_SET(kword template, Type)

DLAF_CUBLAS_LARFT_GEMV_ETI(extern, float);
DLAF_CUBLAS_LARFT_GEMV_ETI(extern, double);
DLAF_CUBLAS_LARFT_GEMV_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LARFT_GEMV_ETI(extern, std::complex<double>);

}

#endif
