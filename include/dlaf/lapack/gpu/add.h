//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include <complex>

#include <blas.hh>
#include <whip.hpp>

#include <dlaf/gpu/blas/api.h>
#include <dlaf/types.h>

namespace dlaf::gpulapack {

template <class T>
void add(const blas::Uplo uplo, const SizeType m, const SizeType n, const T& alpha, const T* a,
         const SizeType lda, T* b, const SizeType ldb, const whip::stream_t stream);

#define DLAF_CUBLAS_ADD_ETI(kword, Type)                                                                \
  kword template void add(const blas::Uplo uplo, const SizeType m, const SizeType n, const Type& alpha, \
                          const Type* a, const SizeType lda, Type* b, const SizeType ldb,               \
                          const whip::stream_t stream)

DLAF_CUBLAS_ADD_ETI(extern, float);
DLAF_CUBLAS_ADD_ETI(extern, double);
DLAF_CUBLAS_ADD_ETI(extern, std::complex<float>);
DLAF_CUBLAS_ADD_ETI(extern, std::complex<double>);
}

#endif
