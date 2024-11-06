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
void larftJustGEMVs(const SizeType n, SizeType k, const T* v, const SizeType ldv, T* t,
                    const SizeType ldt, whip::stream_t stream);

#define DLAF_CUBLAS_LARFTJUSTGEMVs_ETI(kword, Type)                                                   \
  kword template void larftJustGEMVs(const SizeType n, SizeType k, const Type* v, const SizeType ldv, \
                                     Type* t, const SizeType ldt, whip::stream_t stream)

DLAF_CUBLAS_LARFTJUSTGEMVs_ETI(extern, float);
DLAF_CUBLAS_LARFTJUSTGEMVs_ETI(extern, double);
DLAF_CUBLAS_LARFTJUSTGEMVs_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LARFTJUSTGEMVs_ETI(extern, std::complex<double>);

template <class T>
void larft(const SizeType n, SizeType k, const T* v, const SizeType ldv, const T* tau, T* t,
           const SizeType ldt, whip::stream_t stream);

#define DLAF_CUBLAS_LARFT_ETI(kword, Type)                                                   \
  kword template void larft(const SizeType n, SizeType k, const Type* v, const SizeType ldv, \
                            const Type* tau, Type* t, const SizeType ldt, whip::stream_t stream)

DLAF_CUBLAS_LARFT_ETI(extern, float);
DLAF_CUBLAS_LARFT_ETI(extern, double);
DLAF_CUBLAS_LARFT_ETI(extern, std::complex<float>);
DLAF_CUBLAS_LARFT_ETI(extern, std::complex<double>);
}

#endif
