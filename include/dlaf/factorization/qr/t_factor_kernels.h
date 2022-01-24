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
#include "dlaf/types.h"

namespace dlaf::factorization::internal::tfactor_l {

#ifdef DLAF_WITH_CUDA
template <class T>
void tfactorImplicit1(const SizeType n, const T* tau, const T* v, const SizeType ldv, T* t,
                      const SizeType ldt, cudaStream_t stream);

#define DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(kword, Type)                            \
  kword template void tfactorImplicit1(const SizeType n, const Type* tau, const Type* v, \
                                       const SizeType ldv, Type* t, const SizeType ldt,  \
                                       cudaStream_t stream)

DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(extern, float);
DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(extern, double);
DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(extern, std::complex<float>);
DLAF_FACTORIZATION_TFACTOR_IMPLICIT1_ETI(extern, std::complex<double>);
// definitions in src/factorization/qr/tfactor_kernels.cu
#endif

}
