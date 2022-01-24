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
#include "dlaf/cublas/error.h"

#define DLAF_DECLARE_CUBLAS_OP(Name) \
  template <typename T>              \
  struct Name

#define DLAF_DEFINE_CUBLAS_OP(Name, Type, f)                         \
  template <>                                                        \
  struct Name<Type> {                                                \
    template <typename... Args>                                      \
    static void call(Args&&... args) {                               \
      DLAF_CUBLAS_CALL(cublas##f##_v2(std::forward<Args>(args)...)); \
    }                                                                \
  }

namespace dlaf::cublas {

// Level 1
DLAF_DECLARE_CUBLAS_OP(Axpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, float, Saxpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, double, Daxpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, std::complex<float>, Caxpy);
DLAF_DEFINE_CUBLAS_OP(Axpy, std::complex<double>, Zaxpy);

// Level 2
DLAF_DECLARE_CUBLAS_OP(Gemv);
DLAF_DEFINE_CUBLAS_OP(Gemv, float, Sgemv);
DLAF_DEFINE_CUBLAS_OP(Gemv, double, Dgemv);
DLAF_DEFINE_CUBLAS_OP(Gemv, std::complex<float>, Cgemv);
DLAF_DEFINE_CUBLAS_OP(Gemv, std::complex<double>, Zgemv);

DLAF_DECLARE_CUBLAS_OP(Trmv);
DLAF_DEFINE_CUBLAS_OP(Trmv, float, Strmv);
DLAF_DEFINE_CUBLAS_OP(Trmv, double, Dtrmv);
DLAF_DEFINE_CUBLAS_OP(Trmv, std::complex<float>, Ctrmv);
DLAF_DEFINE_CUBLAS_OP(Trmv, std::complex<double>, Ztrmv);

// Level 3
DLAF_DECLARE_CUBLAS_OP(Gemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, float, Sgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, double, Dgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, std::complex<float>, Cgemm);
DLAF_DEFINE_CUBLAS_OP(Gemm, std::complex<double>, Zgemm);

DLAF_DECLARE_CUBLAS_OP(Hemm);
DLAF_DEFINE_CUBLAS_OP(Hemm, float, Ssymm);
DLAF_DEFINE_CUBLAS_OP(Hemm, double, Dsymm);
DLAF_DEFINE_CUBLAS_OP(Hemm, std::complex<float>, Chemm);
DLAF_DEFINE_CUBLAS_OP(Hemm, std::complex<double>, Zhemm);

DLAF_DECLARE_CUBLAS_OP(Her2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, float, Ssyr2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, double, Dsyr2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, std::complex<float>, Cher2k);
DLAF_DEFINE_CUBLAS_OP(Her2k, std::complex<double>, Zher2k);

DLAF_DECLARE_CUBLAS_OP(Herk);
DLAF_DEFINE_CUBLAS_OP(Herk, float, Ssyrk);
DLAF_DEFINE_CUBLAS_OP(Herk, double, Dsyrk);
DLAF_DEFINE_CUBLAS_OP(Herk, std::complex<float>, Cherk);
DLAF_DEFINE_CUBLAS_OP(Herk, std::complex<double>, Zherk);

DLAF_DECLARE_CUBLAS_OP(Trmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, float, Strmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, double, Dtrmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, std::complex<float>, Ctrmm);
DLAF_DEFINE_CUBLAS_OP(Trmm, std::complex<double>, Ztrmm);

DLAF_DECLARE_CUBLAS_OP(Trsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, float, Strsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, double, Dtrsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, std::complex<float>, Ctrsm);
DLAF_DEFINE_CUBLAS_OP(Trsm, std::complex<double>, Ztrsm);

}

#endif
