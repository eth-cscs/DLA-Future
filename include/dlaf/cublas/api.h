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

#define DLAF_DECLARE_GPUBLAS_OP(Name) \
  template <typename T>               \
  struct Name

#define DLAF_DEFINE_GPUBLAS_OP(Name, Type, f)                        \
  template <>                                                        \
  struct Name<Type> {                                                \
    template <typename... Args>                                      \
    static void call(Args&&... args) {                               \
      DLAF_CUBLAS_CALL(cublas##f##_v2(std::forward<Args>(args)...)); \
    }                                                                \
  }

#define DLAF_MAKE_GPUBLAS_OP(Name, f)                      \
  DLAF_DECLARE_GPUBLAS_OP(Name);                           \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, S##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, D##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, C##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, Z##f)

#define DLAF_MAKE_GPUBLAS_SYHE_OP(Name, f)                   \
  DLAF_DECLARE_GPUBLAS_OP(Name);                             \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, Ssy##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, Dsy##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, Che##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, Zhe##f)

namespace dlaf::gpublas {

// Level 1
DLAF_MAKE_GPUBLAS_OP(Axpy, axpy);

// Level 2
DLAF_MAKE_GPUBLAS_OP(Gemv, gemv);

DLAF_MAKE_GPUBLAS_OP(Trmv, trmv);

// Level 3
DLAF_MAKE_GPUBLAS_OP(Gemm, gemm);

DLAF_MAKE_GPUBLAS_SYHE_OP(Hemm, mm);

DLAF_MAKE_GPUBLAS_SYHE_OP(Her2k, r2k);

DLAF_MAKE_GPUBLAS_SYHE_OP(Herk, rk);

DLAF_MAKE_GPUBLAS_OP(Trmm, trmm);

DLAF_MAKE_GPUBLAS_OP(Trsm, trsm);

}

#endif
