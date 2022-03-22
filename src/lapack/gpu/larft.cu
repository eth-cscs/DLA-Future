//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cublas_v2.h>
#include "dlaf/common/assert.h"
#include "dlaf/cublas/api.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cuda/assert.cu.h"
#include "dlaf/cuda/error.h"
#include "dlaf/lapack/gpu/larft.h"
#include "dlaf/types.h"
#include "dlaf/util_cublas.h"
#include "dlaf/util_cuda.h"
#include "dlaf/util_math.h"
#include "dlaf/util_matrix.h"

namespace dlaf::gpulapack {

template <class T>
void larft0(cublasHandle_t handle, const SizeType n, SizeType k, const T* v, const SizeType ldv,
            const T* tau, T* t, const SizeType ldt) {
  DLAF_ASSERT(n >= 0, n);
  DLAF_ASSERT(k >= 0, k);

  if (k == 0)
    return;

  cudaStream_t stream;
  DLAF_CUBLAS_CALL(cublasGetStream(handle, &stream));

  DLAF_CUDA_CALL(
      cudaMemset2DAsync(t, sizeof(T) * to_sizet(ldt), 0, sizeof(T) * to_sizet(k), to_sizet(k), stream));

  if (n == 0)
    return;

  // Reduce k excluding reflectors with size 0.
  k = std::min(n, k);

  // Note:
  // T factor is an upper triangular square matrix, built column by column
  // with taus values on the diagonal
  //
  // T(j,j) = tau(j)
  //
  // and in the upper triangular part the following formula applies
  //
  // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
  //
  //
  // The result is achieved in two main steps:
  // 1) t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) T(0:j, j) = T(0:j, 0:j) . t

  // 1st step: compute the column partial result `t`
  // First we compute the matrix vector multiplication for each column
  // -tau(j) . V(j:, 0:j)* . V(j:, j)

  DLAF_CUDA_CALL(cudaMemcpy2DAsync(t, to_sizet(ldt + 1) * sizeof(T), tau, sizeof(T), sizeof(T),
                                   to_sizet(k), cudaMemcpyDefault, stream));

  const int n_ = to_int(n);
  const int k_ = to_int(k);
  const int ldv_ = to_int(ldv);
  const int ldt_ = to_int(ldt);

  auto v_ = [v, ldv_](int i, int j) { return util::blasToCublasCast(v + i + j * ldv_); };
  auto t_ = [t, ldt_](int i, int j) { return util::blasToCublasCast(t + i + j * ldt_); };

  for (int j = 0; j < k_; ++j) {
    const auto mtau = util::blasToCublasCast(-tau[j]);
    const auto one = util::blasToCublasCast(T{1});
    gpublas::Gemv<T>::call(handle, CUBLAS_OP_C, n_ - j, j, &mtau, v_(j, 0), ldv_, v_(j, j), 1, &one,
                           t_(0, j), 1);
  }

  // 2nd step: compute the T factor, by performing the last step on each column
  for (int j = 1; j < k_; ++j) {
    gpublas::Trmv<T>::call(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, j,
                           t_(0, 0), ldt_, t_(0, j), 1);
  }
}

DLAF_CUBLAS_LARFT_ETI(, float);
DLAF_CUBLAS_LARFT_ETI(, double);
DLAF_CUBLAS_LARFT_ETI(, std::complex<float>);
DLAF_CUBLAS_LARFT_ETI(, std::complex<double>);
}
