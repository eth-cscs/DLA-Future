//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <whip.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/gpu/assert.cu.h>
#include <dlaf/gpu/blas/api.h>
#include <dlaf/gpu/blas/gpublas.h>
#include <dlaf/lapack/gpu/larft.h>
#include <dlaf/types.h>
#include <dlaf/util_cublas.h>
#include <dlaf/util_math.h>

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

template <unsigned kts_rows_t, unsigned kts_cols_t, class T>
__global__ void fix_tau(const unsigned k, const T* tau, unsigned inctau, T* t, unsigned ldt) {
  DLAF_GPU_ASSERT_HEAVY(kts_rows_t == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(kts_cols_t == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_rows_t) == gridDim.x);
  DLAF_GPU_ASSERT_HEAVY(ceilDiv(k, kts_cols_t) == gridDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  const unsigned i = blockIdx.x * kts_rows_t + threadIdx.x;
  const unsigned j = blockIdx.y * kts_cols_t + threadIdx.y;

  // quick return if outside of t.
  if (i >= k || j >= k)
    return;

  T& t_ij = t[i + j * ldt];
  T tau_j = tau[j * inctau];
  if (i > j)
    t_ij = {};
  else if (i == j)
    t_ij = tau_j;
  else
    t_ij = -tau_j * t_ij;
}
}

template <class T>
void larft_gemv0(cublasHandle_t handle, const SizeType m, const SizeType k, const T* v,
                 const SizeType ldv, const T* tau, T* t, const SizeType ldt) {
  DLAF_ASSERT(m >= 0, m);
  DLAF_ASSERT(k >= 0, k);
  DLAF_ASSERT(ldv >= m, ldv, m);
  DLAF_ASSERT(ldt >= k, ldt, k);

  const int m_ = to_int(m);
  const int k_ = to_int(k);
  const int ldv_ = to_int(ldv);
  const int ldt_ = to_int(ldt);

  auto v_ = [v, ldv_](int i, int j) { return util::blasToCublasCast(v + i + j * ldv_); };
  auto t_ = [t, ldt_](int i, int j) { return util::blasToCublasCast(t + i + j * ldt_); };

  for (int j = 1; j < k_; ++j) {
    const auto mtau = util::blasToCublasCast(-tau[j]);
    const auto one = util::blasToCublasCast(T{1});
    gpublas::internal::Gemv<T>::call(handle, CUBLAS_OP_C, m_, j, &mtau, v_(0, 0), ldv_, v_(0, j), 1,
                                     &one, t_(0, j), 1);
  }
}

template <class T>
void larft_gemv1_notau(cublasHandle_t handle, const SizeType m, const SizeType k, const T* v,
                       const SizeType ldv, T* t, const SizeType ldt) {
  DLAF_ASSERT(m >= 0, m);
  DLAF_ASSERT(k >= 0, k);
  DLAF_ASSERT(ldv >= m, ldv, m);
  DLAF_ASSERT(ldt >= k, ldt, k);

  const int m_ = to_int(m);
  const int k_ = to_int(k);
  const int ldv_ = to_int(ldv);
  const int ldt_ = to_int(ldt);

  auto v_ = util::blasToCublasCast(v);
  auto t_ = util::blasToCublasCast(t);

  const auto one = util::blasToCublasCast(T{1});
  gpublas::internal::Gemm<T>::call(handle, CUBLAS_OP_C, CUBLAS_OP_N, k_, k_, m_, &one, v_, ldv_, v_,
                                   ldv_, &one, t_, ldt_);
}

template <class T>
void larft_gemv1_fixtau(const SizeType k, const T* tau, const SizeType inctau, T* t, const SizeType ldt,
                        whip::stream_t stream) {
  constexpr unsigned kts_rows_t = 32;
  constexpr unsigned kts_cols_t = 32;

  DLAF_ASSERT(k >= 0, k);
  DLAF_ASSERT(ldt >= k, ldt, k);
  DLAF_ASSERT(inctau >= 1, inctau);

  if (k == 0)
    return;

  const unsigned uk = to_uint(k);
  const unsigned uinctau = to_uint(inctau);
  const unsigned uldt = to_uint(ldt);

  dim3 nr_threads(kts_rows_t, kts_cols_t);
  dim3 nr_blocks(util::ceilDiv(uk, kts_rows_t), util::ceilDiv(uk, kts_cols_t));

  auto tau_ = util::blasToCublasCast(tau);
  auto t_ = util::blasToCublasCast(t);
  kernels::fix_tau<kts_rows_t, kts_cols_t>
      <<<nr_blocks, nr_threads, 0, stream>>>(uk, tau_, uinctau, t_, uldt);
}

DLAF_CUBLAS_LARFT_GEMV_ETI(, float);
DLAF_CUBLAS_LARFT_GEMV_ETI(, double);
DLAF_CUBLAS_LARFT_GEMV_ETI(, std::complex<float>);
DLAF_CUBLAS_LARFT_GEMV_ETI(, std::complex<double>);

}
