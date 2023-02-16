//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <whip.hpp>

#include "dlaf/gpu/assert.cu.h"
#include "dlaf/gpu/blas/api.h"
#include "dlaf/lapack/gpu/add.h"
#include "dlaf/types.h"
#include "dlaf/util_cublas.h"
#include "dlaf/util_math.h"

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

struct AddParams {
  static constexpr unsigned kernel_tile_size_rows = 64;
  static constexpr unsigned kernel_tile_size_cols = 16;
};

template <class T>
__device__ inline void addAlpha(const T& alpha, const T& a, T& b) {
  b = b + alpha * a;
}

template <class T>
__device__ inline void sum(const T& /*alpha*/, const T& a, T& b) {
  b = b + a;
}

template <class T>
__device__ inline void sub(const T& /*alpha*/, const T& a, T& b) {
  b = b - a;
}

template <class T, void (*add)(const T&, const T&, T&)>
__device__ void addAllInternal(const unsigned m, const unsigned n, const T& alpha, const T* a,
                               const unsigned lda, T* b, const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = AddParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = AddParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  if (i >= m)
    return;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);

  for (unsigned k = j; k < k_max; ++k)
    add(alpha, a[i + k * lda], b[i + k * ldb]);
}

template <class T>
__device__ inline void addAll(const unsigned m, const unsigned n, const T& alpha, const T* a,
                              const unsigned lda, T* b, const unsigned ldb) {
  if (real(alpha) == 1 && imag(alpha) == 0)
    addAllInternal<T, sum>(m, n, alpha, a, lda, b, ldb);
  else if (real(alpha) == -1 && imag(alpha) == 0)
    addAllInternal<T, sub>(m, n, alpha, a, lda, b, ldb);
  else
    addAllInternal<T, addAlpha>(m, n, alpha, a, lda, b, ldb);
}

template <bool (*comp)(unsigned, unsigned), class T, void (*add)(const T&, const T&, T&)>
__device__ void addDiagInternal(const unsigned m, const unsigned n, const T& alpha, const T* a,
                                const unsigned lda, T* b, const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = AddParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = AddParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  if (i >= m)
    return;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);

  for (unsigned k = j; k < k_max; ++k)
    if (comp(i, k))
      add(alpha, a[i + k * lda], b[i + k * ldb]);
}

template <bool (*comp)(unsigned, unsigned), class T>
__device__ inline void addDiag(const unsigned m, const unsigned n, const T& alpha, const T* a,
                               const unsigned lda, T* b, const unsigned ldb) {
  if (real(alpha) == 1 && imag(alpha) == 0)
    addDiagInternal<comp, T, sum>(m, n, alpha, a, lda, b, ldb);
  else if (real(alpha) == -1 && imag(alpha) == 0)
    addDiagInternal<comp, T, sub>(m, n, alpha, a, lda, b, ldb);
  else
    addDiagInternal<comp, T, addAlpha>(m, n, alpha, a, lda, b, ldb);
}

template <class T>
__global__ void add(cublasFillMode_t uplo, const unsigned m, const unsigned n, const T alpha, const T* a,
                    const unsigned lda, T* b, const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = AddParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = AddParams::kernel_tile_size_cols;

  DLAF_GPU_ASSERT_HEAVY(kernel_tile_size_rows % kernel_tile_size_cols == 0);
  DLAF_GPU_ASSERT_HEAVY(kernel_tile_size_rows == blockDim.x);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.y);
  DLAF_GPU_ASSERT_HEAVY(1 == blockDim.z);
  DLAF_GPU_ASSERT_HEAVY(gridDim.x == ceilDiv(m, kernel_tile_size_rows));
  DLAF_GPU_ASSERT_HEAVY(gridDim.y == ceilDiv(n, kernel_tile_size_cols));
  DLAF_GPU_ASSERT_HEAVY(1 == gridDim.z);

  const unsigned i = blockIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols / kernel_tile_size_rows;

  // Note: if (i == j) the kernel tile contains parts of the diagonal

  switch (uplo) {
    case CUBLAS_FILL_MODE_LOWER:
      if (i == j)
        addDiag<dlaf::util::isLower>(m, n, alpha, a, lda, b, ldb);
      else if (i > j)
        addAll(m, n, alpha, a, lda, b, ldb);
      break;
    case CUBLAS_FILL_MODE_UPPER:
      if (i == j)
        addDiag<dlaf::util::isUpper>(m, n, alpha, a, lda, b, ldb);
      else if (i < j)
        addAll(m, n, alpha, a, lda, b, ldb);
      break;
    case CUBLAS_FILL_MODE_FULL:
      addAll(m, n, alpha, a, lda, b, ldb);
      break;
  }
}
}

template <class T>
void add(const blas::Uplo uplo, const SizeType m, const SizeType n, const T& alpha, const T* a,
         const SizeType lda, T* b, const SizeType ldb, const whip::stream_t stream) {
  if (m == 0 || n == 0)
    return;

  DLAF_ASSERT_HEAVY(m <= lda, m, lda);
  DLAF_ASSERT_HEAVY(m <= ldb, m, ldb);

  constexpr unsigned kernel_tile_size_rows = kernels::AddParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = kernels::AddParams::kernel_tile_size_cols;

  const unsigned um = to_uint(m);
  const unsigned un = to_uint(n);

  const dim3 nr_threads(kernel_tile_size_rows, 1);
  const dim3 nr_blocks(util::ceilDiv(um, kernel_tile_size_rows),
                       util::ceilDiv(un, kernel_tile_size_cols));
  kernels::add<<<nr_blocks, nr_threads, 0, stream>>>(util::blasToCublas(uplo), um, un,
                                                     util::cppToCudaCast(alpha), util::cppToCudaCast(a),
                                                     to_uint(lda), util::cppToCudaCast(b), to_uint(ldb));
}

DLAF_CUBLAS_ADD_ETI(, float);
DLAF_CUBLAS_ADD_ETI(, double);
DLAF_CUBLAS_ADD_ETI(, std::complex<float>);
DLAF_CUBLAS_ADD_ETI(, std::complex<double>);
}
