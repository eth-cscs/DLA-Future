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

#include <dlaf/gpu/assert.cu.h>
#include <dlaf/gpu/blas/api.h>
#include <dlaf/lapack/gpu/lacpy.h>
#include <dlaf/types.h>
#include <dlaf/util_cublas.h>
#include <dlaf/util_math.h>

namespace dlaf::gpulapack {
namespace kernels {

using namespace dlaf::util::cuda_operators;

struct LacpyParams {
  static constexpr unsigned kernel_tile_size_rows = 64;
  static constexpr unsigned kernel_tile_size_cols = 16;
};

template <class T>
__device__ void copyAll(const unsigned m, const unsigned n, const T* a, const unsigned lda, T* b,
                        const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LacpyParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  if (i >= m)
    return;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);

  for (unsigned k = j; k < k_max; ++k)
    b[i + k * ldb] = a[i + k * lda];
}

template <bool (*comp)(unsigned, unsigned), class T>
__device__ void copyDiag(const unsigned m, const unsigned n, const T* a, const unsigned lda, T* b,
                         const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LacpyParams::kernel_tile_size_cols;

  const unsigned i = blockIdx.x * kernel_tile_size_rows + threadIdx.x;
  const unsigned j = blockIdx.y * kernel_tile_size_cols;

  if (i >= m)
    return;

  const unsigned k_max = min(j + kernel_tile_size_cols, n);

  for (unsigned k = j; k < k_max; ++k)
    if (comp(i, k))
      b[i + k * ldb] = a[i + k * lda];
}

template <class T>
__global__ void lacpy(cublasFillMode_t uplo, const unsigned m, const unsigned n, const T* a,
                      const unsigned lda, T* b, const unsigned ldb) {
  constexpr unsigned kernel_tile_size_rows = LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = LacpyParams::kernel_tile_size_cols;

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
        copyDiag<dlaf::util::isLower>(m, n, a, lda, b, ldb);
      else if (i > j)
        copyAll(m, n, a, lda, b, ldb);
      break;
    case CUBLAS_FILL_MODE_UPPER:
      if (i == j)
        copyDiag<dlaf::util::isUpper>(m, n, a, lda, b, ldb);
      else if (i < j)
        copyAll(m, n, a, lda, b, ldb);
      break;
    case CUBLAS_FILL_MODE_FULL:
      // Note: it is more appropriate to use cudaMemcpy2DAsync in this case
      DLAF_GPU_ASSERT_HEAVY(false);
      // copyAll(m, n, a, lda, b, ldb);
      break;
  }
}
}

template <class T>
void lacpy(const blas::Uplo uplo, const SizeType m, const SizeType n, const T* a, const SizeType lda,
           T* b, const SizeType ldb, const whip::stream_t stream) {
  if (m == 0 || n == 0)
    return;

  DLAF_ASSERT_HEAVY(m <= lda, m, lda);
  DLAF_ASSERT_HEAVY(m <= ldb, m, ldb);

  constexpr unsigned kernel_tile_size_rows = kernels::LacpyParams::kernel_tile_size_rows;
  constexpr unsigned kernel_tile_size_cols = kernels::LacpyParams::kernel_tile_size_cols;

  if (uplo == blas::Uplo::General) {
    whip::memcpy_kind kind = whip::memcpy_default;

    // If HIP is version 5.6.0 or newer, do not use hipMemcpyDefault as it is buggy with 2D memcpy.
    // Instead try to infer the memory type using hipPointerGetAttributes. See:
    // - https://github.com/ROCm/clr/commit/56daa6c4891b43ec233e9c63f755e3f7b45842b4
    // - https://github.com/ROCm/clr/commit/d3bfb55d7a934355257a72fab538a0a634b43cad
#if defined(DLAF_WITH_HIP) && HIP_VERSION >= 50600000
    hipPointerAttribute_t src_attributes{};
    whip::check_error(hipPointerGetAttributes(&src_attributes, a));

    hipPointerAttribute_t dest_attributes{};
    whip::check_error(hipPointerGetAttributes(&dest_attributes, b));

    if (src_attributes.type == hipMemoryTypeDevice && dest_attributes.type == hipMemoryTypeHost) {
      kind = whip::memcpy_device_to_host;
    }
    else if (src_attributes.type == hipMemoryTypeHost && dest_attributes.type == hipMemoryTypeDevice) {
      kind = whip::memcpy_host_to_device;
    }
    else if (src_attributes.type == hipMemoryTypeDevice && dest_attributes.type == hipMemoryTypeDevice) {
      kind = whip::memcpy_device_to_device;
    }
    else if (src_attributes.type == hipMemoryTypeHost && dest_attributes.type == hipMemoryTypeHost) {
      DLAF_ASSERT(
          false,
          "Attempting to do a HIP lacpy with host source and destination, use the CPU lacpy instead");
    }
    else {
      DLAF_ASSERT(false,
                  "Attempting to do a HIP lacpy with unsupported source and destination memory type",
                  src_attributes.type, dest_attributes.type);
    }
#endif

    whip::memcpy_2d_async(b, to_sizet(ldb) * sizeof(T), a, to_sizet(lda) * sizeof(T),
                          to_sizet(m) * sizeof(T), to_sizet(n), kind, stream);
  }
  else {
    const unsigned um = to_uint(m);
    const unsigned un = to_uint(n);

    const dim3 nr_threads(kernel_tile_size_rows, 1);
    const dim3 nr_blocks(util::ceilDiv(um, kernel_tile_size_rows),
                         util::ceilDiv(un, kernel_tile_size_cols));
    kernels::lacpy<<<nr_blocks, nr_threads, 0, stream>>>(util::blasToCublas(uplo), um, un,
                                                         util::cppToCudaCast(a), to_uint(lda),
                                                         util::cppToCudaCast(b), to_uint(ldb));
  }
}

DLAF_CUBLAS_LACPY_ETI(, float);
DLAF_CUBLAS_LACPY_ETI(, double);
DLAF_CUBLAS_LACPY_ETI(, std::complex<float>);
DLAF_CUBLAS_LACPY_ETI(, std::complex<double>);
}
