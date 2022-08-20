//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/misc_gpu_kernels.h"
#include "dlaf/gpu/api.h"
#include "dlaf/gpu/lapack/error.h"
#include "dlaf/memory/memory_chunk.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/util_cuda.h"
#include "dlaf/util_math.h"

#include <cuComplex.h>
#include <cusolverDn.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/merge.h>
#include <thrust/partition.h>

namespace dlaf::eigensolver::internal {

template <class T>
T maxElementOnDevice(SizeType len, const T* arr, cudaStream_t stream) {
  auto d_max_ptr = thrust::max_element(thrust::cuda::par.on(stream), arr, arr + len);
  T max_el;
  // TODO: this is slow and should be fixed eventually
  cudaMemcpy(&max_el, d_max_ptr, sizeof(T), cudaMemcpyDeviceToHost);
  return max_el;
}

DLAF_CUDA_MAX_ELEMENT_ETI(, float);
DLAF_CUDA_MAX_ELEMENT_ETI(, double);

// Note: that this blocks the thread until the kernels complete
SizeType stablePartitionIndexOnDevice(SizeType n, const ColType* c_ptr, const SizeType* in_ptr,
                                      SizeType* out_ptr, cudaStream_t stream) {
  // The number of non-deflated values
  SizeType k = n - thrust::count(thrust::cuda::par.on(stream), c_ptr, c_ptr + n, ColType::Deflated);

  // Partition while preserving relative order such that deflated entries are at the end
  auto cmp = [c_ptr] __device__(const SizeType& i) { return c_ptr[i] != ColType::Deflated; };
  thrust::stable_partition_copy(thrust::cuda::par.on(stream), in_ptr, in_ptr + n, out_ptr, out_ptr + k,
                                std::move(cmp));
  return k;
}

// https://github.com/NVIDIA/thrust/issues/1515
//
template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr, cudaStream_t stream) {
  auto cmp = [v_ptr] __device__(const SizeType& i1, const SizeType& i2) {
    return v_ptr[i1] < v_ptr[i2];
  };
  // NOTE: The call may be synchronous, to avoid that either wrap in a __global__ function as shown in
  // thrust's `examples/cuda/async_reduce.cu` or use the policy `thrust::cuda::par_nosync.on(stream)` in
  // Thrust >= 1.16 (not shipped with the most recent CUDA Toolkit yet).
  //
  thrust::merge(thrust::cuda::par.on(stream), begin_ptr, split_ptr, split_ptr, end_ptr, out_ptr,
                std::move(cmp));
}

DLAF_CUDA_MERGE_INDICES_ETI(, float);
DLAF_CUDA_MERGE_INDICES_ETI(, double);

constexpr unsigned apply_index_sz = 256;

template <class T>
__global__ void applyIndexOnDevice(SizeType len, const SizeType* index_arr, const T* in_arr,
                                   T* out_arr) {
  const SizeType i = blockIdx.x * apply_index_sz + threadIdx.x;
  if (i >= len)
    return;

  out_arr[i] = in_arr[index_arr[i]];
}

template <class T>
void applyIndexOnDevice(SizeType len, const SizeType* index, const T* in, T* out, cudaStream_t stream) {
  dim3 nr_threads(apply_index_sz);
  dim3 nr_blocks(util::ceilDiv(to_sizet(len), to_sizet(apply_index_sz)));
  applyIndexOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, index, util::cppToCudaCast(in),
                                                           util::cppToCudaCast(out));
}

DLAF_CUDA_APPLY_INDEX_ETI(, float);
DLAF_CUDA_APPLY_INDEX_ETI(, double);

constexpr unsigned cast_complex_kernel_tile_rows = 64;
constexpr unsigned cast_complex_kernel_tile_cols = 16;

template <class T, class CT>
__global__ void castTileToComplex(const unsigned m, const unsigned n, SizeType ld, const T* in,
                                  CT* out) {
  const unsigned i = blockIdx.x * cast_complex_kernel_tile_rows + threadIdx.x;
  const unsigned j = blockIdx.y * cast_complex_kernel_tile_cols + threadIdx.y;

  if (i >= m || j >= n)
    return;

  SizeType idx = i + j * ld;
  if constexpr (std::is_same<T, float>::value) {
    out[idx] = make_cuComplex(in[idx], 0);
  }
  else {
    out[idx] = make_cuDoubleComplex(in[idx], 0);
  }
}

template <class T>
void castTileToComplex(SizeType m, SizeType n, SizeType ld, const T* in, std::complex<T>* out,
                       cudaStream_t stream) {
  const unsigned um = to_uint(m);
  const unsigned un = to_uint(n);
  dim3 nr_threads(cast_complex_kernel_tile_rows, cast_complex_kernel_tile_cols);
  dim3 nr_blocks(util::ceilDiv(um, cast_complex_kernel_tile_rows),
                 util::ceilDiv(un, cast_complex_kernel_tile_cols));
  castTileToComplex<<<nr_blocks, nr_threads, 0, stream>>>(um, un, ld, util::cppToCudaCast(in),
                                                          util::cppToCudaCast(out));
}

DLAF_CUDA_CAST_TO_COMPLEX(, float);
DLAF_CUDA_CAST_TO_COMPLEX(, double);

constexpr unsigned invert_index_kernel_sz = 256;

__global__ void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out) {
  const SizeType i = blockIdx.x * invert_index_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  out[in[i]] = i;
}

void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out, cudaStream_t stream) {
  dim3 nr_threads(invert_index_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_sizet(len), to_sizet(invert_index_kernel_sz)));
  invertIndexOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, in, out);
}

constexpr unsigned init_index_tile_kernel_sz = 256;

__global__ void initIndexTile(SizeType offset, SizeType len, SizeType* index_arr) {
  const SizeType i = blockIdx.x * init_index_tile_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  index_arr[i] = i + offset;
}

void initIndexTile(SizeType offset, SizeType len, SizeType* index_arr, cudaStream_t stream) {
  dim3 nr_threads(init_index_tile_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), init_index_tile_kernel_sz));
  initIndexTile<<<nr_blocks, nr_threads, 0, stream>>>(offset, len, index_arr);
}

constexpr unsigned coltype_kernel_sz = 256;

__global__ void setColTypeTile(ColType ct, SizeType len, ColType* ct_arr) {
  const SizeType i = blockIdx.x * coltype_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  ct_arr[i] = ct;
}

void setColTypeTile(ColType ct, SizeType len, ColType* ct_arr, cudaStream_t stream) {
  dim3 nr_threads(coltype_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), coltype_kernel_sz));
  setColTypeTile<<<nr_blocks, nr_threads, 0, stream>>>(ct, len, ct_arr);
}

constexpr unsigned copy_tile_row_kernel_sz = 256;

template <class T>
__global__ void copyTileRowAndNormalizeOnDevice(int sign, SizeType len, SizeType tile_ld, const T* tile,
                                                T* col) {
  const SizeType i = blockIdx.x * copy_tile_row_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  if constexpr (std::is_same<T, float>::value) {
    col[i] = sign * tile[i * tile_ld] / sqrtf(T(2));
  }
  else {
    col[i] = sign * tile[i * tile_ld] / sqrt(T(2));
  }
}

template <class T>
void copyTileRowAndNormalizeOnDevice(int sign, SizeType len, SizeType tile_ld, const T* tile, T* col,
                                     cudaStream_t stream) {
  dim3 nr_threads(copy_tile_row_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), copy_tile_row_kernel_sz));
  copyTileRowAndNormalizeOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(sign, len, tile_ld, tile, col);
}

DLAF_COPY_TILE_ROW_ETI(, float);
DLAF_COPY_TILE_ROW_ETI(, double);

constexpr unsigned givens_rot_kernel_sz = 256;

template <class T>
__global__ void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s) {
  const SizeType i = blockIdx.x * givens_rot_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  T tmp = c * x[i] + s * y[i];
  y[i] = c * y[i] - s * x[i];
  x[i] = tmp;
}

template <class T>
void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s, cudaStream_t stream) {
  dim3 nr_threads(givens_rot_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), givens_rot_kernel_sz));
  givensRotationOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, x, y, c, s);
}

DLAF_GIVENS_ROT_ETI(, float);
DLAF_GIVENS_ROT_ETI(, double);

constexpr unsigned set_diag_kernel_sz = 256;

template <class T>
__global__ void setUnitDiagTileOnDevice(SizeType len, SizeType ld, T* tile) {
  const SizeType i = blockIdx.x * givens_rot_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  tile[i + i * ld] = T(1);
}

template <class T>
void setUnitDiagTileOnDevice(SizeType len, SizeType ld, T* tile, cudaStream_t stream) {
  dim3 nr_threads(set_diag_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), set_diag_kernel_sz));
  setUnitDiagTileOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, ld, tile);
}

DLAF_SET_UNIT_DIAG_ETI(, float);
DLAF_SET_UNIT_DIAG_ETI(, double);

constexpr unsigned copy_diag_tile_kernel_sz = 256;

template <class T>
__global__ void copyDiagTileFromTridiagTile(SizeType len, const T* tridiag, T* diag) {
  const SizeType i = blockIdx.x * copy_diag_tile_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  diag[i] = tridiag[i];
}

template <class T>
void copyDiagTileFromTridiagTile(SizeType len, const T* tridiag, T* diag, cudaStream_t stream) {
  dim3 nr_threads(copy_diag_tile_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), copy_diag_tile_kernel_sz));
  copyDiagTileFromTridiagTile<<<nr_blocks, nr_threads, 0, stream>>>(len, tridiag, diag);
}

DLAF_COPY_DIAG_TILE_ETI(, float);
DLAF_COPY_DIAG_TILE_ETI(, double);

constexpr unsigned tridiag_kernel_sz = 32;

template <class T>
__global__ void expandTridiagonalToLowerTriangular(SizeType n, const T* diag, const T* offdiag,
                                                   SizeType ld_evecs, T* evecs) {
  const SizeType i = blockIdx.x * tridiag_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * tridiag_kernel_sz + threadIdx.y;
  const SizeType idx = i + j * ld_evecs;

  if (i >= n || j >= n || j > i) {
    return;  // only the lower triangular part is set
  }
  else if (i == j) {
    evecs[idx] = diag[i];
  }
  else if (i == j + 1) {
    evecs[idx] = offdiag[j];
  }
  else {
    evecs[idx] = T(0);
  }
}

__global__ void assertSyevdInfo(int* info) {
  if (*info != 0) {
    printf("Error SYEVD: info != 0 (%d)\n", *info);
    __trap();
  }
}

// `evals` [in / out] on entry holds the diagonal of the tridiagonal matrix, on exit holds the
// eigenvalues in the first column
//
// `evecs` [out] first holds the tridiagonal tile converted to an expanded form - lower triangular
// matrix, on exit it holds the eigenvectors tridiag holds the eigenvectors on exit
//
// Example of using cusolverDnXsyevd() is provided here:
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xsyevd/cusolver_Xsyevd_example.cu
template <class T>
void syevdTile(cusolverDnHandle_t handle, SizeType n, T* evals, const T* offdiag, SizeType ld_evecs,
               T* evecs) {
  // Expand from compact tridiagonal form into lower triangular form
  cudaStream_t stream;
  DLAF_GPULAPACK_CHECK_ERROR(cusolverDnGetStream(handle, &stream));
  const unsigned un = to_uint(n);
  dim3 nr_threads(tridiag_kernel_sz, tridiag_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(un, tridiag_kernel_sz), util::ceilDiv(un, tridiag_kernel_sz));
  expandTridiagonalToLowerTriangular<<<nr_blocks, nr_threads, 0, stream>>>(n, evals, offdiag, ld_evecs,
                                                                           evecs);

  // Determine additional memory needed and solve the symmetric eigenvalue problem
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;  // compute both eigenvalues and eigenvectors
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;     // the symmetric matrix is stored in the lower part
  cudaDataType dtype = (std::is_same<T, float>::value) ? CUDA_R_32F : CUDA_R_64F;

  size_t workspaceInBytesOnDevice;
  size_t workspaceInBytesOnHost;
  DLAF_GPULAPACK_CHECK_ERROR(
      cusolverDnXsyevd_bufferSize(handle, NULL, jobz, uplo, n, dtype, evecs, ld_evecs, dtype, evals,
                                  dtype, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  void* bufferOnDevice = memory::internal::getUmpireDeviceAllocator().allocate(workspaceInBytesOnDevice);
  void* bufferOnHost = memory::internal::getUmpireHostAllocator().allocate(workspaceInBytesOnHost);

  // Note: `info` has to be stored on device!
  memory::MemoryView<int, Device::GPU> info(1);
  DLAF_GPULAPACK_CHECK_ERROR(cusolverDnXsyevd(handle, NULL, jobz, uplo, n, dtype, evecs, ld_evecs, dtype,
                                              evals, dtype, bufferOnDevice, workspaceInBytesOnDevice,
                                              bufferOnHost, workspaceInBytesOnHost, info()));
  assertSyevdInfo<<<1, 1, 0, stream>>>(info());

  memory::internal::getUmpireDeviceAllocator().deallocate(bufferOnDevice);
  memory::internal::getUmpireHostAllocator().deallocate(bufferOnHost);
}

DLAF_CUSOLVER_SYEVC_ETI(, float);
DLAF_CUSOLVER_SYEVC_ETI(, double);

}
