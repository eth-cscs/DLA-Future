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
#include <cub/cub.cuh>

namespace dlaf::eigensolver::internal {

template <class T>
T maxElementOnDevice(SizeType len, const T* arr, cudaStream_t stream) {
  auto d_max_ptr = thrust::max_element(thrust::cuda::par.on(stream), arr, arr + len);
  T max_el;
  // TODO: this is a peformance pessimization, the value is on device
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

template <class T>
__global__ void cuppensDecompOnDevice(const T* offdiag_val, T* top_diag_val, T* bottom_diag_val) {
  if constexpr (std::is_same<T, float>::value) {
    *top_diag_val -= fabsf(*offdiag_val);
    *bottom_diag_val -= fabsf(*offdiag_val);
  }
  else {
    *top_diag_val -= fabs(*offdiag_val);
    *bottom_diag_val -= fabs(*offdiag_val);
  }
}

// Refence: Lapack working notes: LAWN 69, Serial Cuppen algorithm, Chapter 3
//
template <class T>
T cuppensDecompOnDevice(const T* d_offdiag_val, T* d_top_diag_val, T* d_bottom_diag_val,
                        cudaStream_t stream) {
  cuppensDecompOnDevice<<<1, 1, 0, stream>>>(d_offdiag_val, d_top_diag_val, d_bottom_diag_val);

  // TODO: this is a peformance pessimization, the value is on device
  T h_offdiag_val;
  cudaMemcpy(&h_offdiag_val, d_offdiag_val, sizeof(T), cudaMemcpyDeviceToHost);
  return h_offdiag_val;
}

DLAF_CUDA_CUPPENS_DECOMP_ETI(, float);
DLAF_CUDA_CUPPENS_DECOMP_ETI(, double);

// --- Eigenvector formation kernels ---

constexpr unsigned evecs_diag_kernel_sz = 32;

template <class T>
__global__ void scaleByDiagonal(SizeType nrows, SizeType ncols, SizeType ld, const T* d_rows,
                                const T* d_cols, const T* evecs, T* ws) {
  const SizeType i = blockIdx.x * evecs_diag_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * evecs_diag_kernel_sz + threadIdx.y;

  if (i >= nrows || j >= ncols)
    return;

  const SizeType idx = i + j * ld;
  const T di = d_rows[i];
  const T dj = d_cols[j];

  // Save rows of `evals` into columns of `ws` to allow subsequent multiply-reduce along rows of contiguos memory
  ws[j + i * ld] = (di == dj) ? evecs[idx] : evecs[idx] / (di - dj);
}

template <class T>
__global__ void transposeFirstRow(SizeType ncols, SizeType ld, T* ws) {
  const SizeType i = blockIdx.x * evecs_diag_kernel_sz + threadIdx.x;

  if (i >= ncols)
    return;

  ws[i] = ws[i * ld];
}

template <class T>
void updateEigenvectorsWithDiagonal(SizeType nrows, SizeType ncols, SizeType ld, const T* d_rows,
                                    const T* d_cols, const T* evecs, T* ws, cudaStream_t stream) {
  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  {
    dim3 nr_threads(evecs_diag_kernel_sz, evecs_diag_kernel_sz);
    dim3 nr_blocks(util::ceilDiv(unrows, evecs_diag_kernel_sz),
                   util::ceilDiv(uncols, evecs_diag_kernel_sz));
    scaleByDiagonal<<<nr_threads, nr_blocks, 0, stream>>>(nrows, ncols, ld, d_rows, d_cols, evecs, ws);
  }

  // Multiply along rows
  //
  // Note: the output of the reduction is saved in the first column.
  auto mult_op = [] __device__(const T& a, const T& b) { return a * b; };

  size_t temp_storage_bytes;
  cub::DeviceReduce::Reduce(NULL, temp_storage_bytes, &ws[0], &ws[0], ncols, mult_op, T(1), stream);
  void* d_temp_storage = memory::internal::getUmpireDeviceAllocator().allocate(temp_storage_bytes);

  // TODO: consider using a custom iterator that can do this in bulk for all rows with a single call
  for (SizeType i = 0; i < nrows; ++i) {
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, &ws[i * ld], &ws[i * ld], ncols,
                              mult_op, T(1), stream);
  }

  // Transpose values from first row into first column
  {
    dim3 nr_threads(evecs_diag_kernel_sz);
    dim3 nr_blocks(util::ceilDiv(unrows, evecs_diag_kernel_sz));
    transposeFirstRow<<<nr_threads, nr_blocks, 0, stream>>>(nrows, ld, ws);
  }
}

DLAF_CUDA_UPDATE_EVECS_WITH_DIAG_ETI(, float);
DLAF_CUDA_UPDATE_EVECS_WITH_DIAG_ETI(, double);

constexpr unsigned mult_cols_kernel_sz = 256;

template <class T>
__global__ void multiplyColumns(SizeType len, const T* in, T* out) {
  const SizeType i = blockIdx.x * copy_tile_row_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  out[i] *= in[i];
}

template <class T>
void multiplyColumns(SizeType len, const T* in, T* out, cudaStream_t stream) {
  dim3 nr_threads(mult_cols_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), mult_cols_kernel_sz));
  multiplyColumns<<<nr_threads, nr_blocks, 0, stream>>>(len, in, out);
}

DLAF_CUDA_MULTIPLY_COLS_ETI(, float);
DLAF_CUDA_MULTIPLY_COLS_ETI(, double);

constexpr unsigned weight_vec_kernel_sz = 32;

template <class T>
__global__ void calcEvecsFromWeightVec(SizeType nrows, SizeType ncols, SizeType ld, const T* rank1_vec,
                                       const T* weight_vec, T* evecs) {
  const SizeType i = blockIdx.x * evecs_diag_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * evecs_diag_kernel_sz + threadIdx.y;

  if (i >= nrows || j >= ncols)
    return;

  T ws_el = weight_vec[i];
  T z_el = rank1_vec[i];
  T& el_evec = evecs[i + j * ld];

  if constexpr (std::is_same<T, float>::value) {
    el_evec = copysignf(sqrtf(fabsf(ws_el)), z_el) / el_evec;
  }
  else {
    el_evec = copysign(sqrt(fabs(ws_el)), z_el) / el_evec;
  }
}

template <class T>
void calcEvecsFromWeightVec(SizeType nrows, SizeType ncols, SizeType ld, const T* rank1_vec,
                            const T* weight_vec, T* evecs, cudaStream_t stream) {
  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  dim3 nr_threads(weight_vec_kernel_sz, weight_vec_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unrows, weight_vec_kernel_sz),
                 util::ceilDiv(uncols, weight_vec_kernel_sz));
  calcEvecsFromWeightVec<<<nr_threads, nr_blocks, 0, stream>>>(nrows, ncols, ld, rank1_vec, weight_vec,
                                                               evecs);
}

DLAF_CUDA_EVECS_FROM_WEIGHT_VEC_ETI(, float);
DLAF_CUDA_EVECS_FROM_WEIGHT_VEC_ETI(, double);

constexpr unsigned sq_kernel_sz = 32;

template <class T>
__global__ void sqTile(SizeType nrows, SizeType ncols, SizeType ld, const T* in, T* out) {
  const SizeType i = blockIdx.x * sq_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * sq_kernel_sz + threadIdx.y;

  if (i >= nrows || j >= ncols)
    return;

  const SizeType idx = i + j * ld;
  out[idx] = in[idx] * in[idx];
}

template <class T>
void sumSqTileOnDevice(SizeType nrows, SizeType ncols, SizeType ld, const T* in, T* out,
                       cudaStream_t stream) {
  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  dim3 nr_threads(sq_kernel_sz, sq_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unrows, sq_kernel_sz), util::ceilDiv(uncols, sq_kernel_sz));
  sqTile<<<nr_threads, nr_blocks, 0, stream>>>(nrows, ncols, ld, in, out);

  // Sum along columns
  //
  // Note: the output of the reduction is saved in the first row.
  size_t temp_storage_bytes;
  cub::DeviceReduce::Sum(NULL, temp_storage_bytes, &out[0], &out[0], nrows, stream);
  void* d_temp_storage = memory::internal::getUmpireDeviceAllocator().allocate(temp_storage_bytes);

  // TODO: consider using a custom iterator which can do the reductions in bulk
  for (SizeType j = 0; j < ncols; ++j) {
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &out[j * ld], &out[j * ld], nrows,
                           stream);
  }
}

DLAF_CUDA_SUM_SQ_TILE_ETI(, float);
DLAF_CUDA_SUM_SQ_TILE_ETI(, double);

constexpr unsigned add_first_rows_kernel_sz = 256;

template <class T>
__global__ void addFirstRows(SizeType len, SizeType ld, const T* in, T* out) {
  const SizeType i = blockIdx.x * add_first_rows_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  out[i * ld] += in[i * ld];
}

template <class T>
void addFirstRows(SizeType len, SizeType ld, const T* in, T* out, cudaStream_t stream) {
  dim3 nr_threads(add_first_rows_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), add_first_rows_kernel_sz));
  addFirstRows<<<nr_threads, nr_blocks, 0, stream>>>(len, ld, in, out);
}

DLAF_CUDA_ADD_FIRST_ROWS_ETI(, float);
DLAF_CUDA_ADD_FIRST_ROWS_ETI(, double);

constexpr unsigned scale_tile_with_row_kernel_sz = 32;

template <class T>
__global__ void scaleTileWithRow(SizeType nrows, SizeType ncols, SizeType ld, const T* norms, T* evecs) {
  const SizeType i = blockIdx.x * scale_tile_with_row_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * scale_tile_with_row_kernel_sz + threadIdx.y;

  if (i >= nrows || j >= ncols)
    return;

  if constexpr (std::is_same<T, float>::value) {
    evecs[i + j * ld] /= sqrtf(norms[j * ld]);
  }
  else {
    evecs[i + j * ld] /= sqrt(norms[j * ld]);
  }
}

template <class T>
void scaleTileWithRow(SizeType nrows, SizeType ncols, SizeType ld, const T* norms, T* evecs,
                      cudaStream_t stream) {
  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  dim3 nr_threads(scale_tile_with_row_kernel_sz, scale_tile_with_row_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unrows, scale_tile_with_row_kernel_sz),
                 util::ceilDiv(uncols, scale_tile_with_row_kernel_sz));
  scaleTileWithRow<<<nr_threads, nr_blocks, 0, stream>>>(nrows, ncols, ld, norms, evecs);
}

DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(, float);
DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(, double);

}
