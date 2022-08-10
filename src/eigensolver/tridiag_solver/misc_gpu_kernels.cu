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
#include "dlaf/util_cuda.h"
#include "dlaf/util_math.h"

#include <cuComplex.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>

namespace dlaf::eigensolver::internal {

template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr) {
  auto cmp = [v_ptr] __device__(const SizeType& i1, const SizeType& i2) {
    return v_ptr[i1] < v_ptr[i2];
  };
  // TODO: with Thrust > 1.16 use `thrust::cuda::par_nosync.on(args...)` instead of `thrust::device`
  thrust::merge(thrust::device, begin_ptr, split_ptr, split_ptr, end_ptr, out_ptr, std::move(cmp));
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
  const SizeType i = blockIdx.x * init_index_tile_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  ct_arr[i] = ct;
}

void setColTypeTile(ColType ct, SizeType len, ColType* ct_arr, cudaStream_t stream) { 
  dim3 nr_threads(coltype_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), coltype_kernel_sz));
  setColTypeTile<<<nr_blocks, nr_threads, 0, stream>>>(ct, len, ct_arr);
}

}
