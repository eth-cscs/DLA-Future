//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <whip.hpp>

#include <pika/cuda.hpp>

#include <dlaf/eigensolver/tridiag_solver/kernels.h>
#include <dlaf/gpu/blas/api.h>
#include <dlaf/gpu/blas/error.h>
#include <dlaf/gpu/cub/api.cu.h>
#include <dlaf/gpu/lapack/api.h>
#include <dlaf/gpu/lapack/error.h>
#include <dlaf/memory/memory_chunk.h>
#include <dlaf/memory/memory_view.h>
#include <dlaf/util_cuda.h>
#include <dlaf/util_math.h>

namespace dlaf::eigensolver::internal {

constexpr unsigned cast_complex_kernel_tile_rows = 64;
constexpr unsigned cast_complex_kernel_tile_cols = 16;

template <class T, class CT>
__global__ void castToComplex(const unsigned m, const unsigned n, SizeType ld, const T* in, CT* out) {
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
void castToComplex(const matrix::Tile<const T, Device::GPU>& in,
                   const matrix::Tile<std::complex<T>, Device::GPU>& out, whip::stream_t stream) {
  SizeType m = in.size().rows();
  SizeType n = in.size().cols();
  SizeType ld = in.ld();
  const T* in_ptr = in.ptr();
  std::complex<T>* out_ptr = out.ptr();

  const unsigned um = to_uint(m);
  const unsigned un = to_uint(n);
  dim3 nr_threads(cast_complex_kernel_tile_rows, cast_complex_kernel_tile_cols);
  dim3 nr_blocks(util::ceilDiv(um, cast_complex_kernel_tile_rows),
                 util::ceilDiv(un, cast_complex_kernel_tile_cols));
  castToComplex<<<nr_blocks, nr_threads, 0, stream>>>(um, un, ld, util::cppToCudaCast(in_ptr),
                                                      util::cppToCudaCast(out_ptr));
}

DLAF_GPU_CAST_TO_COMPLEX_ETI(, float);
DLAF_GPU_CAST_TO_COMPLEX_ETI(, double);

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::GPU>& diag_tile,
                                        whip::stream_t stream) {
  SizeType len = tridiag_tile.size().rows();
  const T* tridiag_ptr = tridiag_tile.ptr();
  T* diag_ptr = diag_tile.ptr();

  whip::memcpy_async(diag_ptr, tridiag_ptr, sizeof(T) * to_sizet(len), whip::memcpy_default, stream);
}

DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(, float);
DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(, double);

constexpr unsigned assemble_rank1_kernel_sz = 256;

template <class T>
__global__ void assembleRank1UpdateVectorTile(int sign, SizeType len, SizeType tile_ld, const T* tile,
                                              T* col) {
  const SizeType i = blockIdx.x * assemble_rank1_kernel_sz + threadIdx.x;
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
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::GPU>& evecs_tile,
                                   const matrix::Tile<T, Device::GPU>& rank1_tile,
                                   whip::stream_t stream) {
  // Copy the bottom row of the top tile or the top row of the bottom tile
  SizeType row = (is_top_tile) ? rank1_tile.size().rows() - 1 : 0;

  // Negate Q1's last row if rho < 0
  //
  // lapack 3.10.0, dlaed2.f, line 280 and 281
  int sign = (is_top_tile && rho < 0) ? -1 : 1;

  SizeType len = evecs_tile.size().cols();
  SizeType tile_ld = evecs_tile.ld();
  const T* tile = evecs_tile.ptr(TileElementIndex(row, 0));
  T* col = rank1_tile.ptr();

  dim3 nr_threads(assemble_rank1_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), assemble_rank1_kernel_sz));
  assembleRank1UpdateVectorTile<<<nr_blocks, nr_threads, 0, stream>>>(sign, len, tile_ld, tile, col);
}

DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(, float);
DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(, double);

template <class T>
__global__ void maxElementInColumnTileOnDevice(const T* begin_ptr, const T* end_ptr,
                                               T* device_max_el_ptr) {
#ifdef DLAF_WITH_CUDA
  constexpr auto par = ::thrust::cuda::par;
#elif defined(DLAF_WITH_HIP)
  constexpr auto par = ::thrust::hip::par;
#endif

  // NOTE: This could also use max_element. However, it fails to compile with
  // HIP, so we use reduce as an alternative which works with both HIP and CUDA.
  *device_max_el_ptr = thrust::reduce(par, begin_ptr, end_ptr, *begin_ptr, thrust::maximum<T>());
}

template <class T>
void maxElementInColumnTile(const matrix::Tile<const T, Device::GPU>& tile, T* host_max_el_ptr,
                            T* device_max_el_ptr, whip::stream_t stream) {
  SizeType len = tile.size().rows();
  const T* arr = tile.ptr();

  DLAF_ASSERT(len > 0, len);

  maxElementInColumnTileOnDevice<<<1, 1, 0, stream>>>(arr, arr + len, device_max_el_ptr);
  whip::memcpy_async(host_max_el_ptr, device_max_el_ptr, sizeof(T), whip::memcpy_device_to_host, stream);
}

DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(, float);
DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(, double);

void setColTypeTile(const ColType& ct, const matrix::Tile<ColType, Device::GPU>& tile,
                    whip::stream_t stream) {
  std::size_t len = to_sizet(tile.size().rows()) * sizeof(ColType);
  ColType* arr = tile.ptr();
  whip::memset_async(arr, static_cast<int>(ct), len, stream);
}

constexpr unsigned init_index_tile_kernel_sz = 256;

__global__ void initIndexTile(SizeType offset, SizeType len, SizeType* index_arr) {
  const SizeType i = blockIdx.x * init_index_tile_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  index_arr[i] = i + offset;
}

void initIndexTile(SizeType offset, const matrix::Tile<SizeType, Device::GPU>& tile,
                   whip::stream_t stream) {
  SizeType len = tile.size().rows();
  SizeType* index_arr = tile.ptr();

  dim3 nr_threads(init_index_tile_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), init_index_tile_kernel_sz));
  initIndexTile<<<nr_blocks, nr_threads, 0, stream>>>(offset, len, index_arr);
}

struct StrideOp {
  SizeType ld;
  SizeType offset;

  __host__ __device__ __forceinline__ SizeType operator()(const SizeType i) const {
    return offset + i * ld;
  }
};

template <class T>
struct Row2ColMajor {
  SizeType ld;
  SizeType ncols;
  T* data;

  __host__ __device__ __forceinline__ T operator()(const SizeType idx) const {
    SizeType i = idx / ncols;
    SizeType j = idx - i * ncols;
    return data[i + j * ld];
  }
};

constexpr unsigned set_diag_kernel_sz = 256;

template <class T>
__global__ void setUnitDiagTileOnDevice(SizeType len, SizeType ld, T* tile) {
  const SizeType i = blockIdx.x * set_diag_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  tile[i + i * ld] = T(1);
}

template <class T>
void setUnitDiagonal(const SizeType& k, const SizeType& tile_begin,
                     const matrix::Tile<T, Device::GPU>& tile, whip::stream_t stream) {
  SizeType tile_offset = k - tile_begin;
  if (tile_offset < 0)
    tile_offset = 0;
  else if (tile_offset >= tile.size().rows())
    return;

  SizeType len = tile.size().rows() - tile_offset;
  SizeType ld = tile.ld();
  T* tile_ptr = tile.ptr(TileElementIndex(tile_offset, tile_offset));

  dim3 nr_threads(set_diag_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), set_diag_kernel_sz));
  setUnitDiagTileOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, ld, tile_ptr);
}

DLAF_GPU_SET_UNIT_DIAGONAL_ETI(, float);
DLAF_GPU_SET_UNIT_DIAGONAL_ETI(, double);

// -----------------------------------------
// This is a separate struct with a call operator instead of a lambda, because
// nvcc does not compile the file with a lambda.
struct PartitionIndicesPredicate {
  const ColType* c_ptr;
  __device__ bool operator()(const SizeType i) {
    return c_ptr[i] != ColType::Deflated;
  }
};

__global__ void stablePartitionIndexOnDevice(SizeType n, const ColType* c_ptr, const SizeType* in_ptr,
                                             SizeType* out_ptr, SizeType* device_k_ptr) {
#ifdef DLAF_WITH_CUDA
  constexpr auto par = thrust::cuda::par;
#elif defined(DLAF_WITH_HIP)
  constexpr auto par = thrust::hip::par;
#endif

  SizeType& k = *device_k_ptr;

  // The number of non-deflated values
  k = n - thrust::count(par, c_ptr, c_ptr + n, ColType::Deflated);

  // Partition while preserving relative order such that deflated entries are at the end
  thrust::stable_partition_copy(par, in_ptr, in_ptr + n, out_ptr, out_ptr + k,
                                PartitionIndicesPredicate{c_ptr});
}

void stablePartitionIndexOnDevice(SizeType n, const ColType* c_ptr, const SizeType* in_ptr,
                                  SizeType* out_ptr, SizeType* host_k_ptr, SizeType* device_k_ptr,
                                  whip::stream_t stream) {
  stablePartitionIndexOnDevice<<<1, 1, 0, stream>>>(n, c_ptr, in_ptr, out_ptr, device_k_ptr);
  whip::memcpy_async(host_k_ptr, device_k_ptr, sizeof(SizeType), whip::memcpy_device_to_host, stream);
}

template <class T>
__global__ void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr,
                                     const SizeType* end_ptr, SizeType* out_ptr, const T* v_ptr) {
  auto cmp = [v_ptr](const SizeType& i1, const SizeType& i2) { return v_ptr[i1] < v_ptr[i2]; };

#ifdef DLAF_WITH_CUDA
  constexpr auto par = thrust::cuda::par;
#elif defined(DLAF_WITH_HIP)
  constexpr auto par = thrust::hip::par;
#endif

  thrust::merge(par, begin_ptr, split_ptr, split_ptr, end_ptr, out_ptr, std::move(cmp));
}

template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr, whip::stream_t stream) {
  mergeIndicesOnDevice<<<1, 1, 0, stream>>>(begin_ptr, split_ptr, end_ptr, out_ptr, v_ptr);
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
void applyIndexOnDevice(SizeType len, const SizeType* index, const T* in, T* out,
                        whip::stream_t stream) {
  dim3 nr_threads(apply_index_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), apply_index_sz));
  applyIndexOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, index, util::cppToCudaCast(in),
                                                           util::cppToCudaCast(out));
}

DLAF_CUDA_APPLY_INDEX_ETI(, float);
DLAF_CUDA_APPLY_INDEX_ETI(, double);

constexpr unsigned invert_index_kernel_sz = 256;

__global__ void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out) {
  const SizeType i = blockIdx.x * invert_index_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  out[in[i]] = i;
}

void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out, whip::stream_t stream) {
  dim3 nr_threads(invert_index_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), invert_index_kernel_sz));
  invertIndexOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, in, out);
}

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
void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s, whip::stream_t stream) {
  dim3 nr_threads(givens_rot_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(len), givens_rot_kernel_sz));
  givensRotationOnDevice<<<nr_blocks, nr_threads, 0, stream>>>(len, x, y, c, s);
}

DLAF_GIVENS_ROT_ETI(, float);
DLAF_GIVENS_ROT_ETI(, double);

}
