//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/eigensolver/tridiag_solver/kernels.h"

#include "dlaf/gpu/blas/error.h"
#include "dlaf/gpu/lapack/error.h"
#include "dlaf/memory/memory_chunk.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/util_cuda.h"
#include "dlaf/util_math.h"

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <pika/cuda.hpp>
#include <whip.hpp>
#include "dlaf/gpu/blas/api.h"
#include "dlaf/gpu/cub/api.h"
#include "dlaf/gpu/lapack/api.h"
#include "dlaf/gpu/lapack/error.h"

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
__global__ void cuppensDecompOnDevice(const T* offdiag_val, T* top_diag_val, T* bottom_diag_val) {
  const T offdiag = *offdiag_val;
  T& top_diag = *top_diag_val;
  T& bottom_diag = *bottom_diag_val;

  if constexpr (std::is_same<T, float>::value) {
    top_diag -= fabsf(offdiag);
    bottom_diag -= fabsf(offdiag);
  }
  else {
    top_diag -= fabs(offdiag);
    bottom_diag -= fabs(offdiag);
  }
}

// Refence: Lapack working notes: LAWN 69, Serial Cuppen algorithm, Chapter 3
//
template <class T>
T cuppensDecomp(const matrix::Tile<T, Device::GPU>& top, const matrix::Tile<T, Device::GPU>& bottom,
                whip::stream_t stream) {
  TileElementIndex offdiag_idx{top.size().rows() - 1, 1};
  TileElementIndex top_idx{top.size().rows() - 1, 0};
  TileElementIndex bottom_idx{0, 0};
  const T* d_offdiag_val = top.ptr(offdiag_idx);
  T* d_top_diag_val = top.ptr(top_idx);
  T* d_bottom_diag_val = bottom.ptr(bottom_idx);

  cuppensDecompOnDevice<<<1, 1, 0, stream>>>(d_offdiag_val, d_top_diag_val, d_bottom_diag_val);

  // TODO: this is a peformance pessimization, the value is on device
  T h_offdiag_val;
  whip::memcpy_async(&h_offdiag_val, d_offdiag_val, sizeof(T), whip::memcpy_device_to_host, stream);

  return h_offdiag_val;
}

DLAF_GPU_CUPPENS_DECOMP_ETI(, float);
DLAF_GPU_CUPPENS_DECOMP_ETI(, double);

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::GPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::GPU>& diag_tile,
                                        whip::stream_t stream) {
  SizeType len = tridiag_tile.size().rows();
  const T* tridiag_ptr = tridiag_tile.ptr();
  T* diag_ptr = diag_tile.ptr();

  whip::memcpy_async(diag_ptr, tridiag_ptr, sizeof(T) * to_sizet(len), whip::memcpy_device_to_device,
                     stream);
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
T maxElementInColumnTile(const matrix::Tile<const T, Device::GPU>& tile, whip::stream_t stream) {
  SizeType len = tile.size().rows();
  const T* arr = tile.ptr();

#ifdef DLAF_WITH_CUDA
  constexpr auto par = ::thrust::cuda::par;
#elif defined(DLAF_WITH_HIP)
  constexpr auto par = ::thrust::hip::par;
#endif
  auto d_max_ptr = thrust::max_element(par.on(stream), arr, arr + len);
  T max_el;
  // TODO: this is a performance pessimization, the value is on device
  whip::memcpy_async(&max_el, d_max_ptr, sizeof(T), whip::memcpy_device_to_host, stream);
  return max_el;
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

  ws[idx] = (di == dj) ? evecs[idx] : evecs[idx] / (di - dj);
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

template <class T>
void divideEvecsByDiagonal(const SizeType& k, const SizeType& i_subm_el, const SizeType& j_subm_el,
                           const matrix::Tile<const T, Device::GPU>& diag_rows,
                           const matrix::Tile<const T, Device::GPU>& diag_cols,
                           const matrix::Tile<const T, Device::GPU>& evecs_tile,
                           const matrix::Tile<T, Device::GPU>& ws_tile, whip::stream_t stream) {
  if (i_subm_el >= k || j_subm_el >= k)
    return;

  SizeType nrows = std::min(k - i_subm_el, evecs_tile.size().rows());
  SizeType ncols = std::min(k - j_subm_el, evecs_tile.size().cols());

  SizeType ld = evecs_tile.ld();
  const T* d_rows = diag_rows.ptr();
  const T* d_cols = diag_cols.ptr();
  const T* evecs = evecs_tile.ptr();
  T* ws = ws_tile.ptr();

  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  dim3 nr_threads(evecs_diag_kernel_sz, evecs_diag_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unrows, evecs_diag_kernel_sz),
                 util::ceilDiv(uncols, evecs_diag_kernel_sz));
  scaleByDiagonal<<<nr_blocks, nr_threads, 0, stream>>>(nrows, ncols, ld, d_rows, d_cols, evecs, ws);

  // Multiply along rows
  //
  // Note: the output of the reduction is saved in the first column.
  auto mult_op = [] __device__(const T& a, const T& b) { return a * b; };
  size_t temp_storage_bytes;

  using OffsetIterator =
      dlaf::gpucub::TransformInputIterator<SizeType, StrideOp,
                                           dlaf::gpucub::CountingInputIterator<SizeType>>;
  using InputIterator =
      dlaf::gpucub::TransformInputIterator<T, Row2ColMajor<T>,
                                           dlaf::gpucub::CountingInputIterator<SizeType>>;

  dlaf::gpucub::CountingInputIterator<SizeType> count_iter(0);
  OffsetIterator begin_offsets(count_iter, StrideOp{ncols, 0});  // first column
  OffsetIterator end_offsets = begin_offsets + 1;                // last column
  InputIterator in_iter(count_iter, Row2ColMajor<T>{ld, ncols, ws});

  whip::check_error(dlaf::gpucub::DeviceSegmentedReduce::Reduce(NULL, temp_storage_bytes, in_iter, ws,
                                                                nrows, begin_offsets, end_offsets,
                                                                mult_op, T(1), stream));
  void* d_temp_storage = memory::internal::getUmpireDeviceAllocator().allocate(temp_storage_bytes);
  whip::check_error(dlaf::gpucub::DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes,
                                                                in_iter, ws, nrows, begin_offsets,
                                                                end_offsets, mult_op, T(1), stream));
  // Deallocate memory
  auto extend_info = [d_temp_storage](whip::error_t status) {
    whip::check_error(status);
    memory::internal::getUmpireDeviceAllocator().deallocate(d_temp_storage);
  };
  pika::cuda::experimental::detail::add_event_callback(std::move(extend_info), stream);
}

DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(, float);
DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(, double);

constexpr unsigned mult_cols_kernel_sz = 256;

template <class T>
__global__ void multiplyColumns(SizeType len, const T* in, T* out) {
  const SizeType i = blockIdx.x * mult_cols_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  out[i] *= in[i];
}

template <class T>
void multiplyFirstColumns(const SizeType& k, const SizeType& row, const SizeType& col,
                          const matrix::Tile<const T, Device::GPU>& in,
                          const matrix::Tile<T, Device::GPU>& out, whip::stream_t stream) {
  if (row >= k || col >= k)
    return;

  SizeType nrows = std::min(k - row, in.size().rows());

  const T* in_ptr = in.ptr();
  T* out_ptr = out.ptr();

  dim3 nr_threads(mult_cols_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(nrows), mult_cols_kernel_sz));
  multiplyColumns<<<nr_blocks, nr_threads, 0, stream>>>(nrows, in_ptr, out_ptr);
}

DLAF_GPU_MULTIPLY_FIRST_COLUMNS_ETI(, float);
DLAF_GPU_MULTIPLY_FIRST_COLUMNS_ETI(, double);

constexpr unsigned weight_vec_kernel_sz = 32;

template <class T>
__global__ void calcEvecsFromWeightVec(SizeType nrows, SizeType ncols, SizeType ld, const T* rank1_vec,
                                       const T* weight_vec, T* evecs) {
  const SizeType i = blockIdx.x * weight_vec_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * weight_vec_kernel_sz + threadIdx.y;

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
void calcEvecsFromWeightVec(const SizeType& k, const SizeType& row, const SizeType& col,
                            const matrix::Tile<const T, Device::GPU>& z_tile,
                            const matrix::Tile<const T, Device::GPU>& ws_tile,
                            const matrix::Tile<T, Device::GPU>& evecs_tile, whip::stream_t stream) {
  if (row >= k || col >= k)
    return;

  SizeType nrows = std::min(k - row, evecs_tile.size().rows());
  SizeType ncols = std::min(k - col, evecs_tile.size().cols());

  SizeType ld = evecs_tile.ld();
  const T* rank1_vec = z_tile.ptr();
  const T* weight_vec = ws_tile.ptr();
  T* evecs = evecs_tile.ptr();

  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  dim3 nr_threads(weight_vec_kernel_sz, weight_vec_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unrows, weight_vec_kernel_sz),
                 util::ceilDiv(uncols, weight_vec_kernel_sz));
  calcEvecsFromWeightVec<<<nr_blocks, nr_threads, 0, stream>>>(nrows, ncols, ld, rank1_vec, weight_vec,
                                                               evecs);
}

DLAF_GPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(, float);
DLAF_GPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(, double);

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
void sumsqCols(const SizeType& k, const SizeType& row, const SizeType& col,
               const matrix::Tile<const T, Device::GPU>& evecs_tile,
               const matrix::Tile<T, Device::GPU>& ws_tile, whip::stream_t stream) {
  if (row >= k || col >= k)
    return;

  SizeType nrows = std::min(k - row, evecs_tile.size().rows());
  SizeType ncols = std::min(k - col, evecs_tile.size().cols());

  SizeType ld = evecs_tile.ld();
  const T* in = evecs_tile.ptr();
  T* out = ws_tile.ptr();

  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  dim3 nr_threads(sq_kernel_sz, sq_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unrows, sq_kernel_sz), util::ceilDiv(uncols, sq_kernel_sz));
  sqTile<<<nr_blocks, nr_threads, 0, stream>>>(nrows, ncols, ld, in, out);

  // Sum along columns
  //
  // Note: the output of the reduction is saved in the first row.
  // TODO: use a segmented reduce sum with fancy iterators
  size_t temp_storage_bytes;
#ifdef DLAF_WITH_CUDA
  whip::check_error(
      dlaf::gpucub::DeviceReduce::Sum(NULL, temp_storage_bytes, &out[0], &out[0], nrows, stream));
#elif defined(DLAF_WITH_HIP)
  whip::check_error(
      dlaf::gpucub::DeviceReduce::Sum(NULL, temp_storage_bytes, &out[0], &out[0], unrows, stream));
#endif
  void* d_temp_storage = memory::internal::getUmpireDeviceAllocator().allocate(temp_storage_bytes);

  for (SizeType j = 0; j < ncols; ++j) {
#ifdef DLAF_WITH_CUDA
    whip::check_error(dlaf::gpucub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &out[j * ld],
                                                      &out[j * ld], nrows, stream));
#elif defined(DLAF_WITH_HIP)
    whip::check_error(dlaf::gpucub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, &out[j * ld],
                                                      &out[j * ld], unrows, stream));
#endif
  }

  // Deallocate memory
  auto extend_info = [d_temp_storage](whip::error_t status) {
    whip::check_error(status);
    memory::internal::getUmpireDeviceAllocator().deallocate(d_temp_storage);
  };
  pika::cuda::experimental::detail::add_event_callback(std::move(extend_info), stream);
}

DLAF_GPU_SUMSQ_COLS_ETI(, float);
DLAF_GPU_SUMSQ_COLS_ETI(, double);

constexpr unsigned add_first_rows_kernel_sz = 256;

template <class T>
__global__ void addFirstRows(SizeType len, SizeType ld, const T* in, T* out) {
  const SizeType i = blockIdx.x * add_first_rows_kernel_sz + threadIdx.x;
  if (i >= len)
    return;

  out[i * ld] += in[i * ld];
}

template <class T>
void addFirstRows(const SizeType& k, const SizeType& row, const SizeType& col,
                  const matrix::Tile<const T, Device::GPU>& in, const matrix::Tile<T, Device::GPU>& out,
                  whip::stream_t stream) {
  if (row >= k || col >= k)
    return;

  SizeType ncols = std::min(k - col, in.size().cols());

  SizeType ld = in.ld();
  const T* in_ptr = in.ptr();
  T* out_ptr = out.ptr();

  dim3 nr_threads(add_first_rows_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(ncols), add_first_rows_kernel_sz));
  addFirstRows<<<nr_blocks, nr_threads, 0, stream>>>(ncols, ld, in_ptr, out_ptr);
}

DLAF_GPU_ADD_FIRST_ROWS_ETI(, float);
DLAF_GPU_ADD_FIRST_ROWS_ETI(, double);

constexpr unsigned scale_tile_with_row_kernel_sz = 32;

template <class T>
__global__ void scaleTileWithRow(SizeType nrows, SizeType ncols, SizeType in_ld, const T* in_ptr,
                                 SizeType out_ld, T* out_ptr) {
  const SizeType i = blockIdx.x * scale_tile_with_row_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * scale_tile_with_row_kernel_sz + threadIdx.y;

  if (i >= nrows || j >= ncols)
    return;

  const T in_el = in_ptr[j * in_ld];
  T& out_el = out_ptr[i + j * out_ld];

  if constexpr (std::is_same<T, float>::value) {
    out_el = out_el / sqrtf(in_el);
  }
  else {
    out_el = out_el / sqrt(in_el);
  }
}

template <class T>
void divideColsByFirstRow(const SizeType& k, const SizeType& row, const SizeType& col,
                          const matrix::Tile<const T, Device::GPU>& in,
                          const matrix::Tile<T, Device::GPU>& out, whip::stream_t stream) {
  if (row >= k || col >= k)
    return;

  SizeType nrows = std::min(k - row, out.size().rows());
  SizeType ncols = std::min(k - col, out.size().cols());

  SizeType in_ld = in.ld();
  const T* in_ptr = in.ptr();
  SizeType out_ld = out.ld();
  T* out_ptr = out.ptr();

  const unsigned unrows = to_uint(nrows);
  const unsigned uncols = to_uint(ncols);
  dim3 nr_threads(scale_tile_with_row_kernel_sz, scale_tile_with_row_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unrows, scale_tile_with_row_kernel_sz),
                 util::ceilDiv(uncols, scale_tile_with_row_kernel_sz));
  scaleTileWithRow<<<nr_blocks, nr_threads, 0, stream>>>(nrows, ncols, in_ld, in_ptr, out_ld, out_ptr);
}

DLAF_GPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(, float);
DLAF_GPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(, double);

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

// Reference to CUBLAS 1D copy(): https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-copy
template <class T>
void copy1D(cublasHandle_t handle, const SizeType& k, const SizeType& row, const SizeType& col,
            const Coord& in_coord, const matrix::Tile<const T, Device::GPU>& in_tile,
            const Coord& out_coord, const matrix::Tile<T, Device::GPU>& out_tile) {
  if (row >= k || col >= k)
    return;

  const T* in_ptr = in_tile.ptr();
  T* out_ptr = out_tile.ptr();

  int in_ld = (in_coord == Coord::Col) ? 1 : to_int(in_tile.ld());
  int out_ld = (out_coord == Coord::Col) ? 1 : to_int(out_tile.ld());

  // if `in_tile` is the column buffer
  SizeType len = (out_coord == Coord::Col) ? std::min(out_tile.size().rows(), k - row)
                                           : std::min(out_tile.size().cols(), k - col);
  // if out_tile is the column buffer
  if (out_tile.size().cols() == 1) {
    len = (in_coord == Coord::Col) ? std::min(in_tile.size().rows(), k - row)
                                   : std::min(in_tile.size().cols(), k - col);
  }

  if constexpr (std::is_same<T, float>::value) {
    DLAF_GPUBLAS_CHECK_ERROR(cublasScopy(handle, len, in_ptr, in_ld, out_ptr, out_ld));
  }
  else {
    DLAF_GPUBLAS_CHECK_ERROR(cublasDcopy(handle, len, in_ptr, in_ld, out_ptr, out_ld));
  }
}

DLAF_GPU_COPY_1D_ETI(, float);
DLAF_GPU_COPY_1D_ETI(, double);

// -----------------------------------------

// Note: that this blocks the thread until the kernels complete
SizeType stablePartitionIndexOnDevice(SizeType n, const ColType* c_ptr, const SizeType* in_ptr,
                                      SizeType* out_ptr, whip::stream_t stream) {
  // The number of non-deflated values
#ifdef DLAF_WITH_CUDA
  constexpr auto par = thrust::cuda::par;
#elif defined(DLAF_WITH_HIP)
  constexpr auto par = thrust::hip::par;
#endif
  SizeType k = n - thrust::count(par.on(stream), c_ptr, c_ptr + n, ColType::Deflated);

  // Partition while preserving relative order such that deflated entries are at the end
  auto cmp = [c_ptr] __device__(const SizeType& i) { return c_ptr[i] != ColType::Deflated; };
  thrust::stable_partition_copy(par.on(stream), in_ptr, in_ptr + n, out_ptr, out_ptr + k,
                                std::move(cmp));
  return k;
}

// https://github.com/NVIDIA/thrust/issues/1515
//
template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr, whip::stream_t stream) {
  auto cmp = [v_ptr] __device__(const SizeType& i1, const SizeType& i2) {
    return v_ptr[i1] < v_ptr[i2];
  };
  // NOTE: The call may be synchronous, to avoid that either wrap in a __global__ function as shown in
  // thrust's `examples/cuda/async_reduce.cu` or use the policy `thrust::cuda::par_nosync.on(stream)` in
  // Thrust >= 1.16 (not shipped with the most recent CUDA Toolkit yet).
  //
#ifdef DLAF_WITH_CUDA
  constexpr auto par = thrust::cuda::par;
#elif defined(DLAF_WITH_HIP)
  constexpr auto par = thrust::hip::par;
#endif
  thrust::merge(par.on(stream), begin_ptr, split_ptr, split_ptr, end_ptr, out_ptr, std::move(cmp));
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
  dim3 nr_blocks(util::ceilDiv(to_sizet(len), to_sizet(apply_index_sz)));
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
  dim3 nr_blocks(util::ceilDiv(to_sizet(len), to_sizet(invert_index_kernel_sz)));
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
