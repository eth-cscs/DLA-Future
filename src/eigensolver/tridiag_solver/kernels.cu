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

void setColTypeTile(const ColType& ct, const matrix::Tile<ColType, Device::GPU>& tile,
                    whip::stream_t stream) {
  std::size_t len = to_sizet(tile.size().rows()) * sizeof(ColType);
  ColType* arr = tile.ptr();
  whip::memset_async(arr, static_cast<int>(ct), len, stream);
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
