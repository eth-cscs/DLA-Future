//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <complex>
#include <type_traits>

#include <whip.hpp>

#include <dlaf/eigensolver/tridiag_solver/gpu/kernels.h>
#include <dlaf/util_cuda.h>
#include <dlaf/util_math.h>

namespace dlaf::eigensolver::internal::gpu {

constexpr unsigned cast_complex_kernel_tile_rows = 64;
constexpr unsigned cast_complex_kernel_tile_cols = 16;

template <class T, class CT>
__global__ void castToComplex(const unsigned m, const unsigned n, const T* in, const SizeType ld_in,
                              CT* out, const SizeType ld_out) {
  const unsigned i = blockIdx.x * cast_complex_kernel_tile_rows + threadIdx.x;
  const unsigned j = blockIdx.y * cast_complex_kernel_tile_cols + threadIdx.y;

  if (i >= m || j >= n)
    return;

  SizeType in_id = i + j * ld_in;
  SizeType out_id = i + j * ld_out;

  if constexpr (std::is_same<T, float>::value) {
    out[out_id] = make_cuComplex(in[in_id], 0);
  }
  else {
    out[out_id] = make_cuDoubleComplex(in[in_id], 0);
  }
}

template <class T>
void castToComplex(const SizeType m, const SizeType n, const T* in, const SizeType ld_in,
                   std::complex<T>* out, const SizeType ld_out, whip::stream_t stream) {
  const unsigned um = to_uint(m);
  const unsigned un = to_uint(n);
  dim3 nr_threads(cast_complex_kernel_tile_rows, cast_complex_kernel_tile_cols);
  dim3 nr_blocks(util::ceilDiv(um, cast_complex_kernel_tile_rows),
                 util::ceilDiv(un, cast_complex_kernel_tile_cols));
  castToComplex<<<nr_blocks, nr_threads, 0, stream>>>(um, un, util::cppToCudaCast(in), ld_in,
                                                      util::cppToCudaCast(out), ld_out);
}

DLAF_GPU_CAST_TO_COMPLEX_ETI(, float);
DLAF_GPU_CAST_TO_COMPLEX_ETI(, double);

constexpr unsigned assemble_rank1_kernel_sz = 256;

template <class T>
__global__ void assembleRank1UpdateVectorTile(int sign, SizeType n, SizeType ld_evecs, const T* evecs,
                                              T* rank1) {
  const SizeType j = blockIdx.x * assemble_rank1_kernel_sz + threadIdx.x;
  if (j >= n)
    return;

  if constexpr (std::is_same<T, float>::value) {
    rank1[j] = sign * evecs[j * ld_evecs] / sqrtf(T(2));
  }
  else {
    rank1[j] = sign * evecs[j * ld_evecs] / sqrt(T(2));
  }
}

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho, const SizeType m, const SizeType n,
                                   const T* evecs, const SizeType ld_evecs, T* rank1,
                                   whip::stream_t stream) {
  // Copy the bottom row of the top tile or the top row of the bottom tile
  SizeType i = (is_top_tile) ? m - 1 : 0;

  // Negate Q1's last row if rho < 0
  //
  // lapack 3.10.0, dlaed2.f, line 280 and 281
  int sign = (is_top_tile && rho < 0) ? -1 : 1;

  dim3 nr_threads(assemble_rank1_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(n), assemble_rank1_kernel_sz));
  assembleRank1UpdateVectorTile<<<nr_blocks, nr_threads, 0, stream>>>(sign, n, ld_evecs, evecs + i,
                                                                      rank1);
}

DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(, float);
DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(, double);

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
