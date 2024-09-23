//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <vector>

#include <whip.hpp>

#include <dlaf/permutations/general/perms.h>
#include <dlaf/types.h>
#include <dlaf/util_cuda.h>

namespace dlaf::permutations::internal {

__device__ SizeType getIndex(const MatrixLayout& layout, SizeType row, SizeType col) {
  SizeType tile_row = row / layout.nb;
  SizeType tile_col = col / layout.nb;
  SizeType tile_offset = tile_row * layout.row_offset + tile_col * layout.col_offset;

  SizeType tile_el_row = row - tile_row * layout.nb;
  SizeType tile_el_col = col - tile_col * layout.nb;
  SizeType tile_el_offset = tile_el_row + tile_el_col * layout.ld;

  return tile_offset + tile_el_offset;
}

constexpr unsigned perms_kernel_sz = 32;

__device__ void swapIndices(SizeType& a, SizeType& b) {
  SizeType tmp = a;
  a = b;
  b = tmp;
}

template <class T, Coord coord>
__global__ void applyPermutationsOnDevice(SizeType out_begin_row, SizeType out_begin_col,
                                          SizeType nelems, SizeType nperms, SizeType in_offset,
                                          const SizeType* perms, MatrixLayout in_layout, const T* in,
                                          MatrixLayout out_layout, T* out) {
  const SizeType i_el = blockIdx.x * perms_kernel_sz + threadIdx.x;    // column or row index of element
  const SizeType i_perm = blockIdx.y * perms_kernel_sz + threadIdx.y;  // output row or column index

  if (i_el >= nelems || i_perm >= nperms)
    return;

  // Coord::Col
  SizeType in_row = in_offset + i_el;
  SizeType in_col = perms[i_perm];
  SizeType out_row = i_el;
  SizeType out_col = i_perm;

  // Coord::Row
  if constexpr (coord == Coord::Row) {
    swapIndices(in_row, in_col);
    swapIndices(out_row, out_col);
  }

  SizeType in_idx = getIndex(in_layout, in_row, in_col);
  SizeType out_idx = getIndex(out_layout, out_begin_row + out_row, out_begin_col + out_col);
  out[out_idx] = in[in_idx];
}

template <class T, Coord coord>
void applyPermutationsOnDevice(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                               const SizeType* perms, const MatrixLayout in_layout, const T* in,
                               const MatrixLayout out_layout, T* out, whip::stream_t stream) {
  constexpr Coord orth_coord = orthogonal(coord);
  SizeType nelems = sz.get<orth_coord>();  // number of elements in each row or column
  SizeType nperms = sz.get<coord>();       // number of permuted rows or columns

  const unsigned unelems = to_uint(nelems);
  const unsigned unperms = to_uint(nperms);
  dim3 nr_threads(perms_kernel_sz, perms_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(unelems, perms_kernel_sz), util::ceilDiv(unperms, perms_kernel_sz));
  applyPermutationsOnDevice<typename util::internal::CppToCudaType<T>::type, coord>
      <<<nr_blocks, nr_threads, 0, stream>>>(out_begin.row(), out_begin.col(), nelems, nperms, in_offset,
                                             perms, in_layout, util::cppToCudaCast(in), out_layout,
                                             util::cppToCudaCast(out));
}

DLAF_CUDA_PERMUTE_ON_DEVICE(, float, Coord::Col);
DLAF_CUDA_PERMUTE_ON_DEVICE(, double, Coord::Col);
DLAF_CUDA_PERMUTE_ON_DEVICE(, std::complex<float>, Coord::Col);
DLAF_CUDA_PERMUTE_ON_DEVICE(, std::complex<double>, Coord::Col);

DLAF_CUDA_PERMUTE_ON_DEVICE(, float, Coord::Row);
DLAF_CUDA_PERMUTE_ON_DEVICE(, double, Coord::Row);
DLAF_CUDA_PERMUTE_ON_DEVICE(, std::complex<float>, Coord::Row);
DLAF_CUDA_PERMUTE_ON_DEVICE(, std::complex<double>, Coord::Row);

}
