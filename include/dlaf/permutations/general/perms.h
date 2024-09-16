//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include <vector>

#include <whip.hpp>

#include <dlaf/matrix/index.h>
#include <dlaf/types.h>

namespace dlaf::permutations::internal {

struct MatrixLayout {
  SizeType nb;          // square tile size
  SizeType ld;          // tile leading dimension
  SizeType row_offset;  // tile offset to first element of tile on the next row
  SizeType col_offset;  // tile offset to first element of tile on the next column
};

template <class T, Coord coord>
void applyPermutationsOnDevice(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                               const SizeType* perms, MatrixLayout in_layout, const T* in,
                               MatrixLayout out_layout, T* out, whip::stream_t stream);

#define DLAF_CUDA_PERMUTE_ON_DEVICE(kword, Type, Coord)                                              \
  kword template void applyPermutationsOnDevice<Type, Coord>(                                        \
      GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset, const SizeType* perms, \
      MatrixLayout in_layout, const Type* in, MatrixLayout out_layout, Type* out,                    \
      whip::stream_t stream)

DLAF_CUDA_PERMUTE_ON_DEVICE(extern, float, Coord::Col);
DLAF_CUDA_PERMUTE_ON_DEVICE(extern, double, Coord::Col);
DLAF_CUDA_PERMUTE_ON_DEVICE(extern, std::complex<float>, Coord::Col);
DLAF_CUDA_PERMUTE_ON_DEVICE(extern, std::complex<double>, Coord::Col);

DLAF_CUDA_PERMUTE_ON_DEVICE(extern, float, Coord::Row);
DLAF_CUDA_PERMUTE_ON_DEVICE(extern, double, Coord::Row);
DLAF_CUDA_PERMUTE_ON_DEVICE(extern, std::complex<float>, Coord::Row);
DLAF_CUDA_PERMUTE_ON_DEVICE(extern, std::complex<double>, Coord::Row);

}

#endif
