//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

#include <whip.hpp>

#include <vector>

namespace dlaf::permutations::internal {

template <class T, Coord coord>
void applyPermutationsOnDevice(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                               const matrix::Distribution& distr, const SizeType* perms,
                               const std::vector<matrix::Tile<T, Device::GPU>>& in_tiles,
                               const std::vector<matrix::Tile<T, Device::GPU>>& out_tiles,
                               whip::stream_t stream);

#define DLAF_CUDA_PERMUTE_ON_DEVICE(kword, Type, Coord)                                                 \
  kword template void                                                                                   \
  applyPermutationsOnDevice<Type, Coord>(GlobalElementIndex out_begin, GlobalElementSize sz,            \
                                         SizeType in_offset, const matrix::Distribution& distr,         \
                                         const SizeType* perms,                                         \
                                         const std::vector<matrix::Tile<Type, Device::GPU>>& in_tiles,  \
                                         const std::vector<matrix::Tile<Type, Device::GPU>>& out_tiles, \
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
