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

#include "dlaf/common/index2d.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

namespace dlaf::permutations::internal {

template <Coord C>
void invertAndCopyArr(const matrix::Distribution& dist, const SizeType* in_ptr, SizeType* out_ptr);

#define DLAF_CPU_INVERT_AND_COPY_ETI(kword, Type)                                                      \
  kword template void invertAndCopyArr<Type>(const matrix::Distribution& dist, const SizeType* in_ptr, \
                                             SizeType* out_ptr)

DLAF_CPU_INVERT_AND_COPY_ETI(extern, Coord::Row);
DLAF_CPU_INVERT_AND_COPY_ETI(extern, Coord::Col);

#ifdef DLAF_WITH_GPU
template <Coord C>
void invertAndCopyArr(const matrix::Distribution& dist, const SizeType* in_ptr, SizeType* out_ptr,
                      whip::stream_t stream);

#define DLAF_GPU_INVERT_AND_COPY_ETI(kword, Type)                                                      \
  kword template void invertAndCopyArr<Type>(const matrix::Distribution& dist, const SizeType* in_ptr, \
                                             SizeType* out_ptr, whip::stream_t stream)

DLAF_GPU_INVERT_AND_COPY_ETI(extern, Coord::Row);
DLAF_GPU_INVERT_AND_COPY_ETI(extern, Coord::Col);
#endif

}
