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

#include <whip.hpp>

#include <dlaf/gpu/lapack/api.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal::gpu {

template <class T>
void updatePointers(const std::size_t n, T** a, const SizeType ld, whip::stream_t stream);

#define DLAF_GPU_UPDATE_POINTERS_ETI(kword, Type)                                      \
  kword template void updatePointers(const std::size_t n, Type** b, const SizeType ld, \
                                     whip::stream_t stream)

DLAF_GPU_UPDATE_POINTERS_ETI(extern, float);
DLAF_GPU_UPDATE_POINTERS_ETI(extern, double);
DLAF_GPU_UPDATE_POINTERS_ETI(extern, std::complex<float>);
DLAF_GPU_UPDATE_POINTERS_ETI(extern, std::complex<double>);

}

#endif
