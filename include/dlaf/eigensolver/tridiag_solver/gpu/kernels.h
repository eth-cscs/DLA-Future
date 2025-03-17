//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#ifdef DLAF_WITH_GPU

#include <complex>

#include <whip.hpp>

#include <dlaf/eigensolver/tridiag_solver/coltype.h>
#include <dlaf/gpu/lapack/api.h>
#include <dlaf/types.h>

namespace dlaf::eigensolver::internal::gpu {

template <class T>
void castToComplex(const SizeType m, const SizeType n, const T* in, const SizeType ld_in,
                   std::complex<T>* out, const SizeType ld_out, whip::stream_t stream);

#define DLAF_GPU_CAST_TO_COMPLEX_ETI(kword, Type)                                       \
  kword template void castToComplex(const SizeType m, const SizeType n, const Type* in, \
                                    const SizeType ld_in, std::complex<Type>* out,      \
                                    const SizeType ld_out, whip::stream_t stream)

DLAF_GPU_CAST_TO_COMPLEX_ETI(extern, float);
DLAF_GPU_CAST_TO_COMPLEX_ETI(extern, double);

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho, const SizeType m, const SizeType n,
                                   const T* evecs, const SizeType ld_evecs, T* rank1,
                                   whip::stream_t stream);

#define DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(kword, Type)                               \
  kword template void assembleRank1UpdateVectorTile(bool is_top_tile, Type rho, const SizeType m, \
                                                    const SizeType n, const Type* evecs,          \
                                                    const SizeType ld_evecs, Type* rank1,         \
                                                    whip::stream_t stream)

DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, float);
DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, double);

template <class T>
void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s, whip::stream_t stream);

#define DLAF_GIVENS_ROT_ETI(kword, Type)                                                     \
  kword template void givensRotationOnDevice(SizeType len, Type* x, Type* y, Type c, Type s, \
                                             whip::stream_t stream)

DLAF_GIVENS_ROT_ETI(extern, float);
DLAF_GIVENS_ROT_ETI(extern, double);

}
#endif
