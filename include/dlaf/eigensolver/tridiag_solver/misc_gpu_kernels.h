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

#include "dlaf/eigensolver/tridiag_solver/coltype.h"
#include "dlaf/types.h"

namespace dlaf::eigensolver::internal {

template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr);

#define DLAF_CUDA_MERGE_INDICES_ETI(kword, Type)                                                 \
  kword template void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, \
                                           const SizeType* end_ptr, SizeType* out_ptr,           \
                                           const Type* v_ptr)

DLAF_CUDA_MERGE_INDICES_ETI(extern, float);
DLAF_CUDA_MERGE_INDICES_ETI(extern, double);

template <class T>
void applyIndexOnDevice(SizeType len, const SizeType* index, const T* in, T* out, cudaStream_t stream);

#define DLAF_CUDA_APPLY_INDEX_ETI(kword, Type)                                                \
  kword template void applyIndexOnDevice(SizeType len, const SizeType* index, const Type* in, \
                                         Type* out, cudaStream_t stream)

DLAF_CUDA_APPLY_INDEX_ETI(extern, float);
DLAF_CUDA_APPLY_INDEX_ETI(extern, double);

template <class T>
void castTileToComplex(SizeType m, SizeType n, SizeType ld, const T* in, std::complex<T>* out,
                       cudaStream_t stream);

#define DLAF_CUDA_CAST_TO_COMPLEX(kword, Type)                                               \
  kword template void castTileToComplex(SizeType m, SizeType n, SizeType ld, const Type* in, \
                                        std::complex<Type>* out, cudaStream_t stream)

DLAF_CUDA_CAST_TO_COMPLEX(extern, float);
DLAF_CUDA_CAST_TO_COMPLEX(extern, double);

void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out, cudaStream_t stream);

void initIndexTile(SizeType offset, SizeType len, SizeType* index_arr, cudaStream_t stream);

void setColTypeTile(ColType ct, SizeType len, ColType* ct_arr, cudaStream_t stream);

template <class T>
void copyTileRowAndNormalizeOnDevice(int sign, SizeType len, SizeType tile_ld, const T* tile, T* col,
                                     cudaStream_t stream);

#define DLAF_COPY_TILE_ROW_ETI(kword, Type)                                                     \
  kword template void copyTileRowAndNormalizeOnDevice(int sign, SizeType len, SizeType tile_ld, \
                                                      const Type* tile, Type* col, cudaStream_t stream)

DLAF_COPY_TILE_ROW_ETI(extern, float);
DLAF_COPY_TILE_ROW_ETI(extern, double);

template <class T>
void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s, cudaStream_t stream);

#define DLAF_GIVENS_ROT_ETI(kword, Type)                                                     \
  kword template void givensRotationOnDevice(SizeType len, Type* x, Type* y, Type c, Type s, \
                                             cudaStream_t stream)

DLAF_GIVENS_ROT_ETI(extern, float);
DLAF_GIVENS_ROT_ETI(extern, double);

}

#endif
