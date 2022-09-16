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

#include "dlaf/common/callable_object.h"
#include "dlaf/eigensolver/tridiag_solver/coltype.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_GPU
#include <cusolverDn.h>
#endif

namespace dlaf::eigensolver::internal {

// Cuppen's decomposition
//
// Substracts the offdiagonal element at the split from the top and bottom diagonal elements and
// returns the offdiagonal element. The split is between the last row of the top tile and the first
// row of the bottom tile.
//
template <class T>
T cuppensDecomp(const matrix::Tile<T, Device::CPU>& top, const matrix::Tile<T, Device::CPU>& bottom);

#define DLAF_CPU_CUPPENS_DECOMP_ETI(kword, Type)                                \
  kword template Type cuppensDecomp(const matrix::Tile<Type, Device::CPU>& top, \
                                    const matrix::Tile<Type, Device::CPU>& bottom)

DLAF_CPU_CUPPENS_DECOMP_ETI(extern, float);
DLAF_CPU_CUPPENS_DECOMP_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
T cuppensDecomp(const matrix::Tile<T, Device::GPU>& top, const matrix::Tile<T, Device::GPU>& bottom,
                cudaStream_t stream);

#define DLAF_GPU_CUPPENS_DECOMP_ETI(kword, Type)                                \
  kword template Type cuppensDecomp(const matrix::Tile<Type, Device::GPU>& top, \
                                    const matrix::Tile<Type, Device::GPU>& bottom, cudaStream_t stream)

DLAF_GPU_CUPPENS_DECOMP_ETI(extern, float);
DLAF_GPU_CUPPENS_DECOMP_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(cuppensDecomp);

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::CPU>& diag_tile);

#define DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(kword, Type)                        \
  kword template void                                                                           \
  copyDiagonalFromCompactTridiagonal(const matrix::Tile<const Type, Device::CPU>& tridiag_tile, \
                                     const matrix::Tile<Type, Device::CPU>& diag_tile)

DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, float);
DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::GPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::GPU>& diag_tile,
                                        cudaStream_t stream);

#define DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(kword, Type)                        \
  kword template void                                                                           \
  copyDiagonalFromCompactTridiagonal(const matrix::Tile<const Type, Device::GPU>& tridiag_tile, \
                                     const matrix::Tile<Type, Device::GPU>& diag_tile,          \
                                     cudaStream_t stream)

DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, float);
DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(copyDiagonalFromCompactTridiagonal);

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::CPU>& evecs_tile,
                                   const matrix::Tile<T, Device::CPU>& rank1_tile);

#define DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(kword, Type)                      \
  kword template void                                                                    \
  assembleRank1UpdateVectorTile(bool is_top_tile, Type rho,                              \
                                const matrix::Tile<const Type, Device::CPU>& evecs_tile, \
                                const matrix::Tile<Type, Device::CPU>& rank1_tile)

DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, float);
DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::GPU>& evecs_tile,
                                   const matrix::Tile<T, Device::GPU>& rank1_tile, cudaStream_t stream);

#define DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(kword, Type)                      \
  kword template void                                                                    \
  assembleRank1UpdateVectorTile(bool is_top_tile, Type rho,                              \
                                const matrix::Tile<const Type, Device::GPU>& evecs_tile, \
                                const matrix::Tile<Type, Device::GPU>& rank1_tile, cudaStream_t stream)

DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, float);
DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(assembleRank1UpdateVectorTile);

template <class T>
T maxElementInColumnTile(const matrix::Tile<const T, Device::CPU>& tile);

#define DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(kword, Type) \
  kword template Type maxElementInColumnTile(const matrix::Tile<const Type, Device::CPU>& tile)

DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, float);
DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
T maxElementInColumnTile(const matrix::Tile<const T, Device::GPU>& tile, cudaStream_t stream);

#define DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(kword, Type)                                    \
  kword template Type maxElementInColumnTile(const matrix::Tile<const Type, Device::GPU>& tile, \
                                             cudaStream_t stream)

DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, float);
DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(maxElementInColumnTile);

void setColTypeTile(const ColType& ct, const matrix::Tile<ColType, Device::CPU>& tile);

#ifdef DLAF_WITH_GPU
void setColTypeTile(const ColType& ct, const matrix::Tile<ColType, Device::GPU>& tile,
                    cudaStream_t stream);
#endif

DLAF_MAKE_CALLABLE_OBJECT(setColTypeTile);

void initIndexTile(SizeType offset, const matrix::Tile<SizeType, Device::CPU>& tile);

#ifdef DLAF_WITH_GPU
void initIndexTile(SizeType offset, const matrix::Tile<SizeType, Device::GPU>& tile,
                   cudaStream_t stream);
#endif

DLAF_MAKE_CALLABLE_OBJECT(initIndexTile);

template <class T>
void divideEvecsByDiagonal(const SizeType& k, const SizeType& i_subm_el, const SizeType& j_subm_el,
                           const matrix::Tile<const T, Device::CPU>& diag_rows,
                           const matrix::Tile<const T, Device::CPU>& diag_cols,
                           const matrix::Tile<const T, Device::CPU>& evecs_tile,
                           const matrix::Tile<T, Device::CPU>& ws_tile);

#define DLAF_CPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(kword, Type)                                           \
  kword template void divideEvecsByDiagonal(const SizeType& k, const SizeType& i_subm_el,            \
                                            const SizeType& j_subm_el,                               \
                                            const matrix::Tile<const Type, Device::CPU>& diag_rows,  \
                                            const matrix::Tile<const Type, Device::CPU>& diag_cols,  \
                                            const matrix::Tile<const Type, Device::CPU>& evecs_tile, \
                                            const matrix::Tile<Type, Device::CPU>& ws_tile)

DLAF_CPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(extern, float);
DLAF_CPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void divideEvecsByDiagonal(const SizeType& k, const SizeType& i_subm_el, const SizeType& j_subm_el,
                           const matrix::Tile<const T, Device::GPU>& diag_rows,
                           const matrix::Tile<const T, Device::GPU>& diag_cols,
                           const matrix::Tile<const T, Device::GPU>& evecs_tile,
                           const matrix::Tile<T, Device::GPU>& ws_tile, cudaStream_t stream);

#define DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(kword, Type)                                           \
  kword template void divideEvecsByDiagonal(const SizeType& k, const SizeType& i_subm_el,            \
                                            const SizeType& j_subm_el,                               \
                                            const matrix::Tile<const Type, Device::GPU>& diag_rows,  \
                                            const matrix::Tile<const Type, Device::GPU>& diag_cols,  \
                                            const matrix::Tile<const Type, Device::GPU>& evecs_tile, \
                                            const matrix::Tile<Type, Device::GPU>& ws_tile,          \
                                            cudaStream_t stream)

DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(extern, float);
DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(divideEvecsByDiagonal);

// ---------------------------

#ifdef DLAF_WITH_GPU

// Returns the number of non-deflated entries
SizeType stablePartitionIndexOnDevice(SizeType n, const ColType* c_ptr, const SizeType* in_ptr,
                                      SizeType* out_ptr, cudaStream_t stream);

template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr, cudaStream_t stream);

#define DLAF_CUDA_MERGE_INDICES_ETI(kword, Type)                                                 \
  kword template void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, \
                                           const SizeType* end_ptr, SizeType* out_ptr,           \
                                           const Type* v_ptr, cudaStream_t stream)

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

template <class T>
void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s, cudaStream_t stream);

#define DLAF_GIVENS_ROT_ETI(kword, Type)                                                     \
  kword template void givensRotationOnDevice(SizeType len, Type* x, Type* y, Type c, Type s, \
                                             cudaStream_t stream)

DLAF_GIVENS_ROT_ETI(extern, float);
DLAF_GIVENS_ROT_ETI(extern, double);

template <class T>
void setUnitDiagTileOnDevice(SizeType len, SizeType ld, T* tile, cudaStream_t stream);

#define DLAF_SET_UNIT_DIAG_ETI(kword, Type) \
  kword template void setUnitDiagTileOnDevice(SizeType len, SizeType ld, Type* tile, cudaStream_t stream)

DLAF_SET_UNIT_DIAG_ETI(extern, float);
DLAF_SET_UNIT_DIAG_ETI(extern, double);

template <class T>
void copyDiagTileFromTridiagTile(SizeType len, const T* tridiag, T* diag, cudaStream_t stream);

#define DLAF_COPY_DIAG_TILE_ETI(kword, Type)                                                     \
  kword template void copyDiagTileFromTridiagTile(SizeType len, const Type* tridiag, Type* diag, \
                                                  cudaStream_t stream)

DLAF_COPY_DIAG_TILE_ETI(extern, float);
DLAF_COPY_DIAG_TILE_ETI(extern, double);

template <class T>
void multiplyColumns(SizeType len, const T* in, T* out, cudaStream_t stream);

#define DLAF_CUDA_MULTIPLY_COLS_ETI(kword, Type) \
  kword template void multiplyColumns(SizeType len, const Type* in, Type* out, cudaStream_t stream)

DLAF_CUDA_MULTIPLY_COLS_ETI(extern, float);
DLAF_CUDA_MULTIPLY_COLS_ETI(extern, double);

template <class T>
void calcEvecsFromWeightVec(SizeType nrows, SizeType ncols, SizeType ld, const T* rank1_vec,
                            const T* weight_vec, T* evecs, cudaStream_t stream);

#define DLAF_CUDA_EVECS_FROM_WEIGHT_VEC_ETI(kword, Type)                                    \
  kword template void calcEvecsFromWeightVec(SizeType nrows, SizeType ncols, SizeType ld,   \
                                             const Type* rank1_vec, const Type* weight_vec, \
                                             Type* evecs, cudaStream_t stream)

DLAF_CUDA_EVECS_FROM_WEIGHT_VEC_ETI(extern, float);
DLAF_CUDA_EVECS_FROM_WEIGHT_VEC_ETI(extern, double);

template <class T>
void sumSqTileOnDevice(SizeType nrows, SizeType ncols, SizeType ld, const T* in, T* out,
                       cudaStream_t stream);

#define DLAF_CUDA_SUM_SQ_TILE_ETI(kword, Type)                                                       \
  kword template void sumSqTileOnDevice(SizeType nrows, SizeType ncols, SizeType ld, const Type* in, \
                                        Type* out, cudaStream_t stream)

DLAF_CUDA_SUM_SQ_TILE_ETI(extern, float);
DLAF_CUDA_SUM_SQ_TILE_ETI(extern, double);

template <class T>
void addFirstRows(SizeType len, SizeType ld, const T* in, T* out, cudaStream_t stream);

#define DLAF_CUDA_ADD_FIRST_ROWS_ETI(kword, Type)                                        \
  kword template void addFirstRows(SizeType len, SizeType ld, const Type* in, Type* out, \
                                   cudaStream_t stream)

DLAF_CUDA_ADD_FIRST_ROWS_ETI(extern, float);
DLAF_CUDA_ADD_FIRST_ROWS_ETI(extern, double);

template <class T>
void scaleTileWithRow(SizeType nrows, SizeType ncols, SizeType ld_norms, const T* norms,
                      SizeType ld_evecs, T* evecs, cudaStream_t stream);

#define DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(kword, Type)                                    \
  kword template void scaleTileWithRow(SizeType nrows, SizeType ncols, SizeType ld_norms, \
                                       const Type* norms, SizeType ld_evecs, Type* evecs, \
                                       cudaStream_t stream)

DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(extern, float);
DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(extern, double);

#endif

}
