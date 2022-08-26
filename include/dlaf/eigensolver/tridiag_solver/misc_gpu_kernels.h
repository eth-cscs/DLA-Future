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

#include <cusolverDn.h>

namespace dlaf::eigensolver::internal {

// Returns the maximum element in the array
template <class T>
T maxElementOnDevice(SizeType len, const T* arr, cudaStream_t stream);

#define DLAF_CUDA_MAX_ELEMENT_ETI(kword, Type) \
  kword template Type maxElementOnDevice(SizeType len, const Type* arr, cudaStream_t stream)

DLAF_CUDA_MAX_ELEMENT_ETI(extern, float);
DLAF_CUDA_MAX_ELEMENT_ETI(extern, double);

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
void syevdTile(cusolverDnHandle_t handle, SizeType n, T* evals, const T* offdiag, SizeType ld_evecs,
               T* evecs);

#define DLAF_CUSOLVER_SYEVC_ETI(kword, Type)                                        \
  kword template void syevdTile(cusolverDnHandle_t handle, SizeType n, Type* evals, \
                                const Type* offdiag, SizeType ld_evecs, Type* evecs)

DLAF_CUSOLVER_SYEVC_ETI(extern, float);
DLAF_CUSOLVER_SYEVC_ETI(extern, double);

template <class T>
T cuppensDecompOnDevice(const T* d_offdiag_val, T* d_top_diag_val, T* d_bottom_diag_val,
                        cudaStream_t stream);

#define DLAF_CUDA_CUPPENS_DECOMP_ETI(kword, Type)                                            \
  kword template Type cuppensDecompOnDevice(const Type* d_offdiag_val, Type* d_top_diag_val, \
                                            Type* d_bottom_diag_val, cudaStream_t stream)

DLAF_CUDA_CUPPENS_DECOMP_ETI(extern, float);
DLAF_CUDA_CUPPENS_DECOMP_ETI(extern, double);

template <class T>
void updateEigenvectorsWithDiagonal(SizeType nrows, SizeType ncols, SizeType ld, const T* d_rows,
                                    const T* d_cols, const T* evecs, T* ws, cudaStream_t stream);

#define DLAF_CUDA_UPDATE_EVECS_WITH_DIAG_ETI(kword, Type)                                         \
  kword template void updateEigenvectorsWithDiagonal(SizeType nrows, SizeType ncols, SizeType ld, \
                                                     const Type* d_rows, const Type* d_cols,      \
                                                     const Type* evecs, Type* ws, cudaStream_t stream)

DLAF_CUDA_UPDATE_EVECS_WITH_DIAG_ETI(extern, float);
DLAF_CUDA_UPDATE_EVECS_WITH_DIAG_ETI(extern, double);

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
void scaleTileWithRow(SizeType nrows, SizeType ncols, SizeType ld, const T* norms, T* evecs,
                      cudaStream_t stream);

#define DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(kword, Type)                                                 \
  kword template void scaleTileWithRow(SizeType nrows, SizeType ncols, SizeType ld, const Type* norms, \
                                       Type* evecs, cudaStream_t stream)

DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(extern, float);
DLAF_CUDA_SCALE_TILE_WITH_ROW_ETI(extern, double);

}

#endif
