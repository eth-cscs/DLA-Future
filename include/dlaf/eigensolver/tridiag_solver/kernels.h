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
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_chunk.h"
#include "dlaf/sender/transform.h"
#include "dlaf/types.h"

#ifdef DLAF_WITH_GPU
#include <whip.hpp>

#include "dlaf/gpu/lapack/api.h"
#endif

namespace dlaf::eigensolver::internal {

template <Device D, class DiagTileSenderIn, class DiagTileSenderOut>
void stedcAsync(DiagTileSenderIn&& in, DiagTileSenderOut&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto sender = ex::when_all(std::forward<DiagTileSenderIn>(in), std::forward<DiagTileSenderOut>(out));
  ex::start_detached(tile::stedc(di::Policy<DefaultBackend_v<D>>(), std::move(sender)));
}

template <class T>
void castToComplex(const matrix::Tile<const T, Device::CPU>& in,
                   const matrix::Tile<std::complex<T>, Device::CPU>& out);

#define DLAF_CPU_CAST_TO_COMPLEX_ETI(kword, Type)                                    \
  kword template void castToComplex(const matrix::Tile<const Type, Device::CPU>& in, \
                                    const matrix::Tile<std::complex<Type>, Device::CPU>& out)

DLAF_CPU_CAST_TO_COMPLEX_ETI(extern, float);
DLAF_CPU_CAST_TO_COMPLEX_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void castToComplex(const matrix::Tile<const T, Device::GPU>& in,
                   const matrix::Tile<std::complex<T>, Device::GPU>& out, whip::stream_t stream);

#define DLAF_GPU_CAST_TO_COMPLEX_ETI(kword, Type)                                             \
  kword template void castToComplex(const matrix::Tile<const Type, Device::GPU>& in,          \
                                    const matrix::Tile<std::complex<Type>, Device::GPU>& out, \
                                    whip::stream_t stream)

DLAF_GPU_CAST_TO_COMPLEX_ETI(extern, float);
DLAF_GPU_CAST_TO_COMPLEX_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(castToComplex);

template <Device D, class InTileSender, class OutTileSender>
void castToComplexAsync(InTileSender&& in, OutTileSender&& out) {
  namespace di = dlaf::internal;
  namespace ex = pika::execution::experimental;
  auto sender = ex::when_all(std::forward<InTileSender>(in), std::forward<OutTileSender>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), castToComplex_o, std::move(sender));
}

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
void cuppensDecomp(const matrix::Tile<T, Device::GPU>& top, const matrix::Tile<T, Device::GPU>& bottom,
                   T* host_offdiag_val_ptr, whip::stream_t stream);

#define DLAF_GPU_CUPPENS_DECOMP_ETI(kword, Type)                                   \
  kword template void cuppensDecomp(const matrix::Tile<Type, Device::GPU>& top,    \
                                    const matrix::Tile<Type, Device::GPU>& bottom, \
                                    Type* host_offdiag_val_ptr, whip::stream_t stream)

DLAF_GPU_CUPPENS_DECOMP_ETI(extern, float);
DLAF_GPU_CUPPENS_DECOMP_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(cuppensDecomp);

template <class T, Device D, class TopTileSender, class BottomTileSender>
auto cuppensDecompAsync(TopTileSender&& top, BottomTileSender&& bottom) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  using ElementType = dlaf::internal::SenderElementType<TopTileSender>;
  constexpr auto backend = dlaf::DefaultBackend_v<D>;

  if constexpr (backend == dlaf::Backend::GPU) {
    return ex::when_all(std::forward<TopTileSender>(top), std::forward<BottomTileSender>(bottom),
                        ex::just(memory::MemoryChunk<ElementType, Device::CPU>{1})) |
           ex::let_value([](auto& top, auto& bottom, auto& host_offdiag_val) {
             return ex::just(std::ref(top), std::ref(bottom), host_offdiag_val()) |
                    di::transform(di::Policy<backend>(), cuppensDecomp_o) |
                    ex::then([&host_offdiag_val]() { return *host_offdiag_val(); });
           });
  }
  else {
    return ex::when_all(std::forward<TopTileSender>(top), std::forward<BottomTileSender>(bottom)) |
           di::transform(di::Policy<backend>(), cuppensDecomp_o);
  }
}

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
                                        whip::stream_t stream);

#define DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(kword, Type)                        \
  kword template void                                                                           \
  copyDiagonalFromCompactTridiagonal(const matrix::Tile<const Type, Device::GPU>& tridiag_tile, \
                                     const matrix::Tile<Type, Device::GPU>& diag_tile,          \
                                     whip::stream_t stream)

DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, float);
DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(copyDiagonalFromCompactTridiagonal);

template <Device D, class TridiagTile, class DiagTile>
void copyDiagonalFromCompactTridiagonalAsync(TridiagTile&& in, DiagTile&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto sender = ex::when_all(std::forward<TridiagTile>(in), std::forward<DiagTile>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), copyDiagonalFromCompactTridiagonal_o,
                      std::move(sender));
}

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
                                   const matrix::Tile<T, Device::GPU>& rank1_tile,
                                   whip::stream_t stream);

#define DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(kword, Type)                      \
  kword template void                                                                    \
  assembleRank1UpdateVectorTile(bool is_top_tile, Type rho,                              \
                                const matrix::Tile<const Type, Device::GPU>& evecs_tile, \
                                const matrix::Tile<Type, Device::GPU>& rank1_tile,       \
                                whip::stream_t stream)

DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, float);
DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(assembleRank1UpdateVectorTile);

template <class T, Device D, class RhoSender, class EvecsTileSender, class Rank1TileSender>
void assembleRank1UpdateVectorTileAsync(bool top_tile, RhoSender&& rho, EvecsTileSender&& evecs,
                                        Rank1TileSender&& rank1) {
  namespace di = dlaf::internal;
  auto sender =
      di::whenAllLift(top_tile, std::forward<RhoSender>(rho), std::forward<EvecsTileSender>(evecs),
                      std::forward<Rank1TileSender>(rank1));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), assembleRank1UpdateVectorTile_o,
                      std::move(sender));
}

template <class T>
T maxElementInColumnTile(const matrix::Tile<const T, Device::CPU>& tile);

#define DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(kword, Type) \
  kword template Type maxElementInColumnTile(const matrix::Tile<const Type, Device::CPU>& tile)

DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, float);
DLAF_CPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
void maxElementInColumnTile(const matrix::Tile<const T, Device::GPU>& tile, T* host_max_el_ptr,
                            T* device_max_el_ptr, whip::stream_t stream);

#define DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(kword, Type)                                    \
  kword template void maxElementInColumnTile(const matrix::Tile<const Type, Device::GPU>& tile, \
                                             Type* host_max_el_ptr, Type* device_max_el_ptr,    \
                                             whip::stream_t stream)

DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, float);
DLAF_GPU_MAX_ELEMENT_IN_COLUMN_TILE_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(maxElementInColumnTile);

template <class T, Device D, class TileSender>
auto maxElementInColumnTileAsync(TileSender&& tile) {
  namespace di = dlaf::internal;
  namespace ex = pika::execution::experimental;

  using ElementType = dlaf::internal::SenderElementType<TileSender>;
  constexpr auto backend = dlaf::DefaultBackend_v<D>;

  if constexpr (backend == dlaf::Backend::GPU) {
    return ex::when_all(std::forward<TileSender>(tile),
                        ex::just(memory::MemoryChunk<ElementType, Device::CPU>{1},
                                 memory::MemoryChunk<ElementType, Device::GPU>{1})) |
           ex::let_value([](auto& tile, auto& host_max_el, auto& device_max_el) {
             DLAF_ASSERT(tile.is_ready(), "");
             return ex::just(tile, host_max_el(), device_max_el()) |
                    di::transform(di::Policy<backend>(), maxElementInColumnTile_o) |
                    ex::then([&host_max_el]() { return *host_max_el(); });
           });
  }
  else {
    return std::forward<TileSender>(tile) |
           di::transform(di::Policy<backend>(), maxElementInColumnTile_o);
  }
}

void setColTypeTile(const ColType& ct, const matrix::Tile<ColType, Device::CPU>& tile);

#ifdef DLAF_WITH_GPU
void setColTypeTile(const ColType& ct, const matrix::Tile<ColType, Device::GPU>& tile,
                    whip::stream_t stream);
#endif

DLAF_MAKE_CALLABLE_OBJECT(setColTypeTile);

template <Device D, class TileSender>
void setColTypeTileAsync(ColType val, TileSender&& tile) {
  namespace di = dlaf::internal;

  auto sender = di::whenAllLift(val, std::forward<TileSender>(tile));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), setColTypeTile_o, std::move(sender));
}

void initIndexTile(SizeType offset, const matrix::Tile<SizeType, Device::CPU>& tile);

#ifdef DLAF_WITH_GPU
void initIndexTile(SizeType offset, const matrix::Tile<SizeType, Device::GPU>& tile,
                   whip::stream_t stream);
#endif

DLAF_MAKE_CALLABLE_OBJECT(initIndexTile);

template <Device D, class TileSender>
void initIndexTileAsync(SizeType tile_row, TileSender&& tile) {
  namespace di = dlaf::internal;

  auto sender = di::whenAllLift(tile_row, std::forward<TileSender>(tile));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), initIndexTile_o, std::move(sender));
}

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
                           const matrix::Tile<T, Device::GPU>& ws_tile, whip::stream_t stream);

#define DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(kword, Type)                                           \
  kword template void divideEvecsByDiagonal(const SizeType& k, const SizeType& i_subm_el,            \
                                            const SizeType& j_subm_el,                               \
                                            const matrix::Tile<const Type, Device::GPU>& diag_rows,  \
                                            const matrix::Tile<const Type, Device::GPU>& diag_cols,  \
                                            const matrix::Tile<const Type, Device::GPU>& evecs_tile, \
                                            const matrix::Tile<Type, Device::GPU>& ws_tile,          \
                                            whip::stream_t stream)

DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(extern, float);
DLAF_GPU_DIVIDE_EVECS_BY_DIAGONAL_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(divideEvecsByDiagonal);

template <Device D, class KSender, class DiagRowsTileSender, class DiagColsTileSender,
          class EvecsTileSender, class TempTileSender>
void divideEvecsByDiagonalAsync(KSender&& k, SizeType i_subm_el, SizeType j_subm_el,
                                DiagRowsTileSender&& diag_rows, DiagColsTileSender&& diag_cols,
                                EvecsTileSender&& evecs, TempTileSender&& temp) {
  namespace di = dlaf::internal;
  auto sender =
      di::whenAllLift(std::forward<KSender>(k), i_subm_el, j_subm_el,
                      std::forward<DiagRowsTileSender>(diag_rows),
                      std::forward<DiagColsTileSender>(diag_cols), std::forward<EvecsTileSender>(evecs),
                      std::forward<TempTileSender>(temp));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), divideEvecsByDiagonal_o, std::move(sender));
}

template <class T>
void multiplyFirstColumns(const SizeType& k, const SizeType& row, const SizeType& col,
                          const matrix::Tile<const T, Device::CPU>& in,
                          const matrix::Tile<T, Device::CPU>& out);

#define DLAF_CPU_MULTIPLY_FIRST_COLUMNS_ETI(kword, Type)                                                \
  kword template void multiplyFirstColumns(const SizeType& k, const SizeType& row, const SizeType& col, \
                                           const matrix::Tile<const Type, Device::CPU>& in,             \
                                           const matrix::Tile<Type, Device::CPU>& out)

DLAF_CPU_MULTIPLY_FIRST_COLUMNS_ETI(extern, float);
DLAF_CPU_MULTIPLY_FIRST_COLUMNS_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void multiplyFirstColumns(const SizeType& k, const SizeType& row, const SizeType& col,
                          const matrix::Tile<const T, Device::GPU>& in,
                          const matrix::Tile<T, Device::GPU>& out, whip::stream_t stream);

#define DLAF_GPU_MULTIPLY_FIRST_COLUMNS_ETI(kword, Type)                                                \
  kword template void multiplyFirstColumns(const SizeType& k, const SizeType& row, const SizeType& col, \
                                           const matrix::Tile<const Type, Device::GPU>& in,             \
                                           const matrix::Tile<Type, Device::GPU>& out,                  \
                                           whip::stream_t stream)

DLAF_GPU_MULTIPLY_FIRST_COLUMNS_ETI(extern, float);
DLAF_GPU_MULTIPLY_FIRST_COLUMNS_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(multiplyFirstColumns);

template <Device D, class KSender, class InTileSender, class OutTileSender>
void multiplyFirstColumnsAsync(KSender&& k, SizeType row, SizeType col, InTileSender&& in,
                               OutTileSender&& out) {
  namespace di = dlaf::internal;
  auto sender = di::whenAllLift(std::forward<KSender>(k), row, col, std::forward<InTileSender>(in),
                                std::forward<OutTileSender>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), multiplyFirstColumns_o, std::move(sender));
}

template <class T>
void calcEvecsFromWeightVec(const SizeType& k, const SizeType& row, const SizeType& col,
                            const matrix::Tile<const T, Device::CPU>& z_tile,
                            const matrix::Tile<const T, Device::CPU>& ws_tile,
                            const matrix::Tile<T, Device::CPU>& evecs_tile);

#define DLAF_CPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(kword, Type)                                       \
  kword template void calcEvecsFromWeightVec(const SizeType& k, const SizeType& row,               \
                                             const SizeType& col,                                  \
                                             const matrix::Tile<const Type, Device::CPU>& z_tile,  \
                                             const matrix::Tile<const Type, Device::CPU>& ws_tile, \
                                             const matrix::Tile<Type, Device::CPU>& evecs_tile)

DLAF_CPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(extern, float);
DLAF_CPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void calcEvecsFromWeightVec(const SizeType& k, const SizeType& row, const SizeType& col,
                            const matrix::Tile<const T, Device::GPU>& z_tile,
                            const matrix::Tile<const T, Device::GPU>& ws_tile,
                            const matrix::Tile<T, Device::GPU>& evecs_tile, whip::stream_t stream);

#define DLAF_GPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(kword, Type)                                       \
  kword template void calcEvecsFromWeightVec(const SizeType& k, const SizeType& row,               \
                                             const SizeType& col,                                  \
                                             const matrix::Tile<const Type, Device::GPU>& z_tile,  \
                                             const matrix::Tile<const Type, Device::GPU>& ws_tile, \
                                             const matrix::Tile<Type, Device::GPU>& evecs_tile,    \
                                             whip::stream_t stream)

DLAF_GPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(extern, float);
DLAF_GPU_CALC_EVECS_FROM_WEIGHT_VEC_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(calcEvecsFromWeightVec);

template <Device D, class KSender, class Rank1TileSender, class TempTileSender, class EvecsTileSender>
void calcEvecsFromWeightVecAsync(KSender&& k, SizeType row, SizeType col, Rank1TileSender&& rank1,
                                 TempTileSender&& temp, EvecsTileSender&& evecs) {
  namespace di = dlaf::internal;

  auto sender =
      di::whenAllLift(std::forward<KSender>(k), row, col, std::forward<Rank1TileSender>(rank1),
                      std::forward<TempTileSender>(temp), std::forward<EvecsTileSender>(evecs));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), calcEvecsFromWeightVec_o, std::move(sender));
}

template <class T>
void sumsqCols(const SizeType& k, const SizeType& row, const SizeType& col,
               const matrix::Tile<const T, Device::CPU>& evecs_tile,
               const matrix::Tile<T, Device::CPU>& ws_tile);

#define DLAF_CPU_SUMSQ_COLS_ETI(kword, Type)                                                 \
  kword template void sumsqCols(const SizeType& k, const SizeType& row, const SizeType& col, \
                                const matrix::Tile<const Type, Device::CPU>& evecs_tile,     \
                                const matrix::Tile<Type, Device::CPU>& ws_tile)

DLAF_CPU_SUMSQ_COLS_ETI(extern, float);
DLAF_CPU_SUMSQ_COLS_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void sumsqCols(const SizeType& k, const SizeType& row, const SizeType& col,
               const matrix::Tile<const T, Device::GPU>& evecs_tile,
               const matrix::Tile<T, Device::GPU>& ws_tile, whip::stream_t stream);

#define DLAF_GPU_SUMSQ_COLS_ETI(kword, Type)                                                 \
  kword template void sumsqCols(const SizeType& k, const SizeType& row, const SizeType& col, \
                                const matrix::Tile<const Type, Device::GPU>& evecs_tile,     \
                                const matrix::Tile<Type, Device::GPU>& ws_tile, whip::stream_t stream)

DLAF_GPU_SUMSQ_COLS_ETI(extern, float);
DLAF_GPU_SUMSQ_COLS_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(sumsqCols);

template <Device D, class KSender, class EvecsTileSender, class TempTileSender>
void sumsqColsAsync(KSender&& k, SizeType row, SizeType col, EvecsTileSender&& evecs,
                    TempTileSender&& temp) {
  namespace di = dlaf::internal;

  auto sender = di::whenAllLift(std::forward<KSender>(k), row, col, std::forward<EvecsTileSender>(evecs),
                                std::forward<TempTileSender>(temp));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), sumsqCols_o, std::move(sender));
}

template <class T>
void addFirstRows(const SizeType& k, const SizeType& row, const SizeType& col,
                  const matrix::Tile<const T, Device::CPU>& in, const matrix::Tile<T, Device::CPU>& out);

#define DLAF_CPU_ADD_FIRST_ROWS_ETI(kword, Type)                                                \
  kword template void addFirstRows(const SizeType& k, const SizeType& row, const SizeType& col, \
                                   const matrix::Tile<const Type, Device::CPU>& in,             \
                                   const matrix::Tile<Type, Device::CPU>& out)

DLAF_CPU_ADD_FIRST_ROWS_ETI(extern, float);
DLAF_CPU_ADD_FIRST_ROWS_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void addFirstRows(const SizeType& k, const SizeType& row, const SizeType& col,
                  const matrix::Tile<const T, Device::GPU>& evecs_tile,
                  const matrix::Tile<T, Device::GPU>& ws_tile, whip::stream_t stream);

#define DLAF_GPU_ADD_FIRST_ROWS_ETI(kword, Type)                                                \
  kword template void addFirstRows(const SizeType& k, const SizeType& row, const SizeType& col, \
                                   const matrix::Tile<const Type, Device::GPU>& in,             \
                                   const matrix::Tile<Type, Device::GPU>& out, whip::stream_t stream)

DLAF_GPU_ADD_FIRST_ROWS_ETI(extern, float);
DLAF_GPU_ADD_FIRST_ROWS_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(addFirstRows);

template <Device D, class KSender, class InTileSender, class OutTileSender>
void addFirstRowsAsync(KSender&& k, SizeType row, SizeType col, InTileSender&& in, OutTileSender&& out) {
  namespace di = dlaf::internal;

  auto sender = di::whenAllLift(std::forward<KSender>(k), row, col, std::forward<InTileSender>(in),
                                std::forward<OutTileSender>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), addFirstRows_o, std::move(sender));
}

template <class T>
void divideColsByFirstRow(const SizeType& k, const SizeType& row, const SizeType& col,
                          const matrix::Tile<const T, Device::CPU>& in,
                          const matrix::Tile<T, Device::CPU>& out);

#define DLAF_CPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(kword, Type)                                              \
  kword template void divideColsByFirstRow(const SizeType& k, const SizeType& row, const SizeType& col, \
                                           const matrix::Tile<const Type, Device::CPU>& in,             \
                                           const matrix::Tile<Type, Device::CPU>& out)

DLAF_CPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(extern, float);
DLAF_CPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void divideColsByFirstRow(const SizeType& k, const SizeType& row, const SizeType& col,
                          const matrix::Tile<const T, Device::GPU>& evecs_tile,
                          const matrix::Tile<T, Device::GPU>& ws_tile, whip::stream_t stream);

#define DLAF_GPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(kword, Type)                                              \
  kword template void divideColsByFirstRow(const SizeType& k, const SizeType& row, const SizeType& col, \
                                           const matrix::Tile<const Type, Device::GPU>& in,             \
                                           const matrix::Tile<Type, Device::GPU>& out,                  \
                                           whip::stream_t stream)

DLAF_GPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(extern, float);
DLAF_GPU_DIVIDE_COLS_BY_FIRST_ROW_ETI(extern, double);
#endif

DLAF_MAKE_CALLABLE_OBJECT(divideColsByFirstRow);

template <Device D, class KSender, class InTileSender, class OutTileSender>
void divideColsByFirstRowAsync(KSender&& k, SizeType row, SizeType col, InTileSender&& in,
                               OutTileSender&& out) {
  namespace di = dlaf::internal;

  auto sender = di::whenAllLift(std::forward<KSender>(k), row, col, std::forward<InTileSender>(in),
                                std::forward<OutTileSender>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), divideColsByFirstRow_o, std::move(sender));
}

template <class T>
void setUnitDiagonal(const SizeType& k, const SizeType& tile_begin,
                     const matrix::Tile<T, Device::CPU>& tile);

#define DLAF_CPU_SET_UNIT_DIAGONAL_ETI(kword, Type)                                  \
  kword template void setUnitDiagonal(const SizeType& k, const SizeType& tile_begin, \
                                      const matrix::Tile<Type, Device::CPU>& tile)

DLAF_CPU_SET_UNIT_DIAGONAL_ETI(extern, float);
DLAF_CPU_SET_UNIT_DIAGONAL_ETI(extern, double);

#ifdef DLAF_WITH_GPU
template <class T>
void setUnitDiagonal(const SizeType& k, const SizeType& tile_begin,
                     const matrix::Tile<T, Device::GPU>& tile, whip::stream_t stream);

#define DLAF_GPU_SET_UNIT_DIAGONAL_ETI(kword, Type)                                  \
  kword template void setUnitDiagonal(const SizeType& k, const SizeType& tile_begin, \
                                      const matrix::Tile<Type, Device::GPU>& tile,   \
                                      whip::stream_t stream)

DLAF_GPU_SET_UNIT_DIAGONAL_ETI(extern, float);
DLAF_GPU_SET_UNIT_DIAGONAL_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(setUnitDiagonal);

template <Device D, class KSender, class TileSender>
void setUnitDiagonalAsync(KSender&& k, SizeType tile_begin, TileSender&& tile) {
  namespace di = dlaf::internal;
  auto sender = di::whenAllLift(std::forward<KSender>(k), tile_begin, std::forward<TileSender>(tile));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(), setUnitDiagonal_o, std::move(sender));
}

// Copy the part of the first row/column (@p in_coord) of @p in_tile into the first row/column (@p
// out_coord) of @p out_tile that is inside a sqaure defined by the global element index @p k where both
// tiles begin at global element index (@p row, @p col).
template <class T>
void copy1D(const SizeType& k, const SizeType& row, const SizeType& col, const Coord& in_coord,
            const matrix::Tile<const T, Device::CPU>& in_tile, const Coord& out_coord,
            const matrix::Tile<T, Device::CPU>& out_tile);

#define DLAF_CPU_COPY_1D_ETI(kword, Type)                                                 \
  kword template void copy1D(const SizeType& k, const SizeType& row, const SizeType& col, \
                             const Coord& in_coord,                                       \
                             const matrix::Tile<const Type, Device::CPU>& in_tile,        \
                             const Coord& out_coord, const matrix::Tile<Type, Device::CPU>& out_tile)

DLAF_CPU_COPY_1D_ETI(extern, float);
DLAF_CPU_COPY_1D_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
void copy1D(cublasHandle_t handle, const SizeType& k, const SizeType& row, const SizeType& col,
            const Coord& in_coord, const matrix::Tile<const T, Device::GPU>& in_tile,
            const Coord& out_coord, const matrix::Tile<T, Device::GPU>& out_tile);

#define DLAF_GPU_COPY_1D_ETI(kword, Type)                                                   \
  kword template void copy1D(cublasHandle_t handle, const SizeType& k, const SizeType& row, \
                             const SizeType& col, const Coord& in_coord,                    \
                             const matrix::Tile<const Type, Device::GPU>& in_tile,          \
                             const Coord& out_coord, const matrix::Tile<Type, Device::GPU>& out_tile)

DLAF_GPU_COPY_1D_ETI(extern, float);
DLAF_GPU_COPY_1D_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(copy1D);

template <Device D, class KSender, class InTileSender, class OutTileSender>
void copy1DAsync(KSender&& k, SizeType row, SizeType col, Coord in_coord, InTileSender&& in,
                 Coord out_coord, OutTileSender&& out) {
  namespace di = dlaf::internal;
  namespace ex = pika::execution::experimental;
  auto sender =
      di::whenAllLift(std::forward<KSender>(k), row, col, in_coord, std::forward<InTileSender>(in),
                      out_coord, std::forward<OutTileSender>(out));
  ex::start_detached(di::transform<di::TransformDispatchType::Blas>(di::Policy<DefaultBackend_v<D>>(),
                                                                    copy1D_o, std::move(sender)));
}

// ---------------------------

#ifdef DLAF_WITH_GPU

// Returns the number of non-deflated entries
void stablePartitionIndexOnDevice(SizeType n, const ColType* c_ptr, const SizeType* in_ptr,
                                  SizeType* out_ptr, SizeType* host_k_ptr, SizeType* device_k_ptr,
                                  whip::stream_t stream);

template <class T>
void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, const SizeType* end_ptr,
                          SizeType* out_ptr, const T* v_ptr, whip::stream_t stream);

#define DLAF_CUDA_MERGE_INDICES_ETI(kword, Type)                                                 \
  kword template void mergeIndicesOnDevice(const SizeType* begin_ptr, const SizeType* split_ptr, \
                                           const SizeType* end_ptr, SizeType* out_ptr,           \
                                           const Type* v_ptr, whip::stream_t stream)

DLAF_CUDA_MERGE_INDICES_ETI(extern, float);
DLAF_CUDA_MERGE_INDICES_ETI(extern, double);

template <class T>
void applyIndexOnDevice(SizeType len, const SizeType* index, const T* in, T* out, whip::stream_t stream);

#define DLAF_CUDA_APPLY_INDEX_ETI(kword, Type)                                                \
  kword template void applyIndexOnDevice(SizeType len, const SizeType* index, const Type* in, \
                                         Type* out, whip::stream_t stream)

DLAF_CUDA_APPLY_INDEX_ETI(extern, float);
DLAF_CUDA_APPLY_INDEX_ETI(extern, double);

void invertIndexOnDevice(SizeType len, const SizeType* in, SizeType* out, whip::stream_t stream);

template <class T>
void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s, whip::stream_t stream);

#define DLAF_GIVENS_ROT_ETI(kword, Type)                                                     \
  kword template void givensRotationOnDevice(SizeType len, Type* x, Type* y, Type c, Type s, \
                                             whip::stream_t stream)

DLAF_GIVENS_ROT_ETI(extern, float);
DLAF_GIVENS_ROT_ETI(extern, double);

#endif

}
