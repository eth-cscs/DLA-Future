//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <dlaf/common/callable_object.h>
#include <dlaf/eigensolver/tridiag_solver/coltype.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/memory/memory_chunk.h>
#include <dlaf/sender/traits.h>
#include <dlaf/sender/transform.h>
#include <dlaf/types.h>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>

#include <dlaf/gpu/lapack/api.h>
#endif

namespace dlaf::eigensolver::internal {

template <Device D, class DiagTileSenderIn, class DiagTileSenderOut>
void stedcAsync(DiagTileSenderIn&& in, DiagTileSenderOut&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto sender = ex::when_all(std::forward<DiagTileSenderIn>(in), std::forward<DiagTileSenderOut>(out));
  ex::start_detached(tile::stedc(
      di::Policy<DefaultBackend_v<D>>(pika::execution::thread_stacksize::nostack), std::move(sender)));
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
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(pika::execution::thread_stacksize::nostack),
                      castToComplex_o, std::move(sender));
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

DLAF_MAKE_CALLABLE_OBJECT(cuppensDecomp);

template <class T, class TopTileSender, class BottomTileSender>
auto cuppensDecompAsync(TopTileSender&& top, BottomTileSender&& bottom) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  constexpr auto backend = Backend::MC;

  return ex::when_all(std::forward<TopTileSender>(top), std::forward<BottomTileSender>(bottom)) |
         di::transform(di::Policy<backend>(pika::execution::thread_stacksize::nostack), cuppensDecomp_o);
}

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::CPU>& diag_tile);

#define DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(kword, Type) \
  kword template void copyDiagonalFromCompactTridiagonal(                \
      const matrix::Tile<const Type, Device::CPU>& tridiag_tile,         \
      const matrix::Tile<Type, Device::CPU>& diag_tile)

DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, float);
DLAF_CPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
void copyDiagonalFromCompactTridiagonal(const matrix::Tile<const T, Device::CPU>& tridiag_tile,
                                        const matrix::Tile<T, Device::GPU>& diag_tile,
                                        whip::stream_t stream);

#define DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(kword, Type) \
  kword template void copyDiagonalFromCompactTridiagonal(                \
      const matrix::Tile<const Type, Device::CPU>& tridiag_tile,         \
      const matrix::Tile<Type, Device::GPU>& diag_tile, whip::stream_t stream)

DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, float);
DLAF_GPU_COPY_DIAGONAL_FROM_COMPACT_TRIDIAGONAL_ETI(extern, double);

#endif

DLAF_MAKE_CALLABLE_OBJECT(copyDiagonalFromCompactTridiagonal);

template <Device D, class TridiagTile, class DiagTile>
void copyDiagonalFromCompactTridiagonalAsync(TridiagTile&& in, DiagTile&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto sender = ex::when_all(std::forward<TridiagTile>(in), std::forward<DiagTile>(out));
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(pika::execution::thread_stacksize::nostack),
                      copyDiagonalFromCompactTridiagonal_o, std::move(sender));
}

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::CPU>& evecs_tile,
                                   const matrix::Tile<T, Device::CPU>& rank1_tile);

#define DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(kword, Type)                        \
  kword template void assembleRank1UpdateVectorTile(                                       \
      bool is_top_tile, Type rho, const matrix::Tile<const Type, Device::CPU>& evecs_tile, \
      const matrix::Tile<Type, Device::CPU>& rank1_tile)

DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, float);
DLAF_CPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(extern, double);

#ifdef DLAF_WITH_GPU

template <class T>
void assembleRank1UpdateVectorTile(bool is_top_tile, T rho,
                                   const matrix::Tile<const T, Device::GPU>& evecs_tile,
                                   const matrix::Tile<T, Device::GPU>& rank1_tile,
                                   whip::stream_t stream);

#define DLAF_GPU_ASSEMBLE_RANK1_UPDATE_VECTOR_TILE_ETI(kword, Type)                        \
  kword template void assembleRank1UpdateVectorTile(                                       \
      bool is_top_tile, Type rho, const matrix::Tile<const Type, Device::GPU>& evecs_tile, \
      const matrix::Tile<Type, Device::GPU>& rank1_tile, whip::stream_t stream)

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
  di::transformDetach(di::Policy<DefaultBackend_v<D>>(pika::execution::thread_stacksize::nostack),
                      assembleRank1UpdateVectorTile_o, std::move(sender));
}

#ifdef DLAF_WITH_GPU

template <class T>
void givensRotationOnDevice(SizeType len, T* x, T* y, T c, T s, whip::stream_t stream);

#define DLAF_GIVENS_ROT_ETI(kword, Type)                                                     \
  kword template void givensRotationOnDevice(SizeType len, Type* x, Type* y, Type c, Type s, \
                                             whip::stream_t stream)

DLAF_GIVENS_ROT_ETI(extern, float);
DLAF_GIVENS_ROT_ETI(extern, double);

#endif
}
