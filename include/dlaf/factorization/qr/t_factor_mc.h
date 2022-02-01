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

#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include <blas.hh>

#include "dlaf/factorization/qr/api.h"

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/kernels/all_reduce.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace factorization {
namespace internal {

template <class T>
struct QR_Tfactor<Backend::MC, Device::CPU, T> {
  /// Forms the triangular factor T of a block of reflectors H, which is defined as a product of k
  /// elementary reflectors.
  ///
  /// It is similar to what xLARFT in LAPACK does.
  /// Given @p k elementary reflectors stored in the column of @p v starting at tile @p v_start,
  /// together with related tau values in @p taus, in @p t will be formed the triangular factor for the H
  /// block of reflector, such that
  ///
  /// H = I - V . T . V*
  ///
  /// where H = H1 . H2 . ... . Hk
  ///
  /// in which Hi represents a single elementary reflector transformation
  ///
  /// @param k the number of elementary reflectors to use (from the beginning of the tile)
  /// @param v where the elementary reflectors are stored
  /// @param v_start tile in @p v where the column of reflectors starts
  /// @param taus array of taus, associated with the related elementary reflector
  /// @param t tile where the resulting T factor will be stored in its top-left sub-matrix of size
  /// TileElementSize(k, k)
  ///
  /// @pre k <= t.get().size().rows && k <= t.get().size().cols()
  /// @pre k >= 0
  /// @pre v_start.isIn(v.nrTiles())
  static void call(const SizeType k, Matrix<const T, Device::CPU>& v, const GlobalTileIndex v_start,
                   pika::shared_future<common::internal::vector<T>> taus,
                   pika::future<matrix::Tile<T, Device::CPU>> t);

  /// Forms the triangular factor T of a block of reflectors H, which is defined as a product of k
  /// elementary reflectors.
  ///
  /// It is similar to what xLARFT in LAPACK does.
  /// Given @p k elementary reflectors stored in the column of @p v starting at tile @p v_start,
  /// together with related tau values in @p taus, in @p t will be formed the triangular factor for the H
  /// block of reflector, such that
  ///
  /// H = I - V . T . V*
  ///
  /// where H = H1 . H2 . ... . Hk
  ///
  /// in which Hi represents a single elementary reflector transformation
  ///
  /// @param k the number of elementary reflectors to use (from the beginning of the tile)
  /// @param v where the elementary reflectors are stored
  /// @param v_start tile in @p v where the column of reflectors starts
  /// @param taus array of taus, associated with the related elementary reflector
  /// @param t tile where the resulting T factor will be stored in its top-left sub-matrix of size
  /// TileElementSize(k, k)
  /// @param mpi_col_task_chain where internal communications are issued
  ///
  /// @pre k <= t.get().size().rows && k <= t.get().size().cols()
  /// @pre k >= 0
  /// @pre v_start.isIn(v.nrTiles())
  static void call(const SizeType k, Matrix<const T, Device::CPU>& v, const GlobalTileIndex v_start,
                   pika::shared_future<common::internal::vector<T>> taus,
                   pika::future<matrix::Tile<T, Device::CPU>> t,
                   common::Pipeline<comm::Communicator>& mpi_col_task_chain);
};

template <class T>
pika::future<matrix::Tile<T, Device::CPU>> gemvColumnT(
    bool is_v0, pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_vi,
    pika::shared_future<common::internal::vector<T>>& taus,
    pika::future<matrix::Tile<T, Device::CPU>>& tile_t) {
  auto gemv_func = pika::unwrapping([is_v0](const auto& tile_v, const auto& taus, auto&& tile_t) {
    const SizeType k = tile_t.size().cols();
    DLAF_ASSERT(taus.size() == k, taus.size(), k);

    for (SizeType j = 0; j < k; ++j) {
      const T tau = taus[j];
      // this is the x0 element of the reflector j
      const TileElementIndex x0{j, j};

      const TileElementIndex t_start{0, x0.col()};

      const SizeType first_element_in_tile = is_v0 ? x0.row() : 0;

      // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
      // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
      TileElementSize va_size{tile_v.size().rows() - first_element_in_tile, x0.col()};
      TileElementIndex va_start{first_element_in_tile, 0};
      TileElementIndex vb_start{first_element_in_tile, x0.col()};

      if (is_v0) {
        tile_t(x0) = tau;

        // use implicit 1 for the 2nd operand
        for (SizeType r = 0; !va_size.isEmpty() && r < va_size.cols(); ++r) {
          const TileElementIndex i_v{va_start.row(), r + va_start.col()};
          const TileElementIndex i_t{r + t_start.row(), t_start.col()};

          tile_t(i_t) = -tau * dlaf::conj(tile_v(i_v));
        }

        // skip already managed computations with implicit 1
        va_start = va_start + TileElementSize{1, 0};
        vb_start = vb_start + TileElementSize{1, 0};
        va_size = {va_size.rows() - 1, va_size.cols()};
      }

      if (!va_size.isEmpty()) {
        blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, va_size.rows(), va_size.cols(), -tau,
                   tile_v.ptr(va_start), tile_v.ld(), tile_v.ptr(vb_start), 1, 1, tile_t.ptr(t_start),
                   1);
      }
    }
    return std::move(tile_t);
  });
  return pika::dataflow(getHpExecutor<Backend::MC>(), gemv_func, tile_vi, taus, tile_t);
}

template <class T>
pika::future<matrix::Tile<T, Device::CPU>> trmvUpdateColumn(
    pika::future<matrix::Tile<T, Device::CPU>>& tile_t) {
  // Update each column (in order) t = T . t
  // remember that T is upper triangular, so it is possible to use TRMV
  auto trmv_func = pika::unwrapping([](auto&& tile_t) {
    for (SizeType j = 0; j < tile_t.size().cols(); ++j) {
      const TileElementIndex t_start{0, j};
      const TileElementSize t_size{j, 1};

      blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
                 t_size.rows(), tile_t.ptr(), tile_t.ld(), tile_t.ptr(t_start), 1);
    }
    return std::move(tile_t);
  });
  return pika::dataflow(getHpExecutor<Backend::MC>(), trmv_func, tile_t);
}

template <class T>
void QR_Tfactor<Backend::MC, Device::CPU, T>::call(const SizeType k, Matrix<const T, Device::CPU>& v,
                                                   const GlobalTileIndex v_start,
                                                   pika::shared_future<common::internal::vector<T>> taus,
                                                   pika::future<matrix::Tile<T, Device::CPU>> t) {
  namespace ex = pika::execution::experimental;

  t = splitTile(t, {{0, 0}, {k, k}});

  // Fast return in case of no reflectors
  if (k == 0)
    return;

  const auto panel_width = v.tileSize(v_start).cols();

  DLAF_ASSERT(k <= panel_width, k, panel_width);

  const GlobalTileIndex v_end{v.nrTiles().rows(), std::min(v.nrTiles().cols(), v_start.col() + 1)};

  constexpr auto set0_return_tile = [](matrix::Tile<T, Device::CPU>&& tile) {
    tile::internal::set0<T>(tile);
    return std::move(tile);
  };
  t = dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(
                                    pika::threads::thread_priority::high),
                                set0_return_tile, std::move(t)) |
      ex::make_future();

  // Note:
  // T factor is an upper triangular square matrix, built column by column
  // with taus values on the diagonal
  //
  // T(j,j) = tau(j)
  //
  // and in the upper triangular part the following formula applies
  //
  // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
  //
  //
  // The result is achieved in two main steps:
  // 1) t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) T(0:j, j) = T(0:j, 0:j) . t

  // 1st step: compute the column partial result `t`
  // First we compute the matrix vector multiplication for each column
  // -tau(j) . V(j:, 0:j)* . V(j:, j)
  for (const auto& v_i : iterate_range2d(v_start, v_end)) {
    const bool is_v0 = (v_i.row() == v_start.row());

    // TODO
    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    t = gemvColumnT(is_v0, v.read(v_i), taus, t);
  }

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  t = trmvUpdateColumn(t);
}

template <class T>
void QR_Tfactor<Backend::MC, Device::CPU, T>::call(
    const SizeType k, Matrix<const T, Device::CPU>& v, const GlobalTileIndex v_start,
    pika::shared_future<common::internal::vector<T>> taus, pika::future<matrix::Tile<T, Device::CPU>> t,
    common::Pipeline<comm::Communicator>& mpi_col_task_chain) {
  namespace ex = pika::execution::experimental;

  t = splitTile(t, {{0, 0}, {k, k}});

  // Fast return in case of no reflectors
  if (k == 0)
    return;

  const auto panel_width = v.tileSize(v_start).cols();
  DLAF_ASSERT(0 <= k && k <= panel_width, k, panel_width);

  DLAF_ASSERT_MODERATE(v_start.isIn(v.nrTiles()), v_start, v.nrTiles());
  const auto& dist = v.distribution();
  const comm::Index2D rank = dist.rankIndex();
  const comm::Index2D rank_v0 = dist.rankGlobalTile(v_start);

  // Just the column of ranks with the reflectors participates
  if (rank.col() != rank_v0.col())
    return;

  const LocalTileIndex v_start_loc{
      dist.template nextLocalTileFromGlobalTile<Coord::Row>(v_start.row()),
      dist.template nextLocalTileFromGlobalTile<Coord::Col>(v_start.col()),
  };
  const LocalTileIndex v_end_loc{dist.localNrTiles().rows(), v_start_loc.col() + 1};

  constexpr auto set0_return_tile = [](matrix::Tile<T, Device::CPU>&& tile) {
    tile::internal::set0<T>(tile);
    return std::move(tile);
  };
  t = dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(
                                    pika::threads::thread_priority::high),
                                set0_return_tile, std::move(t)) |
      ex::make_future();

  // Note:
  // T factor is an upper triangular square matrix, built column by column
  // with taus values on the diagonal
  //
  // T(j,j) = tau(j)
  //
  // and in the upper triangular part the following formula applies
  //
  // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
  //
  //
  // The result is achieved in two main steps:
  // 1) t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) T(0:j, j) = T(0:j, 0:j) . t

  // 1st step: compute the column partial result `t`
  // First we compute the matrix vector multiplication for each column
  // -tau(j) . V(j:, 0:j)* . V(j:, j)
  for (const auto& v_i_loc : iterate_range2d(v_start_loc, v_end_loc)) {
    const SizeType v_i = dist.template globalTileFromLocalTile<Coord::Row>(v_i_loc.row());
    const bool is_v0 = (v_i == v_start.row());

    // TODO
    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    t = gemvColumnT(is_v0, v.read(v_i_loc), taus, t);
  }

  // at this point each rank has its partial result for each column
  // so, let's reduce the results (on all ranks, so that everyone can independently compute T factor)
  if (true)  // TODO if the column communicator has more than 1 tile...but I just have the pipeline
    t = scheduleAllReduceInPlace(getMPIExecutor<Backend::MC>(), mpi_col_task_chain(), MPI_SUM,
                                 std::move(t));

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  t = trmvUpdateColumn(t);
}
}
}
}
