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

#include "dlaf/common/pipeline.h"
#include "dlaf/common/vector.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"

namespace dlaf::factorization::internal {

template <Backend backend, Device device, class T>
struct QR {};

template <Backend backend, Device device, class T>
struct QR_Tfactor {
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
  static void call(const SizeType k, Matrix<const T, device>& v, const GlobalTileIndex v_start,
                   pika::shared_future<common::internal::vector<T>> taus,
                   pika::future<matrix::Tile<T, device>> t);

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
  static void call(const SizeType k, Matrix<const T, device>& v, const GlobalTileIndex v_start,
                   pika::shared_future<common::internal::vector<T>> taus,
                   pika::future<matrix::Tile<T, device>> t,
                   common::Pipeline<comm::Communicator>& mpi_col_task_chain);
};

}
