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

#include "dlaf/factorization/qr/api.h"
#include "dlaf/matrix/index.h"

namespace dlaf::factorization::internal {

/// Forms the triangular factor T of a block reflector H of order n,
/// which is defined as a product of k := hh_panel.getWidth() elementary reflectors.
///
/// hh_panel should have the following form
/// H0  0  0 ...    0
///  . H1  0 ...    0
///  .  . H2 ...    0
///  .  .  . ...    0
///  .  .  . ... HK-1
///  .  .  . ...    .
/// H0 H1 H2 ... HK-1
/// Note: The first element of the HH reflectors is NOT implicitly assumed to be 1,
///       it has to be set correctly in the panel (0s as well).
///
/// It is similar to what xLARFT in LAPACK does.
/// Given @p k elementary reflectors stored in the column of @p hh_panel together with related tau values
/// in @p taus, in @p t will be formed the triangular factor for the H block of reflectors, such that
///
/// H = I - V . T . V*
///
/// where H = H1 . H2 . ... . Hk
///
/// in which Hi represents a single elementary reflector transformation.
///
/// A Storage-Efficient WY Representation for Products of Householder Transformations.
/// Schreiber, Robert & VanLoan, Charles. (1989)
/// SIAM Journal on Scientific and Statistical Computing. 10. 10.1137/0910005.
///
/// @param hh_panel where the elementary reflectors are stored
/// @param taus array of taus, associated with the related elementary reflector
/// @param t tile where the resulting T factor will be stored in its top-left sub-matrix of size
/// TileElementSize(k, k)
/// @param mpi_col_task_chain where internal communications are issued
///
/// @pre hh_pane.getWidth() <= t.get().size().rows && hh_panel.size().getWidth() <= t.get().size().cols()
template <Backend backend, Device device, class T>
void computeTFactor(matrix::Panel<Coord::Col, T, device>& hh_panel,
                    pika::shared_future<common::internal::vector<T>> taus,
                    pika::future<matrix::Tile<T, device>> t) {
  QR_Tfactor<backend, device, T>::call(hh_panel, taus, std::move(t));
}

/// Forms the triangular factor T of a block of reflectors H, which is defined as a product of
/// k := hh_panel.getWidth() elementary reflectors.
///
/// hh_panel should have the following form
/// H0  0  0 ...    0
///  . H1  0 ...    0
///  .  . H2 ...    0
///  .  .  . ...    0
///  .  .  . ... HK-1
///  .  .  . ...    .
/// H0 H1 H2 ... HK-1
/// Note: The first element of the HH reflectors is NOT implicitly assumed to be 1,
///       it has to be set correctly in the panel (0s as well).
///
/// It is similar to what xLARFT in LAPACK does.
/// Given @p k elementary reflectors stored in the column of @p hh_panel together with related tau values
/// in @p taus, in @p t will be formed the triangular factor for the H block of reflectors, such that
///
/// H = I - V . T . V*
///
/// where H = H1 . H2 . ... . Hk
///
/// in which Hi represents a single elementary reflector transformation.
///
/// A Storage-Efficient WY Representation for Products of Householder Transformations.
/// Schreiber, Robert & VanLoan, Charles. (1989)
/// SIAM Journal on Scientific and Statistical Computing. 10. 10.1137/0910005.
///
/// @param hh_panel where the elementary reflectors are stored
/// @param taus array of taus, associated with the related elementary reflector
/// @param t tile where the resulting T factor will be stored in its top-left sub-matrix of size
/// TileElementSize(k, k)
/// @param mpi_col_task_chain where internal communications are issued
///
/// @pre hh_pane.getWidth() <= t.get().size().rows && hh_panel.size().getWidth() <= t.get().size().cols()
template <Backend backend, Device device, class T>
void computeTFactor(matrix::Panel<Coord::Col, T, device>& hh_panel,
                    pika::shared_future<common::internal::vector<T>> taus,
                    pika::future<matrix::Tile<T, device>> t,
                    common::Pipeline<comm::Communicator>& mpi_col_task_chain) {
  QR_Tfactor<backend, device, T>::call(hh_panel, taus, std::move(t), mpi_col_task_chain);
}

}
