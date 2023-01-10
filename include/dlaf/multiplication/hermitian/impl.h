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

#include <pika/execution.hpp>
#include <pika/future.hpp>
#include <pika/thread.hpp>

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/round_robin.h"
#include "dlaf/communication/broadcast_panel.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/kernels.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/multiplication/hermitian/api.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/util_matrix.h"

namespace dlaf::multiplication::internal {

namespace hermitian_ll {
template <Backend B, class T, typename ASender, typename BSender, typename CSender>
void hemm(const T alpha, ASender&& a_tile, BSender&& b_tile, const T beta, CSender&& c_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, alpha,
                                  std::forward<ASender>(a_tile), std::forward<BSender>(b_tile), beta,
                                  std::forward<CSender>(c_tile)) |
      tile::hemm(dlaf::internal::Policy<B>(pika::execution::thread_priority::normal)));
}

template <Backend B, class T, typename ASender, typename BSender, typename CSender>
void gemmN(const T alpha, ASender&& a_tile, BSender&& b_tile, const T beta, CSender&& c_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                  std::forward<ASender>(a_tile), std::forward<BSender>(b_tile), beta,
                                  std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<B>(pika::execution::thread_priority::normal)));
}

template <Backend B, class T, typename ASender, typename BSender, typename CSender>
void gemmC(const T alpha, ASender&& a_tile, BSender&& b_tile, const T beta, CSender&& c_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::ConjTrans, blas::Op::NoTrans, alpha,
                                  std::forward<ASender>(a_tile), std::forward<BSender>(b_tile), beta,
                                  std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<B>(pika::execution::thread_priority::normal)));
}
}

template <Backend B, Device D, class T>
void Hermitian<B, D, T>::call_LL(const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b,
                                 const T beta, Matrix<T, D>& mat_c) {
  using namespace hermitian_ll;
  const SizeType k = mat_a.distribution().localNrTiles().cols();

  for (const auto ij : common::iterate_range2d(mat_c.distribution().localNrTiles())) {
    T beta_ij = beta;
    for (SizeType l = 0; l < ij.row(); ++l) {
      auto il = LocalTileIndex{ij.row(), l};
      auto lj = LocalTileIndex{l, ij.col()};
      gemmN<B>(alpha, mat_a.read_sender(il), mat_b.read_sender(lj), beta_ij, mat_c.readwrite_sender(ij));
      beta_ij = 1;
    }

    {
      auto ii = LocalTileIndex{ij.row(), ij.row()};
      hemm<B>(alpha, mat_a.read_sender(ii), mat_b.read_sender(ij), beta_ij, mat_c.readwrite_sender(ij));
      beta_ij = 1;
    }

    for (SizeType l = ij.row() + 1; l < k; ++l) {
      auto li = LocalTileIndex{l, ij.row()};
      auto lj = LocalTileIndex{l, ij.col()};
      gemmC<B>(alpha, mat_a.read_sender(li), mat_b.read_sender(lj), beta_ij, mat_c.readwrite_sender(ij));
      beta_ij = 1;
    }
  }
}

template <Backend B, Device D, class T>
void Hermitian<B, D, T>::call_LL(comm::CommunicatorGrid grid, const T alpha, Matrix<const T, D>& mat_a,
                                 Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());
}

}
