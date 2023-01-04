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
/*template <Backend backend, class T, typename InSender, typename OutSender>
void trmmBPanelTile(pika::execution::thread_priority priority, blas::Diag diag, T alpha,
                    InSender&& in_tile, OutSender&& out_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Side::Left, blas::Uplo::Lower, blas::Op::NoTrans, diag, alpha,
                                  std::forward<InSender>(in_tile), std::forward<OutSender>(out_tile)) |
      tile::trmm(dlaf::internal::Policy<backend>(priority)));
}

template <Backend backend, class T, typename ASender, typename BSender, typename CSender>
void gemmTrailingMatrixTile(pika::execution::thread_priority priority, T alpha, ASender&& a_tile,
                            BSender&& b_tile, CSender&& c_tile) {
  pika::execution::experimental::start_detached(
      dlaf::internal::whenAllLift(blas::Op::NoTrans, blas::Op::NoTrans, alpha,
                                  std::forward<ASender>(a_tile), std::forward<BSender>(b_tile), T(1.0),
                                  std::forward<CSender>(c_tile)) |
      tile::gemm(dlaf::internal::Policy<backend>(priority)));
}
*/
}

template <Backend B, Device D, class T>
void Hermitian<B, D, T>::call_LL(const T alpha, Matrix<const T, D>& mat_a, Matrix<const T, D>& mat_b,
                                 const T beta, Matrix<T, D>& mat_c) {
  using namespace hermitian_ll;
  using pika::execution::thread_priority;

  const SizeType m = mat_b.nrTiles().rows();
  const SizeType n = mat_b.nrTiles().cols();

/*  for (SizeType k = m - 1; k >= 0; --k) {
    for (SizeType j = 0; j < n; ++j) {
      auto kj = LocalTileIndex{k, j};

      for (SizeType i = k + 1; i < m; ++i) {
        gemmTrailingMatrixTile<backend>(thread_priority::normal, alpha,
                                        mat_a.read_sender(LocalTileIndex{i, k}), mat_b.read_sender(kj),
                                        mat_b.readwrite_sender(LocalTileIndex{i, j}));
      }

      trmmBPanelTile<backend>(thread_priority::high, diag, alpha,
                              mat_a.read_sender(LocalTileIndex{k, k}), mat_b.readwrite_sender(kj));
    }
  }
*/
}

template <Backend B, Device D, class T>
void Hermitian<B, D, T>::call_LL(comm::CommunicatorGrid grid, const T alpha, Matrix<const T, D>& mat_a,
                                 Matrix<const T, D>& mat_b, const T beta, Matrix<T, D>& mat_c) {
  // Set up MPI executor pipelines
  common::Pipeline<comm::Communicator> mpi_row_task_chain(grid.rowCommunicator().clone());
  common::Pipeline<comm::Communicator> mpi_col_task_chain(grid.colCommunicator().clone());
}

}
