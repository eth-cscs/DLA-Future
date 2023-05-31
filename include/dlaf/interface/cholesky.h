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

#include <mpi.h>

#include <pika/init.hpp>
#include <dlaf/factorization/cholesky.h>
#include <dlaf/interface/blacs.h>
#include <dlaf/interface/utils.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>

namespace dlaf::interface {

template <typename T>
void pxpotrf(char uplo, T* a, int m, int n, int mb, int nb, int lld, const MPI_Comm& communicator,
             int nprow, int npcol) {
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  pika::resume();

  // TODO: Check uplo
  auto dlaf_uplo = (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;

  dlaf::comm::Communicator world(communicator);
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  dlaf::comm::CommunicatorGrid communicator_grid(world, nprow, npcol, dlaf::common::Ordering::RowMajor);

  dlaf::GlobalElementSize matrix_size(m, n);
  dlaf::TileElementSize block_size(mb, nb);

  dlaf::comm::Index2D src_rank_index(0, 0);  // WARN: Is this always the case?

  dlaf::matrix::Distribution distribution(matrix_size, block_size, communicator_grid.size(),
                                          communicator_grid.rank(), src_rank_index);

  dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, lld);

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_host(std::move(distribution), layout, a);

  {
    MatrixMirror matrix(matrix_host);

    dlaf::factorization::cholesky<dlaf::Backend::Default, dlaf::Device::Default, T>(communicator_grid,
                                                                                    dlaf_uplo,
                                                                                    matrix.get());
  }  // Destroy mirror

  matrix_host.waitLocalTiles();

  pika::suspend();
}

template <typename T>
void pxpotrf(char uplo, int n, T* a, int ia, int ja, int* desca, int& info) {
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  utils::check(uplo, desca, info);
  if (info == -1)
    return;
  info = -1;  // Reset info to bad state

  pika::resume();

  auto dlaf_uplo = (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;

  auto dlaf_info = blacs::from_desc(desca);

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_host(std::move(dlaf_info.distribution),
                                                         dlaf_info.layout_info, a);

  {
    MatrixMirror matrix(matrix_host);

    dlaf::factorization::cholesky<dlaf::Backend::Default, dlaf::Device::Default,
                                  T>(dlaf_info.communicator_grid, dlaf_uplo, matrix.get());
  }  // Destroy mirror

  matrix_host.waitLocalTiles();

  pika::suspend();

  info = 0;
}
}
