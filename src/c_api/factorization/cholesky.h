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
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/grid.h>

#include "../grid.h"

template <typename T>
void cholesky(int dlaf_context, char uplo, T* a, DLAF_descriptor dlaf_desca) {
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  pika::resume();

  // TODO: Check uplo
  auto dlaf_uplo = (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;

  auto communicator_grid = dlaf_grids.at(dlaf_context);

  dlaf::GlobalElementSize matrix_size(dlaf_desca.m, dlaf_desca.n);
  dlaf::TileElementSize block_size(dlaf_desca.mb, dlaf_desca.nb);

  dlaf::comm::Index2D src_rank_index(dlaf_desca.isrc, dlaf_desca.jsrc);

  dlaf::matrix::Distribution distribution(matrix_size, block_size, communicator_grid.size(),
                                          communicator_grid.rank(), src_rank_index);

  dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, dlaf_desca.ld);

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
void pxpotrf(char uplo, [[maybe_unused]] int n, T* a, [[maybe_unused]] int ia, [[maybe_unused]] int ja,
             int* desca, int& info) {
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  // TODO: Add checks
  // utils::check(uplo, desca, info);
  // if (info == -1)
  //   return;
  // info = -1;  // Reset info to bad state

  pika::resume();

  auto dlaf_uplo = (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;

  // Get grid corresponding to blacs context in desca
  // The grid needs to be created with dlaf_create_grid_from_blacs
  auto communicator_grid = dlaf_grids.at(desca[1]);
  dlaf::matrix::Distribution distribution({desca[2], desca[3]}, {desca[4], desca[5]},
                                          communicator_grid.size(), communicator_grid.rank(), {desca[6], desca[7]});
  dlaf::matrix::LayoutInfo layout_info = colMajorLayout(distribution, desca[8]);

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_host(std::move(distribution), layout_info, a);

  {
    MatrixMirror matrix(matrix_host);

    dlaf::factorization::cholesky<dlaf::Backend::Default, dlaf::Device::Default, T>(communicator_grid,
                                                                                    dlaf_uplo,
                                                                                    matrix.get());
  }  // Destroy mirror

  matrix_host.waitLocalTiles();

  pika::suspend();

  info = 0;
}
