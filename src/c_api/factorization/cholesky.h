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

#include "../blacs.h"
#include "../grid.h"
#include "../utils.h"

template <typename T>
int cholesky(int dlaf_context, char uplo, T* a, DLAF_descriptor dlaf_desca) noexcept {
  try {
    using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

    DLAF_ASSERT(dlaf_desca.i == 1, dlaf_desca.i);
    DLAF_ASSERT(dlaf_desca.j == 1, dlaf_desca.j);

    pika::resume();

    auto dlaf_uplo = dlaf_uplo_from_char(uplo);

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

    return 0;
  }
  catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

#ifdef DLAF_WITH_SCALAPACK

template <typename T>
void pxpotrf(char uplo, int n, T* a, [[maybe_unused]] int ia, [[maybe_unused]] int ja, int* desca,
             int& info) noexcept {
  try {
    using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

    DLAF_ASSERT(desca[0] == 1, desca[0]);
    DLAF_ASSERT(ia == 1, ia);
    DLAF_ASSERT(ja == 1, ja);

    pika::resume();

    auto dlaf_uplo = dlaf_uplo_from_char(uplo);

    // Get grid corresponding to blacs context in desca
    // The grid needs to be created with dlaf_create_grid_from_blacs
    auto communicator_grid = dlaf_grids.at(desca[1]);
    dlaf::matrix::Distribution distribution({n, n}, {desca[4], desca[5]}, communicator_grid.size(),
                                            communicator_grid.rank(), {desca[6], desca[7]});
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
  catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    info = -1;
  }
}

#endif
