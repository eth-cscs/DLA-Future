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

#include <blas.hh>
#include <mpi.h>

#include <pika/init.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/types.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/grid.h>

#include "../blacs.h"
#include "../grid.h"

template <typename T>
int eigensolver(int dlaf_context, char uplo, T* a, DLAF_descriptor dlaf_desca, dlaf::BaseType<T>* w,
                T* z, DLAF_descriptor dlaf_descz) {
  try {
    using MatrixHost = dlaf::matrix::Matrix<T, dlaf::Device::CPU>;
    using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;
    using MatrixBaseMirror =
        dlaf::matrix::MatrixMirror<dlaf::BaseType<T>, dlaf::Device::Default, dlaf::Device::CPU>;

    DLAF_ASSERT(dlaf_desca.i == 1, dlaf_desca.i);
    DLAF_ASSERT(dlaf_desca.j == 1, dlaf_desca.j);
    DLAF_ASSERT(dlaf_descz.i == 1, dlaf_descz.i);
    DLAF_ASSERT(dlaf_descz.j == 1, dlaf_descz.j);

    pika::resume();

    auto communicator_grid = dlaf_grids.at(dlaf_context);

    dlaf::GlobalElementSize matrix_size(dlaf_desca.m, dlaf_desca.n);
    dlaf::TileElementSize block_size(dlaf_desca.mb, dlaf_desca.nb);

    dlaf::comm::Index2D src_rank_index(dlaf_desca.isrc, dlaf_desca.jsrc);

    dlaf::matrix::Distribution distribution(matrix_size, block_size, communicator_grid.size(),
                                            communicator_grid.rank(), src_rank_index);

    dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, dlaf_desca.ld);

    MatrixHost matrix_host(distribution, layout, a);
    MatrixHost eigenvectors_host(distribution, layout, z);
    auto eigenvalues_host = dlaf::matrix::createMatrixFromColMajor<dlaf::Device::CPU>(
        {dlaf_descz.m, 1}, {distribution.blockSize().rows(), 1}, dlaf_descz.m, w);

    {
      MatrixMirror matrix(matrix_host);
      MatrixMirror eigenvectors(eigenvectors_host);
      MatrixBaseMirror eigenvalues(eigenvalues_host);

      dlaf::eigensolver::eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(
          communicator_grid, blas::char2uplo(uplo), matrix.get(), eigenvalues.get(), eigenvectors.get());
    }  // Destroy mirror

    // Ensure data is copied back to the host
    eigenvalues_host.waitLocalTiles();
    eigenvectors_host.waitLocalTiles();

    pika::suspend();
    return 0;
  }
  catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
  catch (...) {
    std::cerr << "ERROR: Unknown exception caught in DLA-Future's eigensolver." << '\n';
    return -1;
  }
}

#ifdef DLAF_WITH_SCALAPACK

template <typename T>
void pxxxevd(char uplo, int m, T* a, [[maybe_unused]] int ia, [[maybe_unused]] int ja, int* desca,
             dlaf::BaseType<T>* w, T* z, [[maybe_unused]] int iz, [[maybe_unused]] int jz, int* descz,
             int& info) {
  try {
    using MatrixHost = dlaf::matrix::Matrix<T, dlaf::Device::CPU>;
    using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;
    using MatrixBaseMirror =
        dlaf::matrix::MatrixMirror<dlaf::BaseType<T>, dlaf::Device::Default, dlaf::Device::CPU>;

    DLAF_ASSERT(desca[0] == 1, desca[0]);
    DLAF_ASSERT(descz[0] == 1, descz[0]);
    DLAF_ASSERT(ia == 1, ia);
    DLAF_ASSERT(ja == 1, ja);
    DLAF_ASSERT(iz == 1, iz);
    DLAF_ASSERT(iz == 1, iz);

    pika::resume();

    // Get grid corresponding to blacs context in desca
    // The grid needs to be created with dlaf_create_grid_from_blacs
    auto communicator_grid = dlaf_grids.at(desca[1]);
    dlaf::matrix::Distribution distribution({m, m}, {desca[4], desca[5]}, communicator_grid.size(),
                                            communicator_grid.rank(), {desca[6], desca[7]});
    dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, desca[8]);

    MatrixHost matrix_host(distribution, layout, a);
    MatrixHost eigenvectors_host(distribution, layout, z);
    auto eigenvalues_host = dlaf::matrix::createMatrixFromColMajor<dlaf::Device::CPU>(
        {m, 1}, {distribution.blockSize().rows(), 1}, m, w);
    {
      MatrixMirror matrix(matrix_host);
      MatrixMirror eigenvectors(eigenvectors_host);
      MatrixBaseMirror eigenvalues(eigenvalues_host);

      dlaf::eigensolver::eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(
          communicator_grid, blas::char2uplo(uplo), matrix.get(), eigenvalues.get(), eigenvectors.get());
    }  // Destroy mirror

    // Ensure data is copied back to the host
    eigenvalues_host.waitLocalTiles();
    eigenvectors_host.waitLocalTiles();

    pika::suspend();

    info = 0;
  }
  catch (std::exception& e) {
    std::cerr << e.what() << '\n';
    info = -1;
  }
  catch (...) {
    std::cerr << "ERROR: Unknown exception caught in DLA-Future's eigensolver." << '\n';
    info = -1;
  }
}

#endif
