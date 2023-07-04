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

#include <dlaf/common/assert.h>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/grid.h>

#include "../blacs.h"
#include "../grid.h"

template <typename T>
void eigensolver(int dlaf_context, char uplo, T* a, DLAF_descriptor dlaf_desca, T* w, T* z,
                 DLAF_descriptor dlaf_descz) {
  using MatrixHost = dlaf::matrix::Matrix<T, dlaf::Device::CPU>;
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  DLAF_ASSERT(uplo == 'L' || uplo == 'l', uplo);
  DLAF_ASSERT(dlaf_desca.m == dlaf_desca.n, dlaf_desca.m, dlaf_desca.n);
  DLAF_ASSERT(dlaf_descz.m == dlaf_descz.n, dlaf_descz.m, dlaf_descz.n);
  DLAF_ASSERT(dlaf_desca.m == dlaf_descz.m, dlaf_desca.m, dlaf_descz.n);

  pika::resume();

  [[maybe_unused]] auto dlaf_uplo = (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;

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
    MatrixMirror eigenvalues(eigenvalues_host);

    // TODO: Use dlaf_uplo instead of hard-coded blas::Uplo::Lower
    // TODO: blas::Uplo::Uppper is not yet supported in DLA-Future
    dlaf::eigensolver::eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(
        communicator_grid, blas::Uplo::Lower, matrix.get(), eigenvalues.get(), eigenvectors.get());
  }  // Destroy mirror

  // Ensure data is copied back to the host
  eigenvalues_host.waitLocalTiles();
  eigenvectors_host.waitLocalTiles();

  pika::suspend();
}

#ifdef DLAF_WITH_SCALAPACK

template <typename T>
void pxsyevd(char uplo, int m, T* a, [[maybe_unused]] int ia, [[maybe_unused]] int ja, int* desca, T* w,
             T* z, [[maybe_unused]] int iz, [[maybe_unused]] int jz, int* descz, int& info) {
  using MatrixHost = dlaf::matrix::Matrix<T, dlaf::Device::CPU>;
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  DLAF_ASSERT(uplo == 'L' || uplo == 'l', uplo);
  DLAF_ASSERT(desca[0] == 1, desca[0]);
  DLAF_ASSERT(descz[0] == 1, descz[0]);

  pika::resume();

  [[maybe_unused]] auto dlaf_uplo = (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;

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
    MatrixMirror eigenvalues(eigenvalues_host);

    // TODO: Use dlaf_uplo instead of hard-coded blas::Uplo::Lower
    // TODO: blas::Uplo::Uppper is not yet supported in DLA-Future
    dlaf::eigensolver::eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(
        communicator_grid, blas::Uplo::Lower, matrix.get(), eigenvalues.get(), eigenvectors.get());
  }  // Destroy mirror

  // Ensure data is copied back to the host
  eigenvalues_host.waitLocalTiles();
  eigenvectors_host.waitLocalTiles();

  pika::suspend();

  info = 0;
}

#endif
