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
#include "../utils.h"

template <typename T>
int eigensolver(int dlaf_context, char uplo, T* a, DLAF_descriptor dlaf_desca, dlaf::BaseType<T>* w,
                T* z, DLAF_descriptor dlaf_descz) {
  try {
    using MatrixHost = dlaf::matrix::Matrix<T, dlaf::Device::CPU>;
    using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;
    using MatrixBaseMirror =
        dlaf::matrix::MatrixMirror<dlaf::BaseType<T>, dlaf::Device::Default, dlaf::Device::CPU>;

    DLAF_ASSERT(dlaf_desca.i == 0, dlaf_desca.i);
    DLAF_ASSERT(dlaf_desca.j == 0, dlaf_desca.j);
    DLAF_ASSERT(dlaf_descz.i == 0, dlaf_descz.i);
    DLAF_ASSERT(dlaf_descz.j == 0, dlaf_descz.j);

    pika::resume();

    auto communicator_grid = dlaf_grids.at(dlaf_context);

    auto [distribution_a, layout_a] = distribution_and_layout(dlaf_desca, communicator_grid);
    auto [distribution_z, layout_z] = distribution_and_layout(dlaf_desca, communicator_grid);

    MatrixHost matrix_host(distribution_a, layout_a, a);
    MatrixHost eigenvectors_host(distribution_z, layout_z, z);
    auto eigenvalues_host =
        dlaf::matrix::createMatrixFromColMajor<dlaf::Device::CPU>({dlaf_descz.m, 1}, {dlaf_descz.mb, 1},
                                                                  std::max(dlaf_descz.m, 1), w);

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
  catch (const std::exception& e) {
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
  DLAF_ASSERT(desca[0] == 1, desca[0]);
  DLAF_ASSERT(descz[0] == 1, descz[0]);
  DLAF_ASSERT(desca[1] == descz[1], desca[1], descz[1]);
  DLAF_ASSERT(ia == 1, ia);
  DLAF_ASSERT(ja == 1, ja);
  DLAF_ASSERT(iz == 1, iz);
  DLAF_ASSERT(iz == 1, iz);

  auto dlaf_desca = make_dlaf_descriptor(m, m, ia, ja, desca);
  auto dlaf_descz = make_dlaf_descriptor(m, m, iz, jz, descz);

  auto _info = eigensolver(desca[1], uplo, a, dlaf_desca, w, z, dlaf_descz);
  info = _info;
}

#endif
