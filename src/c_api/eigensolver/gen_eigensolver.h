//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
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
#include <dlaf/eigensolver/gen_eigensolver.h>
#include <dlaf/matrix/create_matrix.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/types.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/grid.h>

#include "../blacs.h"
#include "../utils.h"

template <typename T>
int hermitian_generalized_eigensolver_helper(const int dlaf_context, const char uplo, T* a,
                                             const DLAF_descriptor dlaf_desca, T* b,
                                             const DLAF_descriptor dlaf_descb, dlaf::BaseType<T>* w,
                                             T* z, const DLAF_descriptor dlaf_descz, bool factorized) {
  using MatrixHost = dlaf::matrix::Matrix<T, dlaf::Device::CPU>;
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;
  using MatrixBaseMirror =
      dlaf::matrix::MatrixMirror<dlaf::BaseType<T>, dlaf::Device::Default, dlaf::Device::CPU>;

  DLAF_ASSERT(dlaf_desca.i == 0, dlaf_desca.i);
  DLAF_ASSERT(dlaf_desca.j == 0, dlaf_desca.j);
  DLAF_ASSERT(dlaf_descb.i == 0, dlaf_descb.i);
  DLAF_ASSERT(dlaf_descb.j == 0, dlaf_descb.j);
  DLAF_ASSERT(dlaf_descz.i == 0, dlaf_descz.i);
  DLAF_ASSERT(dlaf_descz.j == 0, dlaf_descz.j);

  pika::resume();

  auto& communicator_grid = grid_from_context(dlaf_context);

  auto [distribution_a, layout_a] = distribution_and_layout(dlaf_desca, communicator_grid);
  auto [distribution_b, layout_b] = distribution_and_layout(dlaf_descb, communicator_grid);
  auto [distribution_z, layout_z] = distribution_and_layout(dlaf_descz, communicator_grid);

  MatrixHost matrix_host_a(distribution_a, layout_a, a);
  MatrixHost matrix_host_b(distribution_b, layout_b, b);
  MatrixHost eigenvectors_host(distribution_z, layout_z, z);
  auto eigenvalues_host =
      dlaf::matrix::createMatrixFromColMajor<dlaf::Device::CPU>({dlaf_descz.m, 1}, {dlaf_descz.mb, 1},
                                                                std::max(dlaf_descz.m, 1), w);

  {
    MatrixMirror matrix_a(matrix_host_a);
    MatrixMirror matrix_b(matrix_host_b);
    MatrixMirror eigenvectors(eigenvectors_host);
    MatrixBaseMirror eigenvalues(eigenvalues_host);

    if (!factorized) {
      dlaf::hermitian_generalized_eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(
          communicator_grid, blas::char2uplo(uplo), matrix_a.get(), matrix_b.get(), eigenvalues.get(),
          eigenvectors.get());
    }
    else {
      dlaf::hermitian_generalized_eigensolver_factorized<dlaf::Backend::Default, dlaf::Device::Default,
                                                         T>(communicator_grid, blas::char2uplo(uplo),
                                                            matrix_a.get(), matrix_b.get(),
                                                            eigenvalues.get(), eigenvectors.get());
    }
  }  // Destroy mirror

  // Ensure data is copied back to the host
  eigenvalues_host.waitLocalTiles();
  eigenvectors_host.waitLocalTiles();

  pika::suspend();
  return 0;
}

template <typename T>
int hermitian_generalized_eigensolver(const int dlaf_context, const char uplo, T* a,
                                      const DLAF_descriptor dlaf_desca, T* b,
                                      const DLAF_descriptor dlaf_descb, dlaf::BaseType<T>* w, T* z,
                                      const DLAF_descriptor dlaf_descz) {
  return hermitian_generalized_eigensolver_helper<T>(dlaf_context, uplo, a, dlaf_desca, b, dlaf_descb, w,
                                                     z, dlaf_descz, false);
}

template <typename T>
int hermitian_generalized_eigensolver_factorized(const int dlaf_context, const char uplo, T* a,
                                                 const DLAF_descriptor dlaf_desca, T* b,
                                                 const DLAF_descriptor dlaf_descb, dlaf::BaseType<T>* w,
                                                 T* z, const DLAF_descriptor dlaf_descz) {
  return hermitian_generalized_eigensolver_helper<T>(dlaf_context, uplo, a, dlaf_desca, b, dlaf_descb, w,
                                                     z, dlaf_descz, true);
}

#ifdef DLAF_WITH_SCALAPACK

template <typename T>
void pxhegvd_helper(const char uplo, const int m, T* a, const int ia, const int ja, const int desca[9],
                    T* b, const int ib, const int jb, const int descb[9], dlaf::BaseType<T>* w, T* z,
                    const int iz, int jz, const int descz[9], int& info, bool factorized) {
  DLAF_ASSERT(desca[0] == 1, desca[0]);
  DLAF_ASSERT(descb[0] == 1, descb[0]);
  DLAF_ASSERT(descz[0] == 1, descz[0]);
  DLAF_ASSERT(desca[1] == descb[1], desca[1], descb[1]);
  DLAF_ASSERT(desca[1] == descz[1], desca[1], descz[1]);
  DLAF_ASSERT(ia == 1, ia);
  DLAF_ASSERT(ja == 1, ja);
  DLAF_ASSERT(ib == 1, ib);
  DLAF_ASSERT(jb == 1, jb);
  DLAF_ASSERT(iz == 1, iz);
  DLAF_ASSERT(iz == 1, iz);

  auto dlaf_desca = make_dlaf_descriptor(m, m, ia, ja, desca);
  auto dlaf_descb = make_dlaf_descriptor(m, m, ib, jb, descb);
  auto dlaf_descz = make_dlaf_descriptor(m, m, iz, jz, descz);

  if (!factorized) {
    info = hermitian_generalized_eigensolver<T>(desca[1], uplo, a, dlaf_desca, b, dlaf_descb, w, z,
                                                dlaf_descz);
  }
  else {
    info = hermitian_generalized_eigensolver_factorized<T>(desca[1], uplo, a, dlaf_desca, b, dlaf_descb,
                                                           w, z, dlaf_descz);
  }
}

template <typename T>
void pxhegvd(const char uplo, const int m, T* a, const int ia, const int ja, const int desca[9], T* b,
             const int ib, const int jb, const int descb[9], dlaf::BaseType<T>* w, T* z, const int iz,
             int jz, const int descz[9], int& info) {
  pxhegvd_helper<T>(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, info, false);
}

template <typename T>
void pxhegvd_factorized(const char uplo, const int m, T* a, const int ia, const int ja,
                        const int desca[9], T* b, const int ib, const int jb, const int descb[9],
                        dlaf::BaseType<T>* w, T* z, const int iz, int jz, const int descz[9],
                        int& info) {
  pxhegvd_helper<T>(uplo, m, a, ia, ja, desca, b, ib, jb, descb, w, z, iz, jz, descz, info, true);
}

#endif
