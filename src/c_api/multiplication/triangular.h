//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <utility>

#include <blas.hh>
#include <blas/util.hh>
#include <mpi.h>

#include <pika/init.hpp>

#include <dlaf/blas/enum_parse.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <dlaf/multiplication/triangular.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/utils.h>

#include "../blacs.h"
#include "../utils.h"

template <typename T>
int triangular_multiplication(const int dlaf_context, const char side, const char uplo, const char op,
                              const char diag, const T alpha, const T* a,
                              const DLAF_descriptor dlaf_desca, T* b, const DLAF_descriptor dlaf_descb) {
  using MatrixMirrorA = dlaf::matrix::MatrixMirror<const T, dlaf::Device::Default, dlaf::Device::CPU>;
  using MatrixMirrorB = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

  DLAF_ASSERT(dlaf_desca.i == 0, dlaf_desca.i);
  DLAF_ASSERT(dlaf_desca.j == 0, dlaf_desca.j);
  DLAF_ASSERT(dlaf_descb.i == 0, dlaf_descb.i);
  DLAF_ASSERT(dlaf_descb.j == 0, dlaf_descb.j);

  pika::resume();

  auto& communicator_grid = grid_from_context(dlaf_context);

  auto layout_a = make_layout(dlaf_desca, communicator_grid);
  auto layout_b = make_layout(dlaf_descb, communicator_grid);

  dlaf::matrix::Matrix<const T, dlaf::Device::CPU> matrix_a_host(layout_a, a);
  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_b_host(layout_b, b);

  {
    MatrixMirrorA matrix_a(matrix_a_host);
    MatrixMirrorB matrix_b(matrix_b_host);

    dlaf::triangular_multiplication<dlaf::Backend::Default, dlaf::Device::Default, T>(
        communicator_grid, dlaf::internal::char2side(side), dlaf::internal::char2uplo(uplo),
        dlaf::internal::char2op(op), dlaf::internal::char2diag(diag), alpha, matrix_a.get(),
        matrix_b.get());
  }  // Destroy mirrors

  matrix_b_host.waitLocalTiles();

  pika::suspend();

  return 0;
}

#ifdef DLAF_WITH_SCALAPACK

template <typename T>
void pxtrmm(const char side, const char uplo, const char op, const char diag, const int m, const int n,
            const T alpha, const T* a, const int ia, const int ja, const int desca[9], T* b,
            const int ib, const int jb, const int descb[9]) {
  DLAF_ASSERT(desca[0] == 1, desca[0]);
  DLAF_ASSERT(descb[0] == 1, descb[0]);
  DLAF_ASSERT(ia == 1, ia);
  DLAF_ASSERT(ja == 1, ja);
  DLAF_ASSERT(ib == 1, ib);
  DLAF_ASSERT(jb == 1, jb);

  // Determine size of A based on side
  int a_size = (dlaf::internal::char2side(side) == blas::Side::Left) ? m : n;

  auto dlaf_desca = make_dlaf_descriptor(a_size, a_size, ia, ja, desca);
  auto dlaf_descb = make_dlaf_descriptor(m, n, ib, jb, descb);

  triangular_multiplication(desca[1], side, uplo, op, diag, alpha, a, dlaf_desca, b, dlaf_descb);
}

#endif
