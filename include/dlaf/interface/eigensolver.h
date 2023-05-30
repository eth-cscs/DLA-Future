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

#include <dlaf/interface/cholesky.h>

#include <pika/init.hpp>
#include <dlaf/eigensolver/eigensolver.h>
#include <dlaf/interface/blacs.h>
#include <dlaf/interface/utils.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>

namespace dlaf::interface {

template <typename T>
using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

template <typename T>
void pxsyevd(char uplo, int n, T* a, int* desca, T* w, T* z, int* descz, int& info) {
  // TODO: Check desca matches descz?
  utils::dlaf_check(uplo, desca, info);
  if (info == -1)
    return;
  info = -1;

  pika::resume();

  auto [distribution, layout, comm_grid] = blacs::dlaf_setup_from_desc(desca);

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_host(distribution, layout, a);

  // uplo checked by dlaf_check
  auto dlaf_uplo = uplo == 'U' or uplo == 'u' ? blas::Uplo::Upper : blas::Uplo::Lower;

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> eigenvectors_host(distribution, layout,
                                                               z);  // Distributed eigenvectors
  auto eigenvalues_host =
      dlaf::matrix::createMatrixFromColMajor<dlaf::Device::CPU>({n, 1},
                                                                {distribution.blockSize().rows(), 1}, n,
                                                                w);  // Local eigenvectors

  {
    // Create matrix mirrors
    MatrixMirror<T> matrix(matrix_host);
    MatrixMirror<T> eigenvalues(eigenvalues_host);
    MatrixMirror<T> eigenvectors(eigenvectors_host);

    // WARN: Hard-coded to LOWER, use dlaf_uplo instead
    dlaf::eigensolver::eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(comm_grid,
                                                                                     blas::Uplo::Lower,
                                                                                     matrix.get(),
                                                                                     eigenvalues.get(),
                                                                                     eigenvectors.get());
  }  // Destroy mirrors

  eigenvalues_host.waitLocalTiles();

  pika::suspend();

  info = 0;
}

extern "C" void pssyevd(char uplo, int n, float* a, int* desca, float* w, float* z, int* descz,
                        int& info);

extern "C" void pdsyevd(char uplo, int n, double* a, int* desca, double* w, double* z, int* descz,
                        int& info);

}
