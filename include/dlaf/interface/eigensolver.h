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
void pxsyevd(char uplo, int n, T* a, int* desca, T* w, T* z, int* descz, int& info) {
  using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;
  
  // TODO: Check desca matches descz?
  utils::dlaf_check(uplo, desca, info);
  if (info == -1)
    return;
  info = -1;

  pika::resume();

  auto dlaf_setup = dlaf::interface::blacs::from_desc(desca);

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_host(dlaf_setup.distribution, dlaf_setup.layout_info, a);

  // uplo checked by dlaf_check
  auto dlaf_uplo = uplo == 'U' or uplo == 'u' ? blas::Uplo::Upper : blas::Uplo::Lower;

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> eigenvectors_host(dlaf_setup.distribution, dlaf_setup.layout_info,
                                                               z);  // Distributed eigenvectors
  auto eigenvalues_host =
      dlaf::matrix::createMatrixFromColMajor<dlaf::Device::CPU>({n, 1},
                                                                {dlaf_setup.distribution.blockSize().rows(), 1}, n,
                                                                w);  // Local eigenvectors

  {
    // Create matrix mirrors
    MatrixMirror matrix(matrix_host);
    MatrixMirror eigenvalues(eigenvalues_host);
    MatrixMirror eigenvectors(eigenvectors_host);

    // TODO: Use dlaf_uplo instead of hard-coded blas::Uplo::Lower
    // TODO: blas::Uplo::Upper is not yet supported
    dlaf::eigensolver::eigensolver<dlaf::Backend::Default, dlaf::Device::Default, T>(dlaf_setup.communicator_grid,
                                                                                     blas::Uplo::Lower,
                                                                                     matrix.get(),
                                                                                     eigenvalues.get(),
                                                                                     eigenvectors.get());
  }  // Destroy mirrors

  pika::suspend();

  info = 0;
}

extern "C" void pssyevd(char uplo, int n, float* a, int* desca, float* w, float* z, int* descz,
                        int& info);

extern "C" void pdsyevd(char uplo, int n, double* a, int* desca, double* w, double* z, int* descz,
                        int& info);

}
