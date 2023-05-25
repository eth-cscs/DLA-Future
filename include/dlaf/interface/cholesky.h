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

#include <dlaf/factorization/cholesky.h>
#include <dlaf/interface/blacs.h>
#include <dlaf/interface/utils.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_mirror.h>
#include <pika/init.hpp>

namespace dlaf::interface{

template <typename T>
using MatrixMirror = dlaf::matrix::MatrixMirror<T, dlaf::Device::Default, dlaf::Device::CPU>;

template <typename T>
void pxpotrf(char uplo, int n, T* a, int ia, int ja, int* desca, int& info){
  utils::dlaf_check(uplo, desca, info);
  if(info == -1) return;
  
  pika::resume();

  auto dlaf_uplo = (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;

  auto [distribution, layout, communicator_grid] = blacs::dlaf_setup_from_desc(desca);

  dlaf::matrix::Matrix<T, dlaf::Device::CPU> matrix_host(std::move(distribution), layout, a);

  {
    MatrixMirror<T> matrix(matrix_host);
    
    dlaf::factorization::cholesky<dlaf::Backend::Default, dlaf::Device::Default, T>(communicator_grid, dlaf_uplo, matrix.get());
  } // Destroy mirror
  
  matrix_host.waitLocalTiles();

  pika::suspend();
  
  info = 0;
}

extern "C" void pdpotrf(char uplo, int n, double* a, int ia, int ja, int* desca, int& info);

extern "C" void pspotrf(char uplo, int n, float* a, int ia, int ja, int* desca, int& info);

}
