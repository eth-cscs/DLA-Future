//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>
#include "dlaf/eigensolver/eigensolver/mc.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace eigensolver {

/// TODO
template <Backend backend, Device device, class T>
ReturnEigensolverType<T, device> eigensolver(blas::Uplo uplo, Matrix<T, device>& mat) {
  DLAF_ASSERT(matrix::local_matrix(mat), mat);
  DLAF_ASSERT(square_size(mat), mat);
  DLAF_ASSERT(square_blocksize(mat), mat);

  return internal::Eigensolver<backend, device, T>::call(uplo, mat);
}
}
}
