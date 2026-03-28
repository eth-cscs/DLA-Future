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

/// @file

#include <string>

#include <blas/util.hh>

namespace dlaf::internal {

inline blas::Side char2side(const char side_c) {
  blas::Side side;
  blas::from_string(std::string(&side_c, 1), &side);
  return side;
}

inline blas::Op char2op(const char op_c) {
  blas::Op op;
  blas::from_string(std::string(&op_c, 1), &op);
  return op;
}

inline blas::Diag char2diag(const char diag_c) {
  blas::Diag diag;
  blas::from_string(std::string(&diag_c, 1), &diag);
  return diag;
}

inline blas::Uplo char2uplo(const char uplo_c) {
  blas::Uplo uplo;
  blas::from_string(std::string(&uplo_c, 1), &uplo);
  return uplo;
}

}
