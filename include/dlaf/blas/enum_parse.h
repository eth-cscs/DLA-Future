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

/// @file

#include <string>

#include <blas/util.hh>

namespace dlaf::internal {

inline blas::Uplo char2uplo(const char uplo_c) {
  blas::Uplo uplo;
  blas::from_string(std::string(&uplo_c, 1), &uplo);
  return uplo;
}

}
