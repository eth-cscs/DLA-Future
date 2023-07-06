//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "utils.h"

#include <blas.hh>

#include <dlaf/common/assert.h>

blas::Uplo dlaf_uplo_from_char(char uplo) {
  DLAF_ASSERT(uplo == 'L' || uplo == 'l' || uplo == 'U' || uplo == 'u', uplo);

  return (uplo == 'U' or uplo == 'u') ? blas::Uplo::Upper : blas::Uplo::Lower;
}
