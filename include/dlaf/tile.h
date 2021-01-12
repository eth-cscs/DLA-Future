//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "dlaf/blas_tile.h"
#include "dlaf/common/callable_object.h"
#include "dlaf/cublas_tile.h"

namespace dlaf {
namespace tile {

DLAF_MAKE_CALLABLE_OBJECT(gemm);
DLAF_MAKE_CALLABLE_OBJECT(hemm);
DLAF_MAKE_CALLABLE_OBJECT(her2k);
DLAF_MAKE_CALLABLE_OBJECT(herk);
DLAF_MAKE_CALLABLE_OBJECT(trsm);

}
}
