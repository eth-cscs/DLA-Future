//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include "blas.hh"

#include "dlaf/common/callable_object.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_blas.h"

// TODO
#include "dlaf/lapack_tile.h"
#include "dlaf/cusolver_tile.h"

namespace dlaf {
namespace tile {

DLAF_MAKE_CALLABLE_OBJECT(potrf);

}
}

