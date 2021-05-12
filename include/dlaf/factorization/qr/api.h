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

#include "dlaf/types.h"

namespace dlaf {
namespace factorization {
namespace internal {

template <Backend backend, Device device, class T>
struct QR {};

template <Backend backend, Device device, class T>
struct QR_Tfactor {};

}
}
}
