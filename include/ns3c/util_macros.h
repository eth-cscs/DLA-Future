//
// NS3C
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <type_traits>

#if NS3C_DOXYGEN
#define RETURN_TYPE_IF(T, V) T
#else
#define RETURN_TYPE_IF(T, V) std::enable_if_t<V, T>
#endif
