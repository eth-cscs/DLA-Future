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

/// @file

namespace ns3c {
namespace util {

/// @brief Returns @c ceiling(@p num / @p den) for integer types.
///
/// @tparam IntType has to be an integer type
/// @pre @a num >= 0 and @a den >= 0
template <class IntType>
constexpr IntType ceilDiv(IntType num, IntType den) {
  return (num + den - 1) / den;
}

}
}
