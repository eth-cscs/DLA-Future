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

namespace ns3c {
namespace util {

/// \brief Returns ceiling(num/den) for integer types.
/// IntType has to be an integer type.
/// Precondition: num >= 0 and den >= 0.
template <class IntType>
constexpr IntType ceilDiv(IntType num, IntType den) {
  return (num + den - 1) / den;
}

}
}
