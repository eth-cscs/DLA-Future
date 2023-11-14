//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <dlaf/common/index2d.h>

namespace dlaf::comm {
enum class CommunicatorType { Row, Col, Full };

constexpr CommunicatorType coord_to_communicator_type(const Coord rc) {
  return rc == Coord::Row ? CommunicatorType::Row : CommunicatorType::Col;
}
} // namespace dlaf::comm
