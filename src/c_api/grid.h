//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <dlaf/communication/communicator_grid.h>
#include <unordered_map>

extern std::unordered_map<int, dlaf::comm::CommunicatorGrid> dlaf_grids;
