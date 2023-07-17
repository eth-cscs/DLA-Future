//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/layout_info.h>

std::tuple<dlaf::matrix::Distribution, dlaf::matrix::LayoutInfo> distribution_and_layout(
    struct DLAF_descriptor dlaf_desc, dlaf::comm::CommunicatorGrid& grid);
