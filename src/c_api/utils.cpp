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

#include <dlaf/communication/communicator_grid.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

struct DLAF_descriptor make_dlaf_descriptor(int m, int n, int i, int j, int* desc) {
  struct DLAF_descriptor dlaf_desc = {m, n, desc[4], desc[5], desc[6], desc[7], i - 1, j - 1, desc[8]};
  return dlaf_desc;
}

std::tuple<dlaf::matrix::Distribution, dlaf::matrix::LayoutInfo> distribution_and_layout(
    struct DLAF_descriptor dlaf_desc, dlaf::comm::CommunicatorGrid& grid) {
  dlaf::GlobalElementSize matrix_size(dlaf_desc.m, dlaf_desc.n);
  dlaf::TileElementSize block_size(dlaf_desc.mb, dlaf_desc.nb);

  dlaf::comm::Index2D src_rank_index(dlaf_desc.isrc, dlaf_desc.jsrc);

  dlaf::matrix::Distribution distribution(matrix_size, block_size, grid.size(), grid.rank(),
                                          src_rank_index);

  dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, dlaf_desc.ld);

  return std::make_tuple(distribution, layout);
}
