//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <iostream>
#include <stdexcept>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf_c/desc.h>
#include <dlaf_c/utils.h>

#include "grid.h"
#include "utils.h"

struct DLAF_descriptor make_dlaf_descriptor(const int m, const int n, const int i, const int j,
                                            const int desc[9]) {
  DLAF_ASSERT(i == 1, i);
  DLAF_ASSERT(j == 1, j);

  struct DLAF_descriptor dlaf_desc = {m, n, desc[4], desc[5], desc[6], desc[7], i - 1, j - 1, desc[8]};

  return dlaf_desc;
}

std::tuple<dlaf::matrix::Distribution, dlaf::matrix::LayoutInfo> distribution_and_layout(
    const struct DLAF_descriptor dlaf_desc, dlaf::comm::CommunicatorGrid& grid) {
  dlaf::GlobalElementSize matrix_size(dlaf_desc.m, dlaf_desc.n);
  dlaf::TileElementSize block_size(dlaf_desc.mb, dlaf_desc.nb);

  dlaf::comm::Index2D src_rank_index(dlaf_desc.isrc, dlaf_desc.jsrc);

  dlaf::matrix::Distribution distribution(matrix_size, block_size, grid.size(), grid.rank(),
                                          src_rank_index);

  dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, dlaf_desc.ld);

  return std::make_tuple(distribution, layout);
}

dlaf::common::Ordering char2order(const char order) {
  return order == 'C' or order == 'c' ? dlaf::common::Ordering::ColumnMajor
                                      : dlaf::common::Ordering::RowMajor;
}

dlaf::comm::CommunicatorGrid& grid_from_context(int dlaf_context) {
  try {
    return dlaf_grids.at(dlaf_context);
  }
  catch (const std::out_of_range& e) {
    std::stringstream ss;
    ss << "[ERROR] No DLA-Future grid for context " << dlaf_context << ". ";
    ss << "Did you forget to call dlaf_create_grid() or dlaf_create_grid_from_blacs()?\n";

    std::cerr << ss.str() << std::flush;

    std::terminate();
  }
}
