//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/permutations/general/invert_and_copy.h"

namespace dlaf::permutations::internal {

template <Coord C>
void invertAndCopyArr(const matrix::Distribution& dist, const SizeType* in_ptr, SizeType* out_ptr) {
  comm::IndexT_MPI this_rank = dist.rankIndex().get<C>();
  SizeType n = dist.size().get<C>();
  SizeType nb = dist.blockSize().get<C>();

  for (SizeType i = 0; i < n; ++i) {
    SizeType in_idx = in_ptr[i];
    if (dist.rankGlobalElement<C>(in_idx) == this_rank) {
      SizeType i_loc_el =
          dist.localTileFromGlobalElement<C>(in_idx) * nb + dist.tileElementFromGlobalElement<C>(in_idx);
      out_ptr[i_loc_el] = i;
    }
  }
}

DLAF_CPU_INVERT_AND_COPY_ETI(, Coord::Row);
DLAF_CPU_INVERT_AND_COPY_ETI(, Coord::Col);

}
