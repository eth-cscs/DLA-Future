//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "test_grid_c_api_wrapper.h"

#include <dlaf_c/grid.h>

char C_grid_ordering(MPI_Comm comm, int nprow, int npcol, int myprow, int mypcol) {
  return grid_ordering(comm, nprow, npcol, myprow, mypcol);
}
