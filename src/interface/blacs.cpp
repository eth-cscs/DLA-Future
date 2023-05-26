//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <dlaf/interface/blacs.h>

#include <dlaf/communication/error.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/layout_info.h>
#include <dlaf/types.h>

#include <mpi.h>

namespace dlaf::interface::blacs{
  static int get_grid_context(int* desc){
    return desc[1]; // BLACS context
  }

  static int get_communicator_context(const int grid_context){
    int communicator_context;
    // SGET_BLACSCONTXT == 10
    Cblacs_get(grid_context, 10, &communicator_context);
    return communicator_context;
  }

  static MPI_Comm get_communicator(const int grid_context){
    int communicator_context = get_communicator_context(grid_context);
    MPI_Comm communicator = Cblacs2sys_handle(communicator_context);
    return communicator;
  }

std::tuple<dlaf::matrix::Distribution, dlaf::matrix::LayoutInfo, dlaf::comm::CommunicatorGrid> dlaf_setup_from_desc(int * desc){
  // Matrix sizes
  int m = desc[2];
  int n = desc[3];
  int mb = desc[4];
  int nb = desc[5];

  auto grid_context = get_grid_context(desc);

  MPI_Comm communicator = get_communicator(grid_context);
  dlaf::comm::Communicator world(communicator);
  DLAF_MPI_CHECK_ERROR(MPI_Barrier(world));

  int dims[2] = {0, 0};
  int coords[2] = {-1, -1};

  Cblacs_gridinfo(grid_context, &dims[0], &dims[1], &coords[0], &coords[1]);

  dlaf::comm::CommunicatorGrid communicator_grid(world, dims[0], dims[1], dlaf::common::Ordering::RowMajor);

  dlaf::GlobalElementSize matrix_size(m, n);
  dlaf::TileElementSize block_size(mb, nb);
  
  dlaf::comm::Index2D src_rank_index(0, 0); // TODO: Get from BLACS?
  
  dlaf::matrix::Distribution distribution(matrix_size, block_size, communicator_grid.size(), communicator_grid.rank(), src_rank_index);
  
  const int lld = desc[8]; // Local leading dimension
  dlaf::matrix::LayoutInfo layout = colMajorLayout(distribution, lld);

  return std::make_tuple(distribution, layout, communicator_grid);
}


}
