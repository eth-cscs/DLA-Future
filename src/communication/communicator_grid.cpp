//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/communication/communicator_grid.h"

namespace dlaf {
namespace comm {

CommunicatorGrid::CommunicatorGrid(Communicator comm, IndexT_MPI nrows, IndexT_MPI ncols,
                                   common::Ordering ordering)
    : ordering_(ordering) {
  if (nrows * ncols > comm.size())
    throw std::invalid_argument("grid is bigger than available ranks in communicator");

  bool is_in_grid = comm.rank() < nrows * ncols;

  int index_row = MPI_UNDEFINED;
  int index_col = MPI_UNDEFINED;
  int key = comm.rank();

  if (is_in_grid) {
    position_ = common::computeCoords<Index2D>(ordering_, comm.rank(), {nrows, ncols});
    index_row = position_.row();
    index_col = position_.col();
  }

  MPI_Comm mpi_full, mpi_col, mpi_row;
  MPI_CALL(MPI_Comm_split(comm, is_in_grid ? 0 : MPI_UNDEFINED, key, &mpi_full));
  MPI_CALL(MPI_Comm_split(comm, index_row, key, &mpi_row));
  MPI_CALL(MPI_Comm_split(comm, index_col, key, &mpi_col));

  if (!is_in_grid)
    return;

  grid_size_ = {nrows, ncols};

  full_ = make_communicator_managed(mpi_full);
  row_ = make_communicator_managed(mpi_row);
  col_ = make_communicator_managed(mpi_col);
}
}
}
