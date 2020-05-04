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
                                   common::Ordering ordering) {
  if (nrows * ncols > comm.size())
    throw std::invalid_argument("grid is bigger than available ranks in communicator");

  bool is_in_grid = comm.rank() < nrows * ncols;

  IndexT_MPI index_row = MPI_UNDEFINED;
  IndexT_MPI index_col = MPI_UNDEFINED;
  IndexT_MPI key_full = MPI_UNDEFINED;
  IndexT_MPI key = comm.rank();

  comm::Size2D grid_size{nrows, ncols};
  if (is_in_grid) {
    position_ = common::computeCoords(ordering, comm.rank(), grid_size);
    key_full = common::computeLinearIndex<IndexT_MPI>(FULL_COMMUNICATOR_ORDER, position_, grid_size);
    index_row = position_.row();
    index_col = position_.col();
  }

  MPI_Comm mpi_full, mpi_col, mpi_row;
  DLAF_MPI_CALL(MPI_Comm_split(comm, is_in_grid ? 0 : MPI_UNDEFINED, key_full, &mpi_full));
  DLAF_MPI_CALL(MPI_Comm_split(comm, index_row, key, &mpi_row));
  DLAF_MPI_CALL(MPI_Comm_split(comm, index_col, key, &mpi_col));

  if (!is_in_grid)
    return;

  grid_size_ = grid_size;

  full_ = make_communicator_managed(mpi_full);
  row_ = make_communicator_managed(mpi_row);
  col_ = make_communicator_managed(mpi_col);
}
}
}
