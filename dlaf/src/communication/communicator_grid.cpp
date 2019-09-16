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

std::array<int, 2> computeGridDims(int nranks) {
  std::array<int, 2> dimensions{0, 0};
  MPI_Dims_create(nranks, 2, dimensions.data());
  return dimensions;
}

CommunicatorGrid::CommunicatorGrid(Communicator comm, int nrows, int ncols,
                                   common::LeadingDimension ordering) {
  if (nrows * ncols > comm.size())
    throw std::invalid_argument("grid is bigger than available ranks in communicator");

  is_in_grid_ = comm.rank() < nrows * ncols;

  int index_row = MPI_UNDEFINED;
  int index_col = MPI_UNDEFINED;
  int key = comm.rank();

  if (is_in_grid_) {
    position_ = common::computeCoords(ordering, comm.rank(), {nrows, ncols});
    index_row = position_.row();
    index_col = position_.col();
  }

  MPI_Comm mpi_col, mpi_row;
  MPI_CALL(MPI_Comm_split(comm, index_row, key, &mpi_row));
  MPI_CALL(MPI_Comm_split(comm, index_col, key, &mpi_col));

  if (!is_in_grid_)
    return;

  row_ = Communicator(mpi_row, Communicator::managed{});
  col_ = Communicator(mpi_col, Communicator::managed{});
}

CommunicatorGrid::CommunicatorGrid(Communicator comm, const std::array<int, 2>& size,
                                   common::LeadingDimension ordering)
    : CommunicatorGrid(comm, size[0], size[1], ordering) {}

common::Index2D CommunicatorGrid::rank() const noexcept {
  return position_;
}

int CommunicatorGrid::rows() const noexcept {
  return col_.size();
}

int CommunicatorGrid::cols() const noexcept {
  return row_.size();
}

Communicator& CommunicatorGrid::row() noexcept {
  return row_;
}

Communicator& CommunicatorGrid::col() noexcept {
  return col_;
}

}
}
