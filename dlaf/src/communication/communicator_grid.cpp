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

#include "communicator_releasable.hpp"

namespace dlaf {
namespace comm {

std::array<int, 2> computeGridDims(int nranks) {
  std::array<int, 2> dimensions{0, 0};
  MPI_Dims_create(nranks, 2, dimensions.data());
  return dimensions;
}

CommunicatorGrid::CommunicatorGrid(Communicator comm, int nrows, int ncols) {
  Communicator all;
  {
    MPI_Comm mpi_grid;
    std::array<int, 2> dimensions({nrows, ncols});
    std::array<int, 2> periodicity({false, false});

    MPI_CALL(MPI_Cart_create(comm, 2, dimensions.data(), periodicity.data(), false, &mpi_grid));
    all = Communicator(mpi_grid);
  }

  // TODO manage out of grid

  row_ = CommunicatorGrid::getAxisCommunicator(0, all);
  col_ = CommunicatorGrid::getAxisCommunicator(1, all);

  std::array<int, 2> coords;
  MPI_CALL(MPI_Cart_coords(all, all.rank(), 2, coords.data()));
  position_ = common::Index2D(coords);

  release_communicator(all);
}

CommunicatorGrid::CommunicatorGrid(Communicator comm, const std::array<int, 2>& size)
    : CommunicatorGrid(comm, size[0], size[1]) {}

CommunicatorGrid::~CommunicatorGrid() noexcept(false) {
  // TODO manage out of grid

  release_communicator(row_);
  release_communicator(col_);
}

common::Index2D CommunicatorGrid::rank() const noexcept {
  return position_;
}

int CommunicatorGrid::rows() const noexcept {
  return row_.size();
}

int CommunicatorGrid::cols() const noexcept {
  return col_.size();
}

Communicator& CommunicatorGrid::row() noexcept {
  return row_;
}

Communicator& CommunicatorGrid::col() noexcept {
  return col_;
}

Communicator CommunicatorGrid::getAxisCommunicator(int axis, Communicator grid) noexcept(false) {
  MPI_Comm mpi_axis;
  std::array<int, 2> keep_axis{false, false};
  keep_axis[static_cast<size_t>(axis)] = true;

  MPI_CALL(MPI_Cart_sub(grid, keep_axis.data(), &mpi_axis));

  return Communicator(mpi_axis);
}

}
}
