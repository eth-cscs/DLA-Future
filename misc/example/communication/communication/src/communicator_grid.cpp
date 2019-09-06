#include "communicator_grid.h"

#include "communicator_releasable.hpp"

namespace dlaf {
namespace comm {

std::array<int, 2> computeGridDims(int nranks) {
  std::array<int, 2> dimensions {0, 0};
  MPI_Dims_create(nranks, 2, dimensions.data());
  return dimensions;
}


Index2D::Index2D() noexcept : row_(-1), col_(-1) {}
Index2D::Index2D(const std::array<int, 2> & coords) noexcept : row_(coords[0]), col_(coords[1]) {}
Index2D::Index2D(int row, int col) noexcept : row_(row), col_(col) {}
int Index2D::row() const noexcept { return row_; }
int Index2D::col() const noexcept { return col_; }


CommunicatorGrid::CommunicatorGrid(Communicator comm, int nrows, int ncols)
: rows_(nrows), cols_(ncols) {
  {
    MPI_Comm mpi_grid;
    std::array<int, 2> dimensions({rows_, cols_});
    std::array<int, 2> periodicity({false, false});

    MPI_CALL(MPI_Cart_create(
      static_cast<MPI_Comm>(comm),
      2,
      dimensions.data(),
      periodicity.data(),
      0,
      &mpi_grid));
    all_ = Communicator(mpi_grid);
  }

  row_ = CommunicatorGrid::getAxisCommunicator(0, all_);
  col_ = CommunicatorGrid::getAxisCommunicator(1, all_);

  // TODO set cart coords
  std::array<int, 2> coords;
  MPI_CALL(MPI_Cart_coords(
    static_cast<MPI_Comm>(all_),
    all_.rank(),
    2,
    coords.data()
  ));
  position_ = Index2D(coords);
}

CommunicatorGrid::CommunicatorGrid(Communicator comm, const std::array<int, 2> & size)
: CommunicatorGrid(comm, size[0], size[1]) {}

CommunicatorGrid::~CommunicatorGrid() noexcept(false) {
  release_communicator(row_);
  release_communicator(col_);
  release_communicator(all_);
}

Index2D CommunicatorGrid::rank() const noexcept { return position_; }
int CommunicatorGrid::rows() const noexcept { return rows_; }
int CommunicatorGrid::cols() const noexcept { return cols_; }
Communicator & CommunicatorGrid::all() noexcept { return all_; }
Communicator & CommunicatorGrid::row() noexcept { return row_; }
Communicator & CommunicatorGrid::col() noexcept { return col_; }

Communicator CommunicatorGrid::getAxisCommunicator(int axis, Communicator grid) noexcept(false) {
  MPI_Comm mpi_axis;
  std::array<int, 2> keep_axis{false, false};
  keep_axis[static_cast<size_t>(axis)] = true;

  MPI_CALL(MPI_Cart_sub(
    static_cast<MPI_Comm>(grid),
    keep_axis.data(),
    &mpi_axis));

  return Communicator(mpi_axis);
}

}
}
