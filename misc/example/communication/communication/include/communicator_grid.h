#pragma once

#include <array>

#include "communicator.h"

namespace dlaf {
namespace comm {

std::array<int, 2> computeGridDims(int nranks) noexcept(false);

class Index2D {
  public:
  Index2D() noexcept;
  Index2D(const std::array<int, 2> & coords) noexcept;
  Index2D(int row, int col) noexcept;

  int row() const noexcept;
  int col() const noexcept;

  protected:
  int row_;
  int col_;
};

/// Input communicator lifetime should be granted for the constructor
/// CommunicatorGrid should go out of scope before the call to MPI_Finalize
/// CommunicatorGrid is responsible for releasing created communicators
class CommunicatorGrid {
  public:
  CommunicatorGrid(Communicator comm, int rows, int cols) noexcept(false);
  CommunicatorGrid(Communicator comm, const std::array<int, 2> & size) noexcept(false);

  ~CommunicatorGrid() noexcept(false);

  // non-copyable
  CommunicatorGrid(const CommunicatorGrid &) = delete;
  CommunicatorGrid & operator=(const CommunicatorGrid &) = delete;

  Index2D rank() const noexcept;

  int rows() const noexcept;
  int cols() const noexcept;

  Communicator & all() noexcept;
  Communicator & row() noexcept;
  Communicator & col() noexcept;

  protected:
  Communicator all_;
  Communicator row_;
  Communicator col_;

  int rows_;
  int cols_;
  Index2D position_;

  private:
  static Communicator getAxisCommunicator(int axis, Communicator grid) noexcept(false);
};

}
}
