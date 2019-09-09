//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <array>

#include "dlaf/common/index2d.h"
#include "communicator.h"

namespace dlaf {
namespace comm {

std::array<int, 2> computeGridDims(int nranks) noexcept(false);

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

  common::Index2D rank() const noexcept;

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
  common::Index2D position_;

  private:
  static Communicator getAxisCommunicator(int axis, Communicator grid) noexcept(false);
};

}
}
