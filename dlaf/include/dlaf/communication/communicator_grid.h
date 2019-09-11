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
#include "dlaf/communication/communicator.h"

namespace dlaf {
namespace comm {

/// @brief Compute valid 2D grid dimensions for a given number of ranks
///
/// @return std::array<int, 2> an array with the two dimensions
/// @post ret_dims[0] * ret_dims[0] == @p nranks
std::array<int, 2> computeGridDims(int nranks) noexcept(false);


/// @brief Create a communicator with a 2D Grid structure
///
/// It creates internal communicators, e.g. for row and column communication, and manages their lifetimes.
/// CommunicatorGrid must be destroyed before calling MPI_Finalize, allowing it to release resources.
class CommunicatorGrid {
  public:
  /// @brief Create a grid @p rows x @p cols
  ///
  /// @p comm must be valid during construction
  CommunicatorGrid(Communicator comm, int rows, int cols) noexcept(false);

  /// @brief Create a grid with dimensions specified by @p size
  ///
  /// @p size[0] rows and @p size[1] columns
  ///
  /// @p comm must be valid during construction
  CommunicatorGrid(Communicator comm, const std::array<int, 2> & size) noexcept(false);

  /// Release all internal resources (i.e. all/row/col Communicator s)
  ~CommunicatorGrid() noexcept(false);

  CommunicatorGrid(const CommunicatorGrid &) = delete;
  CommunicatorGrid & operator=(const CommunicatorGrid &) = delete;

  /// @brief Return the rank of the current process in the CommunicatorGrid
  ///
  /// @return a common::Index2D representing the position in the grid
  common::Index2D rank() const noexcept;

  /// @brief Return the number of rows in the grid
  int rows() const noexcept;

  /// @brief Return the number of columns in the grid
  int cols() const noexcept;

  /// @brief Return a Communicator grouping all ranks in the grid
  Communicator & all() noexcept;
  /// @brief Return a Communicator grouping all ranks in the row (that includes the current process)
  Communicator & row() noexcept;
  /// @brief Return a Communicator grouping all ranks in the col (that includes the current process)
  Communicator & col() noexcept;

  protected:
  Communicator all_;
  Communicator row_;
  Communicator col_;

  common::Index2D position_;

  /// @brief Given a Communicator with grid structure, it returns the Communicator along requested @p axis
  static Communicator getAxisCommunicator(int axis, Communicator grid) noexcept(false);
};

}
}
