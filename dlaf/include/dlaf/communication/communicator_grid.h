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
/// @return std::array<int, 2> an array with the two dimensions
/// @post ret_dims[0] * ret_dims[0] == @p nranks
std::array<int, 2> computeGridDims(int nranks) noexcept(false);

/// @brief Create a communicator with a 2D Grid structure
///
/// Given a communicator, it creates communicators for rows and columns, completely independent from the
/// original one. These new communicators lifetimes management is up to the CommunicatorGrid.
///
/// If the grid size does not cover the entire set of ranks available in the original Communicator,
/// there will be ranks that will be not part of the row and column communicators. On the opposite,
/// if a grid size bigger that overfit the available number of ranks is specified, it will raise an
/// exception.
///
/// CommunicatorGrid must be destroyed before calling MPI_Finalize, to allow it releasing resources.
class CommunicatorGrid {
public:
  /// @brief Create a communicator grid @p rows x @p cols with given @p ordering
  /// @param comm must be valid during construction
  CommunicatorGrid(Communicator comm, int rows, int cols,
                   common::LeadingDimension ordering = common::LeadingDimension::Row) noexcept(false);

  /// @brief Create a communicator grid with dimensions specified by @p size and given @p ordering
  /// @param size with @p size[0] rows and @p size[1] columns
  /// @param comm must be valid during construction
  CommunicatorGrid(Communicator comm, const std::array<int, 2>& size,
                   common::LeadingDimension ordering = common::LeadingDimension::Row) noexcept(false);

  /// Release all internal resources (i.e. all/row/col Communicator s)
  ~CommunicatorGrid() noexcept(false);

  CommunicatorGrid(const CommunicatorGrid&) = delete;
  CommunicatorGrid& operator=(const CommunicatorGrid&) = delete;

  /// @brief Return the rank of the current process in the CommunicatorGrid
  /// @return a common::Index2D representing the position in the grid
  common::Index2D rank() const noexcept;

  /// @brief Return the number of rows in the grid
  int rows() const noexcept;

  /// @brief Return the number of columns in the grid
  int cols() const noexcept;

  /// @brief Return a Communicator grouping all ranks in the row (that includes the current process)
  Communicator& row() noexcept;

  /// @brief Return a Communicator grouping all ranks in the column (that includes the current process)
  Communicator& col() noexcept;

protected:
  Communicator row_;
  Communicator col_;

  bool is_in_grid_;
  common::Index2D position_;
};

}
}
