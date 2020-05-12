//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <array>

#include "dlaf/common/index2d.h"
#include "dlaf/communication/communicator.h"

namespace dlaf {
namespace comm {

/// TAG for strong-typing basic_coords
struct TAG_MPI;

/// Type used for indexes in MPI API
using IndexT_MPI = int;

/// 2D index strong-typed for MPI
using Index2D = common::Index2D<IndexT_MPI, TAG_MPI>;
/// 2D size strong-typed for MPI
using Size2D = common::Size2D<IndexT_MPI, TAG_MPI>;

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
  CommunicatorGrid(Communicator comm, IndexT_MPI rows, IndexT_MPI cols, common::Ordering ordering);

  /// @brief Create a communicator grid with dimensions specified by @p size and given @p ordering
  /// @param size with @p size[0] rows and @p size[1] columns
  /// @param comm must be valid during construction
  CommunicatorGrid(Communicator comm, const std::array<IndexT_MPI, 2>& size, common::Ordering ordering)
      : CommunicatorGrid(comm, size[0], size[1], ordering) {}

  /// Return rank in the grid with all ranks given the 2D index
  IndexT_MPI rankFullCommunicator(const Index2D& index) const noexcept {
    return common::computeLinearIndex<IndexT_MPI>(FULL_COMMUNICATOR_ORDER, index,
                                                  {grid_size_.rows(), grid_size_.cols()});
  }

  /// @brief Return the rank of the current process in the CommunicatorGrid
  /// @return the 2D coordinate representing the position in the grid
  Index2D rank() const noexcept {
    return position_;
  }

  /// @brief Return the number of rows in the grid
  Size2D size() const noexcept {
    return grid_size_;
  }

  /// @brief Return a Communicator grouping all ranks in the grid
  Communicator& fullCommunicator() noexcept {
    return full_;
  }

  /// @brief Return a Communicator grouping all ranks in the row (that includes the current process)
  Communicator& rowCommunicator() noexcept {
    return row_;
  }

  /// @brief Return a Communicator grouping all ranks in the column (that includes the current process)
  Communicator& colCommunicator() noexcept {
    return col_;
  }

protected:
  static constexpr const dlaf::common::Ordering FULL_COMMUNICATOR_ORDER{
      dlaf::common::Ordering::RowMajor};

  Communicator full_;
  Communicator row_;
  Communicator col_;

  Index2D position_;
  Size2D grid_size_ = Size2D(0, 0);
};

}
}
