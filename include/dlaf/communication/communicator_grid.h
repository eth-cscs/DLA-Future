//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <array>

#include <dlaf/common/index2d.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/tune.h>

namespace dlaf {
namespace comm {

/// Create a communicator with a 2D Grid structure.
///
/// Given a communicator, it creates communicators for rows and columns,
/// completely independent from the original one. These new communicators
/// lifetimes management is up to the CommunicatorGrid.
///
/// If the grid size does not cover the entire set of ranks available in the
/// original Communicator, there will be ranks that will be not part of the row
/// and column communicators. On the opposite, if a grid size bigger that
/// overfit the available number of ranks is specified, it will raise an
/// exception.
///
/// CommunicatorGrid must be destroyed before calling MPI_Finalize, to allow it
/// releasing resources.
class CommunicatorGrid {
public:
  /// Create a communicator grid @p rows x @p cols with given @p ordering.
  /// @param comm must be valid during construction.
  CommunicatorGrid(Communicator comm, IndexT_MPI rows, IndexT_MPI cols,
                   common::Ordering ordering,
                   std::size_t npipelines =
                       getTuneParameters().communicator_grid_num_pipelines);

  /// Create a communicator grid with dimensions specified by @p size and given
  /// @p ordering.
  /// @param size with @p size[0] rows and @p size[1] columns,
  /// @param comm must be valid during construction.
  CommunicatorGrid(Communicator comm, const std::array<IndexT_MPI, 2> &size,
                   common::Ordering ordering,
                   std::size_t npipelines =
                       getTuneParameters().communicator_grid_num_pipelines)
      : CommunicatorGrid(comm, size[0], size[1], ordering, npipelines) {}

  CommunicatorGrid(CommunicatorGrid &&) = default;
  CommunicatorGrid &operator=(CommunicatorGrid &&) = default;

  /// Return rank in the grid with all ranks given the 2D index.
  IndexT_MPI rankFullCommunicator(const Index2D &index) const noexcept {
    return common::computeLinearIndex<IndexT_MPI>(
        internal::FULL_COMMUNICATOR_ORDER, index, {grid_size_.rows(), grid_size_.cols()});
  }

  std::size_t num_pipelines() const noexcept {
    DLAF_ASSERT(full_pipelines_.size() == row_pipelines_.size(),
                full_pipelines_.size(), row_pipelines_.size());
    DLAF_ASSERT(full_pipelines_.size() == col_pipelines_.size(),
                full_pipelines_.size(), col_pipelines_.size());
    return full_pipelines_.size();
  }

  /// Return the rank of the current process in the CommunicatorGrid.
  ///
  /// @return the 2D coordinate representing the position in the grid
  Index2D rank() const noexcept { return position_; }

  /// Return the number of rows in the grid.
  Size2D size() const noexcept { return grid_size_; }

  /// Return a Communicator grouping all ranks in the grid.
  Communicator &fullCommunicator() noexcept { return full_; }

  /// Return a Communicator grouping all ranks in the row (that includes the
  /// current process).
  Communicator &rowCommunicator() noexcept { return row_; }

  /// Return a Communicator grouping all ranks in the column (that includes the
  /// current process).
  Communicator &colCommunicator() noexcept { return col_; }

  Communicator &subCommunicator(Coord cr) noexcept {
    if (cr == Coord::Row)
      return row_;
    else
      return col_;
  }

  /// Return a pipeline to a Communicator grouping all ranks in the grid.
  CommunicatorPipeline full_communicator_pipeline() {
    return full_pipelines_.nextResource().sub_pipeline();
  }

  /// Return a pipeline to a Communicator grouping all ranks in the row (that
  /// includes the current process).
  CommunicatorPipeline row_communicator_pipeline() {
    return row_pipelines_.nextResource().sub_pipeline();
  }

  /// Return a pipeline to a Communicator grouping all ranks in the col (that
  /// includes the current process).
  CommunicatorPipeline col_communicator_pipeline() {
    return col_pipelines_.nextResource().sub_pipeline();
  }

  CommunicatorPipeline communicator_pipeline(Coord cr) {
    if (cr == Coord::Row)
      return row_communicator_pipeline();
    else
      return col_communicator_pipeline();
  }

  /// Prints information about the CommunicationGrid.
  friend std::ostream &operator<<(std::ostream &out,
                                  const CommunicatorGrid &grid) {
    return out << "position=" << grid.position_ << ", size=" << grid.grid_size_;
  }

protected:
  Communicator full_;
  Communicator row_;
  Communicator col_;

  using RoundRobinPipeline = dlaf::common::RoundRobin<CommunicatorPipeline>;

  RoundRobinPipeline full_pipelines_;
  RoundRobinPipeline row_pipelines_;
  RoundRobinPipeline col_pipelines_;

  Index2D position_;
  Size2D grid_size_ = Size2D(0, 0);
};

} // namespace comm
} // namespace dlaf
