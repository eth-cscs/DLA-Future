//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

/// @file

#pragma once

#include <utility>

#include <dlaf/common/index2d.h>
#include <dlaf/common/pipeline.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_type.h>
#include <dlaf/communication/index.h>

namespace dlaf::comm {
namespace internal {
constexpr const dlaf::common::Ordering FULL_COMMUNICATOR_ORDER{dlaf::common::Ordering::RowMajor};
}

using CommunicatorPipelineSharedWrapper = typename common::Pipeline<Communicator>::ReadOnlyWrapper;
using CommunicatorPipelineExclusiveWrapper = typename common::Pipeline<Communicator>::ReadWriteWrapper;
using CommunicatorPipelineSharedSender = typename common::Pipeline<Communicator>::ReadOnlySender;
using CommunicatorPipelineExclusiveSender = typename common::Pipeline<Communicator>::ReadWriteSender;

/// A CommunicatorPipeline provides pipelined access to a communicator, as well as metadata about the
/// communicator.
///
/// A CommunicatorPipeline uses a Pipeline to provide shared or exclusive access through senders to the
/// underlying communicator.  @tparam CT signals whether the contained communicator is a row, column, or
/// full grid communicator.
template <CommunicatorType CT>
class CommunicatorPipeline {
  using PipelineType = dlaf::common::Pipeline<Communicator>;

public:
  using SharedWrapper = typename PipelineType::ReadOnlyWrapper;
  using ExclusiveWrapper = typename PipelineType::ReadWriteWrapper;
  using SharedSender = typename PipelineType::ReadOnlySender;
  using ExclusiveSender = typename PipelineType::ReadWriteSender;

  static constexpr CommunicatorType communicator_type = CT;

  /// Create a CommunicatorPipeline by moving in the resource (it takes the
  /// ownership).
  explicit CommunicatorPipeline(Communicator comm, Index2D rank = {-1, -1}, Size2D size = {-1, -1})
      : rank_(comm.rank()), size_(comm.size()), rank_2d_(std::move(rank)), size_2d_(std::move(size)),
        pipeline_(std::move(comm)) {}
  CommunicatorPipeline(CommunicatorPipeline&& other) = default;
  CommunicatorPipeline& operator=(CommunicatorPipeline&& other) = default;
  CommunicatorPipeline(const CommunicatorPipeline&) = delete;
  CommunicatorPipeline& operator=(const CommunicatorPipeline&) = delete;

  /// Return the rank of the current process in the CommunicatorPipeline.
  IndexT_MPI rank() const noexcept {
    return rank_;
  }

  /// Return the size of the grid.
  IndexT_MPI size() const noexcept {
    return size_;
  }

  /// Return the 2D rank of the current process in the grid that this pipeline
  /// belongs to.
  Index2D rank_2d() const noexcept {
    return rank_2d_;
  }

  /// Return the 2D size of the current process in the grid that this pipeline
  /// belongs to.
  Size2D size_2d() const noexcept {
    return size_2d_;
  }

  /// Return rank in the grid with all ranks given the 2D index.
  IndexT_MPI rank_full_communicator(const Index2D& index) const noexcept {
    return common::computeLinearIndex<IndexT_MPI>(internal::FULL_COMMUNICATOR_ORDER, index,
                                                  {size_2d_.rows(), size_2d_.cols()});
  }

  /// Enqueue for exclusive read-write access to the Communicator.
  ///
  /// @return a sender that gives exclusive access to the Communicator as soon as previous accesses are
  /// released.
  /// @pre valid()
  ExclusiveSender exclusive() {
    return pipeline_.readwrite();
  }

  /// Enqueue for shared read-only access to the Communicator.
  ///
  /// @return a sender that gives shared access to the Communicator as soon as previous read-write
  /// accesses are released.
  /// @pre valid()
  SharedSender shared() {
    return pipeline_.read();
  }

  /// Create a sub pipeline to the value contained in the current CommunicatorPipeline
  ///
  /// All accesses to the sub pipeline are sequenced after previous accesses and
  /// before later accesses to the original pipeline, independently of when
  /// values are accessed in the sub pipeline.
  CommunicatorPipeline sub_pipeline() {
    return {pipeline_.sub_pipeline(), rank_2d_, size_2d_, rank_, size_};
  }

  /// Check if the pipeline is valid.
  ///
  /// @return true if the pipeline hasn't been reset, otherwise false.
  bool valid() const noexcept {
    return pipeline_.valid();
  }

  /// Reset the pipeline.
  ///
  /// @post !valid()
  void reset() noexcept {
    pipeline_.reset();
  }

  /// Prints information about the CommunicationPipeline.
  friend std::ostream& operator<<(std::ostream& out, const CommunicatorPipeline& pipeline) {
    return out << "2d rank=" << pipeline.rank_2d_ << ", 2d size=" << pipeline.size_2d_
               << ", rank=" << pipeline.rank_ << ", size=" << pipeline.size_;
  }

private:
  CommunicatorPipeline(PipelineType&& pipeline, Index2D rank_2d, Size2D size_2d, IndexT_MPI rank,
                       IndexT_MPI size)
      : rank_(rank), size_(size), rank_2d_(std::move(rank_2d)), size_2d_(std::move(size_2d)),
        pipeline_(std::move(pipeline)) {}

  IndexT_MPI rank_{-1};
  IndexT_MPI size_{-1};
  Index2D rank_2d_{-1, -1};
  Size2D size_2d_{-1, -1};
  PipelineType pipeline_{};
};
}  // namespace dlaf::comm
