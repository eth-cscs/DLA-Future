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

#include <dlaf/common/pipeline.h>
#include <dlaf/common/round_robin.h>
#include <dlaf/communication/communicator.h>

namespace dlaf::comm {
class CommunicatorPipeline {
  using PipelineType = dlaf::common::Pipeline<Communicator>;

public:
  using ReadOnlyWrapper = typename PipelineType::ReadOnlyWrapper;
  using ReadWriteWrapper = typename PipelineType::ReadWriteWrapper;
  using ReadOnlySender = typename PipelineType::ReadOnlySender;
  using ReadWriteSender = typename PipelineType::ReadWriteSender;

  /// Create a CommunicatorPipeline by moving in the resource (it takes the
  /// ownership).
  explicit CommunicatorPipeline(Communicator comm)
      : pipeline(std::move(comm)) {}
  CommunicatorPipeline(CommunicatorPipeline &&other) = default;
  CommunicatorPipeline &operator=(CommunicatorPipeline &&other) = default;
  CommunicatorPipeline(const CommunicatorPipeline &) = delete;
  CommunicatorPipeline &operator=(const CommunicatorPipeline &) = delete;

  /// TODO: Update docs.
  /// Enqueue for exclusive read-write access to the resource.
  ///
  /// @return a sender that will become ready as soon as the previous user
  /// releases the resource.
  /// @pre valid()
  /// TODO: Rename?
  ReadWriteSender readwrite() { return pipeline.readwrite(); }

  [[deprecated("Use readwrite instead")]] ReadWriteSender operator()() {
    return pipeline.readwrite();
  }

  /// Enqueue for shared read-only access to the resource.
  ///
  /// @return a sender that will become ready as soon as the previous user
  /// releases the resource.
  /// @pre valid()
  /// TODO: Rename?
  ReadOnlySender read() { return pipeline.read(); }

  /// Create a sub pipeline to the value contained in the current
  /// CommunicatorPipeline
  ///
  /// All accesses to the sub pipeline are sequenced after previous accesses and
  /// before later accesses to the original pipeline, independently of when
  /// values are accessed in the sub pipeline.
  CommunicatorPipeline sub_pipeline() { return {pipeline.sub_pipeline()}; }

  /// Check if the pipeline is valid.
  ///
  /// @return true if the pipeline hasn't been reset, otherwise false.
  bool valid() const noexcept { return pipeline.valid(); }

  /// Reset the pipeline.
  ///
  /// @post !valid()
  void reset() noexcept { pipeline.reset(); }

private:
  CommunicatorPipeline(PipelineType &&pipeline)
      : pipeline(std::move(pipeline)) {}

  PipelineType pipeline;
};
} // namespace dlaf::comm
