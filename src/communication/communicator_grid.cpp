//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cstddef>
#include <utility>
#include <vector>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/common/with_result_of.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/communication/error.h>
#include <dlaf/communication/index.h>
#include <dlaf/sender/transform_mpi.h>

namespace dlaf {
namespace comm {

CommunicatorGrid::CommunicatorGrid(Communicator comm, IndexT_MPI nrows, IndexT_MPI ncols,
                                   common::Ordering ordering, std::size_t npipelines) {
  DLAF_ASSERT((nrows * ncols) <= comm.size(), nrows, ncols, comm.size());

  bool is_in_grid = comm.rank() < nrows * ncols;

  IndexT_MPI index_row = MPI_UNDEFINED;
  IndexT_MPI index_col = MPI_UNDEFINED;
  IndexT_MPI key_full = MPI_UNDEFINED;
  IndexT_MPI key = comm.rank();

  comm::Size2D grid_size{nrows, ncols};
  if (is_in_grid) {
    position_ = common::computeCoords(ordering, comm.rank(), grid_size);
    key_full =
        common::computeLinearIndex<IndexT_MPI>(internal::FULL_COMMUNICATOR_ORDER, position_, grid_size);
    index_row = position_.row();
    index_col = position_.col();
  }

  MPI_Comm mpi_full, mpi_col, mpi_row;
  DLAF_MPI_CHECK_ERROR(MPI_Comm_split(comm, is_in_grid ? 0 : MPI_UNDEFINED, key_full, &mpi_full));
  DLAF_MPI_CHECK_ERROR(MPI_Comm_split(comm, index_row, key, &mpi_row));
  DLAF_MPI_CHECK_ERROR(MPI_Comm_split(comm, index_col, key, &mpi_col));

  if (!is_in_grid)
    return;

  grid_size_ = grid_size;

  full_ = make_communicator_managed(mpi_full);
  row_ = make_communicator_managed(mpi_row);
  col_ = make_communicator_managed(mpi_col);

  using dlaf::internal::WithResultOf;

  full_pipelines_ = RoundRobinPipeline<CommunicatorType::Full>(
      npipelines, WithResultOf([&]() {
        return CommunicatorPipeline<CommunicatorType::Full>{full_.clone(), position_, grid_size_};
      }));
  row_pipelines_ = RoundRobinPipeline<CommunicatorType::Row>(
      npipelines, WithResultOf([&]() {
        return CommunicatorPipeline<CommunicatorType::Row>{row_.clone(), position_, grid_size_};
      }));
  col_pipelines_ = RoundRobinPipeline<CommunicatorType::Col>(
      npipelines, WithResultOf([&]() {
        return CommunicatorPipeline<CommunicatorType::Col>{col_.clone(), position_, grid_size_};
      }));
}

void CommunicatorGrid::wait_all_communicators() {
  using pika::execution::experimental::drop_value;
  using pika::execution::experimental::unique_any_sender;
  using pika::execution::experimental::when_all_vector;
  using pika::this_thread::experimental::sync_wait;

  constexpr auto barrier = [](const Communicator& comm, MPI_Request* req) {
    DLAF_MPI_CHECK_ERROR(MPI_Ibarrier(comm, req));
  };

  std::vector<unique_any_sender<>> senders;
  senders.reserve(3 * num_pipelines());
  for (std::size_t i = 0; i < num_pipelines(); ++i) {
    senders.push_back(full_communicator_pipeline().exclusive() | internal::transformMPI(barrier));
    senders.push_back(row_communicator_pipeline().exclusive() | internal::transformMPI(barrier));
    senders.push_back(col_communicator_pipeline().exclusive() | internal::transformMPI(barrier));
  }
  sync_wait(when_all_vector(std::move(senders)));
}
}  // namespace comm
}  // namespace dlaf
