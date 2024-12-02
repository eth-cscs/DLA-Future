//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <utility>
#include <vector>

#include <dlaf/auxiliary/norm/api.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/vector.h>
#include <dlaf/communication/kernels/reduce.h>
#include <dlaf/communication/sync/reduce.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

namespace dlaf::auxiliary::internal {

template <class T>
T max_element(std::vector<T>&& values) {
  DLAF_ASSERT(!values.empty(), "");
  return *std::max_element(values.begin(), values.end());
}

template <class T>
pika::execution::experimental::unique_any_sender<T> reduce_in_place(
    pika::execution::experimental::unique_any_sender<dlaf::comm::CommunicatorPipelineExclusiveWrapper>
        pcomm,
    comm::IndexT_MPI rank, MPI_Op reduce_op, pika::execution::experimental::unique_any_sender<T> value) {
  namespace ex = pika::execution::experimental;

  return std::move(value) | ex::let_value([pcomm = std::move(pcomm), rank, reduce_op](T& local) mutable {
           using dlaf::comm::internal::transformMPI;
           return std::move(pcomm) |
                  transformMPI([rank, reduce_op, &local](const dlaf::comm::Communicator& comm,
                                                         MPI_Request* req) mutable {
                    const bool is_root_rank = comm.rank() == rank;
                    void* in = is_root_rank ? MPI_IN_PLACE : &local;
                    void* out = is_root_rank ? &local : nullptr;
                    DLAF_MPI_CHECK_ERROR(MPI_Ireduce(in, out, 1, dlaf::comm::mpi_datatype<T>::type,
                                                     reduce_op, rank, comm, req));
                  }) |
                  ex::then([&local]() -> T { return local; });
         });
}

// Compute max norm of the lower triangular part of the distributed matrix
// https://en.wikipedia.org/wiki/Matrix_norm#Max_norm
//
// It just addresses tiles with elements belonging to the lower triangular part of the matrix
//
// Thanks to the nature of the max norm, it is valid for:
// - sy/he lower
// - tr lower non-unit
template <class T>
pika::execution::experimental::unique_any_sender<dlaf::BaseType<T>> Norm<
    Backend::MC, Device::CPU, T>::max_L(comm::CommunicatorGrid& comm_grid, comm::Index2D rank,
                                        Matrix<const T, Device::CPU>& matrix) {
  using namespace dlaf::matrix;
  namespace ex = pika::execution::experimental;

  using dlaf::common::make_data;
  using dlaf::common::internal::vector;
  using pika::execution::thread_stacksize;

  using dlaf::tile::internal::lange;
  using dlaf::tile::internal::lantr;

  using NormT = dlaf::BaseType<T>;

  const auto& distribution = matrix.distribution();

  DLAF_ASSERT(square_size(matrix), matrix);
  DLAF_ASSERT(square_blocksize(matrix), matrix);

  vector<ex::unique_any_sender<NormT>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  // for each local tile in the (global) lower triangular matrix, create a task that finds the max element in the tile
  for (auto tile_wrt_local : iterate_range2d(distribution.localNrTiles())) {
    auto tile_wrt_global = distribution.globalTileIndex(tile_wrt_local);

    if (tile_wrt_global.row() < tile_wrt_global.col())
      continue;

    bool is_diag = tile_wrt_global.row() == tile_wrt_global.col();
    auto norm_max_f = [is_diag](const matrix::Tile<const T, Device::CPU>& tile) noexcept -> NormT {
      if (is_diag)
        return lantr(lapack::Norm::Max, blas::Uplo::Lower, blas::Diag::NonUnit, tile);
      else
        return lange(lapack::Norm::Max, tile);
    };
    auto current_tile_max =
        matrix.read(tile_wrt_local) |
        dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(), std::move(norm_max_f));

    tiles_max.push_back(std::move(current_tile_max));
  }

  // then it is necessary to reduce max values from all ranks into a single max value for the matrix

  ex::unique_any_sender<NormT> local_max_value = ex::just(NormT{0});
  if (!tiles_max.empty())
    local_max_value =
        ex::when_all_vector(std::move(tiles_max)) |
        dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(thread_stacksize::nostack),
                                  max_element<NormT>);

  return reduce_in_place(comm_grid.full_communicator_pipeline().exclusive(),
                         comm_grid.rankFullCommunicator(rank), MPI_MAX, std::move(local_max_value));
}

template <class T>
pika::execution::experimental::unique_any_sender<dlaf::BaseType<T>> Norm<
    Backend::MC, Device::CPU, T>::max_G(comm::CommunicatorGrid& comm_grid, comm::Index2D rank,
                                        Matrix<const T, Device::CPU>& matrix) {
  using namespace dlaf::matrix;
  namespace ex = pika::execution::experimental;

  using dlaf::common::make_data;
  using dlaf::common::internal::vector;
  using pika::execution::thread_stacksize;

  using dlaf::tile::internal::lange;
  using dlaf::tile::internal::lantr;

  using NormT = dlaf::BaseType<T>;

  const auto& distribution = matrix.distribution();

  vector<ex::unique_any_sender<NormT>> tiles_max;
  tiles_max.reserve(distribution.localNrTiles().rows() * distribution.localNrTiles().cols());

  for (auto tile_wrt_local : iterate_range2d(distribution.localNrTiles())) {
    auto current_tile_max =
        dlaf::internal::whenAllLift(lapack::Norm::Max, matrix.read(tile_wrt_local)) |
        dlaf::tile::lange(dlaf::internal::Policy<Backend::MC>(thread_stacksize::nostack));

    tiles_max.push_back(std::move(current_tile_max));
  }

  // then it is necessary to reduce max values from all ranks into a single max value for the matrix

  ex::unique_any_sender<NormT> local_max_value = ex::just(NormT{0});
  if (!tiles_max.empty())
    local_max_value =
        ex::when_all_vector(std::move(tiles_max)) |
        dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(thread_stacksize::nostack),
                                  max_element<NormT>);

  return reduce_in_place(comm_grid.full_communicator_pipeline().exclusive(),
                         comm_grid.rankFullCommunicator(rank), MPI_MAX, std::move(local_max_value));
}
}
