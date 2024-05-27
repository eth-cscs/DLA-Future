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

#include <cstddef>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/index2d.h>
#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/index.h>
#include <dlaf/communication/message.h>
#include <dlaf/communication/rdma.h>
#include <dlaf/eigensolver/tridiag_solver/index_manipulation.h>
#include <dlaf/lapack/gpu/lacpy.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/copy_tile.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/distribution_extensions.h>
#include <dlaf/matrix/index.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/matrix_ref.h>
#include <dlaf/permutations/general/api.h>
#include <dlaf/permutations/general/perms.h>
#include <dlaf/schedulers.h>
#include <dlaf/sender/policy.h>
#include <dlaf/sender/transform.h>
#include <dlaf/sender/transform_mpi.h>
#include <dlaf/sender/when_all_lift.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

namespace dlaf::permutations::internal {

// Applies the permutation index `perm_arr` to a portion of the columns/rows(depends on coord) [1] of an
// input submatrix [2] and saves the result into a subregion [3] of an output submatrix [4].
//
// Example column permutations with `perm_arr = [8, 2, 5]`:
//
//          2     5     8      out_begin
//     ┌────────────────────┐    ┌─────┬─────────────┐
//     │  in_offset      in │    │     │         out │
//     │ │                  │    │     └─►┌─┬─┬─┐    │
//     │◄┘ ┌─┐   ┌─┐   ┌─┐  │    │        │c│a│b│    │
//     │   │a│   │b│   │c│  │    │    ┌──►│ │ │ │    │
//     │   │ │ ┌►│ │   │ │◄─┼────┼──┐ │   └─┴─┴─┘    │
//     │   └─┘ │ └─┘   └─┘  │    │  │ │      ▲       │
//     │       │            │    │sz.rows()  │       │
//     │      sz.rows()     │    │           └─      │
//     │                    │    │          sz.cols()│
//     └────────────────────┘    └───────────────────┘
//
// Example row permutations with `perm_arr = [3, 1]`:
//
//             ┌─── in_offset
//             │                   out_begin
//     ┌───────▼────────────┐    ┌──┬────────────────┐
//     │                 in │    │  │            out │
//     │       ┌─────┐      │    │  └►┌─────┐        │
//   1 │       │  a  │      │    │    │  b  │        │
//     │       └─────┘      │    │    ├─────┤ ◄─┐    │
//     │                    │    │    │  a  │   │    │
//     │       ┌─────┐      │    │    └──▲──┘   │    │
//   3 │       │  b  │      │    │       │      │    │
//     │       └──▲──┘      │    │ sz.cols()    │    │
//     │          │         │    │          sz.rows()│
//     └──────────┼─────────┘    └───────────────────┘
//                │
//             sz.cols()
//
//
// [1]: The portion of each input column or row is defined by the interval [in_offset, in_offset +
//      sz.col()) or the interval [in_offset, in_offset + sz.row()) respectively.
// [2]: The input submatrix is defined by `begin_tiles`, `ld_tiles`, `distr` and `in_tiles`
// [3]: The subregion is defined by `begin` and `sz`
// [4]: The output submatrix is defined by `begin_tiles`, `ld_tiles`, `distr` and `out_tiles`
template <class T, Coord C>
void applyPermutationOnCPU(
    const SizeType i_perm, const matrix::Distribution& subm_dist, const SizeType* perm_arr,
    const std::vector<matrix::internal::TileAsyncRwMutexReadOnlyWrapper<T, Device::CPU>>& in_tiles_fut,
    const std::vector<matrix::Tile<T, Device::CPU>>& out_tiles) {
  namespace dist_extra = dlaf::matrix::internal::distribution;

  constexpr auto OC = orthogonal(C);

  const SizeType perm_from = perm_arr[i_perm];
  const SizeType perm_to = i_perm;

  const SizeType j_in_lc = subm_dist.local_tile_from_local_element<C>(perm_from);
  const SizeType j_out_lc = subm_dist.local_tile_from_local_element<C>(perm_to);

  const TileElementIndex ij_in_tl(C, subm_dist.tile_element_from_local_element<C>(perm_from), 0);
  const TileElementIndex ij_out_tl(C, subm_dist.tile_element_from_local_element<C>(perm_to), 0);

  for (SizeType i_lc = 0; i_lc < subm_dist.local_nr_tiles().get<OC>(); ++i_lc) {
    const LocalTileIndex ij_in(OC, i_lc, j_in_lc);
    const LocalTileIndex ij_out(OC, i_lc, j_out_lc);

    const SizeType linear_in = dist_extra::local_tile_linear_index(subm_dist, ij_in);
    const auto& tile_in = in_tiles_fut[to_sizet(linear_in)].get();

    const SizeType linear_out = dist_extra::local_tile_linear_index(subm_dist, ij_out);
    auto& tile_out = out_tiles[to_sizet(linear_out)];

    DLAF_ASSERT_HEAVY(tile_in.size().template get<OC>() == tile_out.size().template get<OC>(),
                      tile_in.size(), tile_out.size());
    const TileElementSize region(OC, tile_in.size().template get<OC>(), 1);

    dlaf::tile::lacpy<T>(region, ij_in_tl, tile_in, ij_out_tl, tile_out);
  }
}

template <Backend B, Device D, class T, Coord C>
void Permutations<B, D, T, C>::call(const SizeType i_begin, const SizeType i_end,
                                    Matrix<const SizeType, D>& perms, Matrix<const T, D>& mat_in,
                                    Matrix<T, D>& mat_out) {
  namespace ex = pika::execution::experimental;
  namespace dist_extra = dlaf::matrix::internal::distribution;
  using dist_extra::local_element_distance_from_global_tile;

  if (i_begin == i_end)
    return;

  const matrix::Distribution& dist = mat_in.distribution();
  const SizeType m = local_element_distance_from_global_tile<Coord::Row>(dist, i_begin, i_end);
  const SizeType n = local_element_distance_from_global_tile<Coord::Col>(dist, i_begin, i_end);
  const matrix::Distribution subm_dist(LocalElementSize(m, n), dist.tile_size());

  if (subm_dist.size().isEmpty())
    return;

  auto next_local_from_global = [&dist](const SizeType i) -> LocalTileIndex {
    return {dist.next_local_tile_from_global_tile<Coord::Row>(i),
            dist.next_local_tile_from_global_tile<Coord::Col>(i)};
  };
  const LocalTileIndex idx_begin_lc = next_local_from_global(i_begin);
  const LocalTileIndex idx_end_lc = next_local_from_global(i_end);

  auto perms_range = common::iterate_range2d(LocalTileIndex(idx_begin_lc.get<C>(), 0),
                                             LocalTileIndex(idx_end_lc.get<C>(), 1));
  auto mat_range = common::iterate_range2d(idx_begin_lc, idx_end_lc);
  auto sender = ex::when_all(ex::when_all_vector(matrix::selectRead(perms, std::move(perms_range))),
                             ex::when_all_vector(matrix::selectRead(mat_in, mat_range)),
                             ex::when_all_vector(matrix::select(mat_out, std::move(mat_range))));

  if constexpr (D == Device::CPU) {
    const SizeType nperms = subm_dist.size().get<C>();

    auto permute_fn = [subm_dist](const auto i_perm, const auto& index_tile_futs,
                                  const auto& mat_in_tiles, const auto& mat_out_tiles) {
      const SizeType* perm_arr = index_tile_futs[0].get().ptr();

      [[maybe_unused]] const SizeType nperms = subm_dist.size().get<C>();
      DLAF_ASSERT_HEAVY(i_perm >= 0 && i_perm < nperms, i_perm, nperms);
      DLAF_ASSERT_HEAVY(perm_arr[i_perm] >= 0 && perm_arr[i_perm] < nperms, perm_arr[i_perm], nperms);

      applyPermutationOnCPU<T, C>(i_perm, subm_dist, perm_arr, mat_in_tiles, mat_out_tiles);
    };

    ex::start_detached(std::move(sender) | ex::transfer(dlaf::internal::getBackendScheduler<B>()) |
                       ex::bulk(nperms, std::move(permute_fn)));
  }
  else {
#if defined(DLAF_WITH_GPU)
    auto permute_fn = [subm_dist](const auto& index_tile_futs, const auto& mat_in_tiles,
                                  const auto& mat_out_tiles, whip::stream_t stream) {
      TileElementIndex zero(0, 0);
      const SizeType* i_ptr = index_tile_futs[0].get().ptr(zero);

      applyPermutationsOnDevice<T, C>(GlobalElementIndex(0, 0), subm_dist.size(), 0, subm_dist, i_ptr,
                                      mat_in_tiles, mat_out_tiles, stream);
    };

    ex::start_detached(std::move(sender) |
                       dlaf::internal::transform(dlaf::internal::Policy<B>(), std::move(permute_fn)));
#endif
  }
}

template <class T, Device D>
auto whenAllReadWriteTilesArray(LocalTileIndex begin, LocalTileIndex end, Matrix<T, D>& matrix) {
  const LocalTileSize sz{end.row() - begin.row(), end.col() - begin.col()};
  namespace ex = pika::execution::experimental;
  return ex::when_all_vector(matrix::select(matrix, common::iterate_range2d(begin, sz)));
}

template <class T, Device D>
auto whenAllReadWriteTilesArray(Matrix<T, D>& matrix) {
  namespace ex = pika::execution::experimental;
  return ex::when_all_vector(matrix::select(
      matrix, common::iterate_range2d(LocalTileIndex(0, 0), matrix.distribution().localNrTiles())));
}

template <class T, Device D>
auto whenAllReadOnlyTilesArray(LocalTileIndex begin, LocalTileIndex end, Matrix<const T, D>& matrix) {
  const LocalTileSize sz{end.row() - begin.row(), end.col() - begin.col()};
  namespace ex = pika::execution::experimental;
  return ex::when_all_vector(matrix::selectRead(matrix, common::iterate_range2d(begin, sz)));
}

template <class T, Device D>
auto whenAllReadOnlyTilesArray(Matrix<const T, D>& matrix) {
  namespace ex = pika::execution::experimental;
  return ex::when_all_vector(matrix::selectRead(
      matrix, common::iterate_range2d(LocalTileIndex(0, 0), matrix.distribution().localNrTiles())));
}

template <class T, Device D, Coord C, class SendCountsSender, class RecvCountsSender>
void all2allData(
    comm::CommunicatorPipeline<comm::coord_to_communicator_type(orthogonal(C))>& sub_task_chain,
    int nranks, LocalElementSize sz_loc, SendCountsSender&& send_counts_sender,
    Matrix<const T, D>& send_mat, RecvCountsSender&& recv_counts_sender, Matrix<T, D>& recv_mat) {
  namespace ex = pika::execution::experimental;

  using dlaf::common::DataDescriptor;

  const SizeType vec_size = sz_loc.get<orthogonal(C)>();
  auto sendrecv_f = [vec_size, sub_sub_task_chain = sub_task_chain.sub_pipeline()](
                        std::vector<int> send_counts, std::vector<int> send_displs,
                        const std::vector<matrix::internal::TileAsyncRwMutexReadOnlyWrapper<T, D>>&
                            send_tiles_fut,
                        std::vector<int> recv_counts, std::vector<int> recv_displs,
                        const std::vector<matrix::Tile<T, D>>& recv_tiles) mutable {
    // Note: both guaranteed to be column-major on allocation
    const T* send_ptr = send_tiles_fut[0].get().ptr();
    T* recv_ptr = recv_tiles[0].ptr();

    const SizeType send_ld = send_tiles_fut[0].get().ld();
    const SizeType recv_ld = recv_tiles[0].ld();

    const SizeType send_perm_stride = (C == Coord::Col) ? send_ld : 1;
    const SizeType recv_perm_stride = (C == Coord::Col) ? recv_ld : 1;

    // cumulative sum for computing rank data displacements in packed vectors
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

    std::vector<ex::unique_any_sender<>> all_comms;
    all_comms.reserve(to_sizet(sub_sub_task_chain.size() - 1) * 2);
    const comm::IndexT_MPI rank = sub_sub_task_chain.rank();
    for (comm::IndexT_MPI rank_partner = 0; rank_partner < sub_sub_task_chain.size(); ++rank_partner) {
      if (rank == rank_partner)
        continue;

      const auto rank_partner_index = to_sizet(rank_partner);

      if (send_counts[rank_partner_index])
        all_comms.push_back(
            sub_sub_task_chain.shared() |
            dlaf::comm::internal::transformMPI([=](const comm::Communicator& comm, MPI_Request* req) {
              const SizeType nperms = send_counts[rank_partner_index];
              auto message = dlaf::comm::make_message(DataDescriptor<const T>(
                  send_ptr + send_displs[rank_partner_index] * send_perm_stride,
                  C == Coord::Col ? nperms : vec_size, C == Coord::Col ? vec_size : nperms, send_ld));

              DLAF_MPI_CHECK_ERROR(MPI_Isend(message.data(), message.count(), message.mpi_type(),
                                             rank_partner, 0, comm, req));
            }));
      if (recv_counts[rank_partner_index])
        all_comms.push_back(
            sub_sub_task_chain.shared() |
            dlaf::comm::internal::transformMPI([=](const comm::Communicator& comm, MPI_Request* req) {
              const SizeType nperms = recv_counts[rank_partner_index];
              auto message = dlaf::comm::make_message(DataDescriptor<T>(
                  recv_ptr + recv_displs[rank_partner_index] * recv_perm_stride,
                  C == Coord::Col ? nperms : vec_size, C == Coord::Col ? vec_size : nperms, recv_ld));

              DLAF_MPI_CHECK_ERROR(MPI_Irecv(message.data(), message.count(), message.mpi_type(),
                                             rank_partner, 0, comm, req));
            }));
    }

    pika::this_thread::experimental::sync_wait(ex::when_all_vector(std::move(all_comms)));
  };

  ex::when_all(std::forward<SendCountsSender>(send_counts_sender),
               ex::just(std::vector<int>(to_sizet(nranks))), whenAllReadOnlyTilesArray(send_mat),
               std::forward<RecvCountsSender>(recv_counts_sender),
               ex::just(std::vector<int>(to_sizet(nranks))), whenAllReadWriteTilesArray(recv_mat)) |
      dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(), std::move(sendrecv_f));
}

// @param nranks number of ranks
// @param offset_sub where the sub-matrix to permute starts in the global matrix
// @param loc2sub_index a column matrix that represents a map from local to sub indices
// @param packing_index a column matrix that represents a map from packed indices to local indices
//
// Note: the order of the packed rows or columns on the send side must match the expected order at
// unpacking on the receive side
template <Coord C, bool BackwardMapping = false>
auto initPackingIndex(comm::IndexT_MPI nranks, SizeType offset_sub, const matrix::Distribution& dist,
                      Matrix<const SizeType, Device::CPU>& loc2sub_index,
                      Matrix<SizeType, Device::CPU>& packing_index) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto counts_fn = [nranks, offset_sub, dist, nperms = packing_index.size().rows()](
                       const auto& loc2gl_index_tiles, const auto& packing_index_tiles) {
    const SizeType* loc2sub = loc2gl_index_tiles[0].get().ptr();
    SizeType* out = packing_index_tiles[0].ptr();

    std::vector<int> counts(to_sizet(nranks));

    for (int rank = 0, rank_displacement = 0; rank < nranks; ++rank) {
      int& nperms_local = counts[to_sizet(rank)] = 0;

      for (SizeType perm_index_local = 0; perm_index_local < nperms; ++perm_index_local) {
        const SizeType perm_index_global = offset_sub + loc2sub[perm_index_local];
        DLAF_ASSERT_HEAVY(perm_index_local >= 0 && perm_index_local < nperms, perm_index_local, nperms);
        DLAF_ASSERT_HEAVY(perm_index_global >= 0 && perm_index_global < dist.size().get<C>(),
                          perm_index_global, dist.size());
        if (dist.rankGlobalElement<C>(perm_index_global) == rank) {
          const SizeType perm_index_packed = rank_displacement + nperms_local;

          if constexpr (BackwardMapping)
            out[perm_index_packed] = perm_index_local;
          else
            out[perm_index_local] = perm_index_packed;

          ++nperms_local;
        }
      }

      // Note:
      // This sorting actually "inverts" the mapping.
      if constexpr (BackwardMapping)
        std::sort(out + rank_displacement, out + rank_displacement + nperms_local,
                  [loc2sub](SizeType i1, SizeType i2) { return loc2sub[i1] < loc2sub[i2]; });

      rank_displacement += nperms_local;
    }

    return counts;
  };

  return ex::ensure_started(ex::when_all(whenAllReadOnlyTilesArray(loc2sub_index),
                                         whenAllReadWriteTilesArray(packing_index)) |
                            di::transform(di::Policy<Backend::MC>{}, std::move(counts_fn)));
}

// Copies index tiles belonging to the current process from the complete index @p global_index into the
// partial index containing only the local parts @p local_index.
template <Device D, Coord C>
void copyLocalPartsFromGlobalIndex(const SizeType i_loc_begin, const matrix::Distribution& dist,
                                   Matrix<const SizeType, D>& global_index,
                                   Matrix<SizeType, D>& local_index) {
  namespace ex = pika::execution::experimental;

  for (const LocalTileIndex i : common::iterate_range2d(local_index.distribution().localNrTiles())) {
    const GlobalTileIndex i_global(dist.globalTileFromLocalTile<C>(i_loc_begin + i.row()), 0);
    ex::start_detached(ex::when_all(global_index.read(i_global), local_index.readwrite(i)) |
                       dlaf::matrix::copy(dlaf::internal::Policy<DefaultBackend_v<D>>{}));
  }
}

// @param index_map a column matrix that represents a map from local `out` to local `in` indices
template <class T, Device D, Coord C, class IndexMapSender, class InSender, class OutSender>
void applyPackingIndex(const matrix::Distribution& subm_dist, IndexMapSender&& index_map, InSender&& in,
                       OutSender&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto sender = ex::when_all(std::forward<IndexMapSender>(index_map), std::forward<InSender>(in),
                             std::forward<OutSender>(out));

  if constexpr (D == Device::CPU) {
    const SizeType nperms = subm_dist.size().get<C>();

    auto permute_fn = [subm_dist](const auto i_perm, const auto& index_tile_futs,
                                  const auto& mat_in_tiles, const auto& mat_out_tiles) {
      const SizeType* perm_arr = index_tile_futs[0].get().ptr();

      [[maybe_unused]] const SizeType nperms = subm_dist.size().get<C>();
      DLAF_ASSERT_HEAVY(i_perm >= 0 && i_perm < nperms, i_perm, nperms);
      DLAF_ASSERT_HEAVY(perm_arr[i_perm] >= 0 && perm_arr[i_perm] < nperms, perm_arr[i_perm], nperms);

      applyPermutationOnCPU<T, C>(i_perm, subm_dist, perm_arr, mat_in_tiles, mat_out_tiles);
    };

    ex::start_detached(std::move(sender) | ex::transfer(di::getBackendScheduler<Backend::MC>()) |
                       ex::bulk(nperms, std::move(permute_fn)));
  }
  else {
#if defined(DLAF_WITH_GPU)
    auto permute_fn = [subm_dist](const auto& index_tile_futs, const auto& mat_in_tiles,
                                  const auto& mat_out_tiles, whip::stream_t stream) {
      TileElementIndex zero(0, 0);
      const SizeType* i_ptr = index_tile_futs[0].get().ptr(zero);

      applyPermutationsOnDevice<T, C>(GlobalElementIndex(0, 0), subm_dist.size(), 0, subm_dist, i_ptr,
                                      mat_in_tiles, mat_out_tiles, stream);
    };

    ex::start_detached(std::move(sender) |
                       dlaf::internal::transform(dlaf::internal::Policy<Backend::GPU>(),
                                                 std::move(permute_fn)));
#endif
  }
}

template <class T, Coord C, class SendCountsSender, class RecvCountsSender, class UnpackingIndexSender,
          class MatSendSender, class MatOutSender>
void unpackLocalOnCPU(const matrix::Distribution& subm_dist, const matrix::Distribution& dist,
                      SendCountsSender&& send_counts, RecvCountsSender&& recv_counts,
                      UnpackingIndexSender&& unpacking_index, MatSendSender&& mat_send,
                      MatOutSender&& mat_out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto setup_unpack_local_f = [subm_dist,
                               rank = dist.rankIndex().get<C>()](auto send_counts, auto recv_counts,
                                                                 auto index_tile_futs, auto mat_in_tiles,
                                                                 auto mat_out_tiles) {
    const size_t rank_index = to_sizet(rank);

    const SizeType* perm_arr = index_tile_futs[0].get().ptr();

    const int a = std::accumulate(send_counts.cbegin(), send_counts.cbegin() + rank, 0);
    const int b = a + send_counts[rank_index];

    // Note:
    // These are copied directly from mat_send, while unpacking permutation applies to indices on
    // the receiver side. So, we have to "align" the unpacking permutation, by applying the offset
    // existing between the send and recv side.
    // This is due to the fact that send and recv buffers might be "unbalanced", e.g. rank1 sends 2
    // and receive 1 with rank0, so resulting in a shift in indices between the two buffer sides,
    // following previous example the local part would start at index (0-based) 2 in mat_send and
    // at index 1 in mat_recv.
    const int a_r = std::accumulate(recv_counts.cbegin(), recv_counts.cbegin() + rank, 0);
    const SizeType offset = to_SizeType(a - a_r);
    std::vector<SizeType> perm_offseted;
    perm_offseted.reserve(to_sizet(subm_dist.size().get<C>()));
    std::transform(perm_arr, perm_arr + subm_dist.size().get<C>(), std::back_inserter(perm_offseted),
                   [offset](const SizeType perm) { return perm + offset; });

    return std::tuple(a, b, std::move(perm_offseted), std::move(mat_in_tiles), std::move(mat_out_tiles));
  };

  auto permutations_unpack_local_f = [subm_dist](const auto i_perm, const auto a, const auto b,
                                                 const auto& perm_offseted, const auto& mat_in_tiles,
                                                 const auto& mat_out_tiles) {
    const SizeType* perm_arr = perm_offseted.data();

    // [a, b)
    if (a <= perm_arr[i_perm] && perm_arr[i_perm] < b) {
      applyPermutationOnCPU<T, C>(i_perm, subm_dist, perm_arr, mat_in_tiles, mat_out_tiles);
    }
  };

  ex::start_detached(
      ex::when_all(std::forward<SendCountsSender>(send_counts),
                   std::forward<RecvCountsSender>(recv_counts),
                   std::forward<UnpackingIndexSender>(unpacking_index),
                   std::forward<MatSendSender>(mat_send), std::forward<MatOutSender>(mat_out)) |
      di::transform(di::Policy<Backend::MC>(), std::move(setup_unpack_local_f)) | ex::unpack() |
      ex::bulk(subm_dist.size().get<C>(), std::move(permutations_unpack_local_f)));
}

template <class T, Coord C, class RecvCountsSender, class UnpackingIndexSender, class MatRecvSender,
          class MatOutSender>
void unpackOthersOnCPU(const matrix::Distribution& subm_dist, const matrix::Distribution& dist,
                       RecvCountsSender&& recv_counts, UnpackingIndexSender&& unpacking_index,
                       MatRecvSender&& mat_recv, MatOutSender&& mat_out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto setup_unpack_f = [rank = dist.rankIndex().get<C>()](auto recv_counts, auto index_tile_futs,
                                                           auto mat_in_tiles, auto mat_out_tiles) {
    const size_t rank_index = to_sizet(rank);
    const int a = std::accumulate(recv_counts.cbegin(), recv_counts.cbegin() + rank, 0);
    const int b = a + recv_counts[rank_index];

    return std::tuple(a, b, std::move(index_tile_futs), std::move(mat_in_tiles),
                      std::move(mat_out_tiles));
  };

  auto permutations_unpack_f = [subm_dist](const auto i_perm, const auto a, const auto b,
                                           const auto& index_tile_futs, const auto& mat_in_tiles,
                                           const auto& mat_out_tiles) {
    const SizeType* perm_arr = index_tile_futs[0].get().ptr();

    // [0, a) and [b, end)
    if (perm_arr[i_perm] < a || b <= perm_arr[i_perm]) {
      applyPermutationOnCPU<T, C>(i_perm, subm_dist, perm_arr, mat_in_tiles, mat_out_tiles);
    }
  };

  ex::start_detached(ex::when_all(std::forward<RecvCountsSender>(recv_counts),
                                  std::forward<UnpackingIndexSender>(unpacking_index),
                                  std::forward<MatRecvSender>(mat_recv),
                                  std::forward<MatOutSender>(mat_out)) |
                     di::transform(di::Policy<Backend::MC>(), std::move(setup_unpack_f)) | ex::unpack() |
                     ex::bulk(subm_dist.size().get<C>(), std::move(permutations_unpack_f)));
}

template <class T, Coord C>
void permuteOnCPU(
    comm::CommunicatorPipeline<comm::coord_to_communicator_type(orthogonal(C))>& sub_task_chain,
    SizeType i_begin, SizeType i_end, Matrix<const SizeType, Device::CPU>& perms,
    Matrix<const T, Device::CPU>& mat_in, Matrix<T, Device::CPU>& mat_out) {
  constexpr Device D = Device::CPU;

  using namespace dlaf::matrix;

  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  if (i_begin == i_end)
    return;

  const Distribution& dist = mat_in.distribution();
  const comm::IndexT_MPI nranks = to_int(dist.commGridSize().get<C>());

  // Local size and index of subproblem [i_begin, i_end)
  const SizeType offset_sub = dist.globalElementFromGlobalTileAndTileElement<C>(i_begin, 0);
  const TileElementSize blk = dist.blockSize();

  const LocalTileIndex i_loc_begin{dist.nextLocalTileFromGlobalTile<Coord::Row>(i_begin),
                                   dist.nextLocalTileFromGlobalTile<Coord::Col>(i_begin)};
  const LocalTileIndex i_loc_end{dist.nextLocalTileFromGlobalTile<Coord::Row>(i_end),
                                 dist.nextLocalTileFromGlobalTile<Coord::Col>(i_end)};
  // Note: the local shape of the permutation region may not be square if the process grid is not square
  const LocalElementSize sz_loc{dist.localElementDistanceFromGlobalTile<Coord::Row>(i_begin, i_end),
                                dist.localElementDistanceFromGlobalTile<Coord::Col>(i_begin, i_end)};

  // If there are no tiles in this rank, nothing to do here
  if (sz_loc.isEmpty())
    return;

  // Create a map from send indices to receive indices (inverse of perms)
  Matrix<SizeType, D> inverse_perms(perms.distribution());
  eigensolver::internal::invertIndex(i_begin, i_end, perms, inverse_perms);

  // Local distribution used for packing and unpacking
  const Distribution subm_dist(sz_loc, blk);

  // Local single tile column matrices representing index maps used for packing and unpacking of
  // communication data
  const SizeType nvecs = sz_loc.get<C>();
  const Distribution index_dist(LocalElementSize(nvecs, 1), TileElementSize(blk.get<C>(), 1));
  Matrix<SizeType, D> local2global_index(index_dist);
  Matrix<SizeType, D> packing_index(index_dist);
  Matrix<SizeType, D> unpacking_index(index_dist);

  // Local matrices used for packing data for communication. Both matrices are in column-major order.
  // The particular constructor is used on purpose to guarantee that columns are stored contiguosly,
  // such that there is no padding and gaps between them.
  const LocalElementSize comm_sz = sz_loc;
  const Distribution comm_dist(comm_sz, blk);
  const LayoutInfo comm_layout = matrix::colMajorLayout(comm_sz, blk, comm_sz.rows());

  Matrix<T, D> mat_send(comm_dist, comm_layout);
  Matrix<T, D> mat_recv(comm_dist, comm_layout);

  // Initialize the unpacking index
  copyLocalPartsFromGlobalIndex<D, C>(i_loc_begin.get<C>(), dist, perms, local2global_index);
  auto recv_counts_sender =
      initPackingIndex<C>(nranks, offset_sub, dist, local2global_index, unpacking_index) | ex::split();

  // Initialize the packing index
  // Here `true` is specified so that the send side matches the order of columns on the receive side
  copyLocalPartsFromGlobalIndex<D, C>(i_loc_begin.get<C>(), dist, inverse_perms, local2global_index);
  auto send_counts_sender =
      initPackingIndex<C, true>(nranks, offset_sub, dist, local2global_index, packing_index) |
      ex::split();

  // Pack local rows or columns to be sent from this rank
  applyPackingIndex<T, D, C>(subm_dist, whenAllReadOnlyTilesArray(packing_index),
                             whenAllReadOnlyTilesArray(i_loc_begin, i_loc_end, mat_in),
                             whenAllReadWriteTilesArray(mat_send));

  // Unpacking
  // separate unpacking:
  // - locals
  // - communicated
  // and then start two different tasks:
  // - the first depends on mat_send instead of mat_recv (no dependency on comm)
  // - the last is the same, but it has to skip the part already done for local

  // LOCAL
  unpackLocalOnCPU<T, C>(subm_dist, dist, send_counts_sender, recv_counts_sender,
                         whenAllReadOnlyTilesArray(unpacking_index), whenAllReadOnlyTilesArray(mat_send),
                         whenAllReadWriteTilesArray(i_loc_begin, i_loc_end, mat_out));
  // COMMUNICATION-dependent
  all2allData<T, D, C>(sub_task_chain, nranks, sz_loc, send_counts_sender, mat_send, recv_counts_sender,
                       mat_recv);
  // OTHERS
  unpackOthersOnCPU<T, C>(subm_dist, dist, std::move(recv_counts_sender),
                          whenAllReadOnlyTilesArray(unpacking_index),
                          whenAllReadOnlyTilesArray(mat_recv),
                          whenAllReadWriteTilesArray(i_loc_begin, i_loc_end, mat_out));
}

template <Backend B, Device D, class T, Coord C>
void Permutations<B, D, T, C>::call(
    comm::CommunicatorPipeline<comm::coord_to_communicator_type(orthogonal(C))>& sub_task_chain,
    SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& perms, Matrix<const T, D>& mat_in,
    Matrix<T, D>& mat_out) {
  if constexpr (D == Device::GPU) {
    // This is a temporary placeholder which avoids diverging GPU API:
    DLAF_UNIMPLEMENTED("GPU implementation not available yet");
    dlaf::internal::silenceUnusedWarningFor(sub_task_chain, i_begin, i_end, perms, mat_in, mat_out);
    return;
  }
  else {
    permuteOnCPU<T, C>(sub_task_chain, i_begin, i_end, perms, mat_in, mat_out);
  }
}
}
