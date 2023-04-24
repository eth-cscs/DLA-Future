//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "dlaf/matrix/index.h"
#include "dlaf/permutations/general/api.h"
#include "dlaf/permutations/general/perms.h"

#include "dlaf/blas/tile.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/message.h"
#include "dlaf/communication/rdma.h"
#include "dlaf/lapack/gpu/lacpy.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/schedulers.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/transform.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include <mpi.h>
#include <numeric>
#include <pika/future.hpp>
#include "pika/algorithm.hpp"

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
//
// Note: `in_tiles` should be `const T` but to avoid extra allocations necessary for unwrapping
//       `std::vector<shared_future<matrix::Tile<const T, D>>>` it is left as non-const
template <class T, Device D, Coord coord, class... Args>
void applyPermutations(GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
                       const matrix::Distribution& distr, const SizeType* perm_arr,
                       const std::vector<pika::shared_future<matrix::Tile<const T, D>>>& in_tiles_fut,
                       const std::vector<matrix::Tile<T, D>>& out_tiles,
                       [[maybe_unused]] Args&&... args) {
  if constexpr (D == Device::CPU) {
    constexpr Coord orth_coord = orthogonal(coord);
    std::vector<SizeType> splits =
        util::interleaveSplits(sz.get<orth_coord>(), distr.blockSize().get<orth_coord>(),
                               distr.distanceToAdjacentTile<orth_coord>(in_offset),
                               distr.distanceToAdjacentTile<orth_coord>(out_begin.get<orth_coord>()));

    // Parallelized over the number of permuted columns or rows
    pika::for_loop(pika::execution::par, to_sizet(0), to_sizet(sz.get<coord>()), [&](SizeType i_perm) {
      for (std::size_t i_split = 0; i_split < splits.size() - 1; ++i_split) {
        SizeType split = splits[i_split];

        GlobalElementIndex i_split_gl_in(split + in_offset, perm_arr[i_perm]);
        GlobalElementIndex i_split_gl_out(split + out_begin.get<orth_coord>(),
                                          out_begin.get<coord>() + i_perm);
        TileElementSize region(splits[i_split + 1] - split, 1);
        if constexpr (coord == Coord::Row) {
          region.transpose();
          i_split_gl_in.transpose();
          i_split_gl_out.transpose();
        }

        TileElementIndex i_subtile_in = distr.tileElementIndex(i_split_gl_in);
        const auto& tile_in = in_tiles_fut[to_sizet(distr.globalTileLinearIndex(i_split_gl_in))].get();
        TileElementIndex i_subtile_out = distr.tileElementIndex(i_split_gl_out);
        auto& tile_out = out_tiles[to_sizet(distr.globalTileLinearIndex(i_split_gl_out))];

        dlaf::tile::lacpy<T>(region, i_subtile_in, tile_in, i_subtile_out, tile_out);
      }
    });
  }
  else if constexpr (D == Device::GPU) {
#if defined(DLAF_WITH_GPU)
    applyPermutationsOnDevice<T, coord>(out_begin, sz, in_offset, distr, perm_arr, in_tiles_fut,
                                        out_tiles, args...);
#endif
  }
}

// FilterFunc is a function with signature bool(*)(SizeType)
template <class T, Device D, Coord C, class FilterFunc>
void applyPermutationsFiltered(
    GlobalElementIndex out_begin, GlobalElementSize sz, SizeType in_offset,
    const matrix::Distribution& subm_dist, const SizeType* perm_arr,
    const std::vector<pika::shared_future<matrix::Tile<const T, D>>>& in_tiles_fut,
    const std::vector<matrix::Tile<T, D>>& out_tiles, FilterFunc&& filter) {
  constexpr auto OC = orthogonal(C);
  std::vector<SizeType> splits =
      dlaf::util::interleaveSplits(sz.get<OC>(), subm_dist.blockSize().get<OC>(),
                                   subm_dist.distanceToAdjacentTile<OC>(in_offset),
                                   subm_dist.distanceToAdjacentTile<OC>(out_begin.get<OC>()));

  const SizeType nperms = subm_dist.size().get<C>();

  // Parallelized over the number of permutations
  pika::for_loop(pika::execution::par, to_sizet(0), to_sizet(nperms), [&](SizeType i_perm) {
    if (!filter(perm_arr[i_perm]))
      return;

    for (std::size_t i_split = 0; i_split < splits.size() - 1; ++i_split) {
      const SizeType split = splits[i_split];

      GlobalElementIndex i_split_gl_in(split + in_offset, perm_arr[i_perm]);
      GlobalElementIndex i_split_gl_out(split + out_begin.get<OC>(), out_begin.get<C>() + i_perm);
      TileElementSize region(splits[i_split + 1] - split, 1);

      if constexpr (C == Coord::Row) {
        region.transpose();
        i_split_gl_in.transpose();
        i_split_gl_out.transpose();
      }

      const TileElementIndex i_subtile_in = subm_dist.tileElementIndex(i_split_gl_in);
      const auto& tile_in = in_tiles_fut[to_sizet(subm_dist.globalTileLinearIndex(i_split_gl_in))].get();
      const TileElementIndex i_subtile_out = subm_dist.tileElementIndex(i_split_gl_out);
      auto& tile_out = out_tiles[to_sizet(subm_dist.globalTileLinearIndex(i_split_gl_out))];

      dlaf::tile::lacpy<T>(region, i_subtile_in, tile_in, i_subtile_out, tile_out);
    }
  });
}

template <Backend B, Device D, class T, Coord C>
void Permutations<B, D, T, C>::call(SizeType i_begin, SizeType i_last, Matrix<const SizeType, D>& perms,
                                    Matrix<const T, D>& mat_in, Matrix<T, D>& mat_out) {
  namespace ut = matrix::util;
  namespace ex = pika::execution::experimental;

  const matrix::Distribution& distr = mat_in.distribution();
  TileElementSize sz_last_tile = distr.tileSize(GlobalTileIndex(i_last, i_last));
  SizeType m = distr.globalTileElementDistance<Coord::Row>(i_begin, i_last) + sz_last_tile.rows();
  SizeType n = distr.globalTileElementDistance<Coord::Col>(i_begin, i_last) + sz_last_tile.cols();
  matrix::Distribution subm_distr(LocalElementSize(m, n), distr.blockSize());
  SizeType ntiles = i_last - i_begin + 1;

  auto sender =
      ex::when_all(ex::when_all_vector(ut::collectReadTiles(LocalTileIndex(i_begin, 0),
                                                            LocalTileSize(ntiles, 1), perms)),
                   ex::when_all_vector(ut::collectReadTiles(LocalTileIndex(i_begin, i_begin),
                                                            LocalTileSize(ntiles, ntiles), mat_in)),
                   ex::when_all_vector(ut::collectReadWriteTiles(LocalTileIndex(i_begin, i_begin),
                                                                 LocalTileSize(ntiles, ntiles),
                                                                 mat_out)));

  auto permute_fn = [subm_distr](const auto& index_tile_futs, const auto& mat_in_tiles,
                                 const auto& mat_out_tiles, auto&&... ts) {
    TileElementIndex zero(0, 0);
    const SizeType* i_ptr = index_tile_futs[0].get().ptr(zero);
    applyPermutations<T, D, C>(GlobalElementIndex(0, 0), subm_distr.size(), 0, subm_distr, i_ptr,
                               mat_in_tiles, mat_out_tiles, std::forward<decltype(ts)>(ts)...);
  };
  ex::start_detached(
      dlaf::internal::transform(dlaf::internal::Policy<B>(), std::move(permute_fn), std::move(sender)));
}

template <class T, Device D>
auto whenAllReadWriteTilesArray(LocalTileIndex begin, LocalTileIndex end, Matrix<T, D>& matrix) {
  LocalTileSize sz{end.row() - begin.row() + 1, end.col() - begin.col() + 1};
  namespace ex = pika::execution::experimental;
  namespace ut = matrix::util;
  return ex::when_all_vector(ut::collectReadWriteTiles(begin, sz, matrix));
}

template <class T, Device D>
auto whenAllReadWriteTilesArray(Matrix<T, D>& matrix) {
  namespace ex = pika::execution::experimental;
  namespace ut = matrix::util;
  return ex::when_all_vector(
      ut::collectReadWriteTiles(LocalTileIndex(0, 0), matrix.distribution().localNrTiles(), matrix));
}

template <class T, Device D>
auto whenAllReadOnlyTilesArray(LocalTileIndex begin, LocalTileIndex end, Matrix<const T, D>& matrix) {
  LocalTileSize sz{end.row() - begin.row() + 1, end.col() - begin.col() + 1};
  namespace ex = pika::execution::experimental;
  namespace ut = matrix::util;
  return ex::when_all_vector(ut::collectReadTiles(begin, sz, matrix));
}

template <class T, Device D>
auto whenAllReadOnlyTilesArray(Matrix<const T, D>& matrix) {
  namespace ex = pika::execution::experimental;
  namespace ut = matrix::util;
  return ex::when_all_vector(
      ut::collectReadTiles(LocalTileIndex(0, 0), matrix.distribution().localNrTiles(), matrix));
}

template <class T, Device D, Coord C, class SendCountsSender, class RecvCountsSender>
void all2allData(common::Pipeline<comm::Communicator>& sub_task_chain, int nranks,
                 LocalElementSize sz_loc, SendCountsSender&& send_counts_sender,
                 Matrix<const T, D>& send_mat, RecvCountsSender&& recv_counts_sender,
                 Matrix<T, D>& recv_mat) {
  namespace ex = pika::execution::experimental;

  using dlaf::common::DataDescriptor;

  const SizeType vec_size = sz_loc.get<orthogonal(C)>();
  auto sendrecv_f =
      [vec_size](comm::Communicator& comm, std::vector<int> send_counts, std::vector<int> send_displs,
                 const std::vector<pika::shared_future<matrix::Tile<const T, D>>>& send_tiles_fut,
                 std::vector<int> recv_counts, std::vector<int> recv_displs,
                 const std::vector<matrix::Tile<T, D>>& recv_tiles) {
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
        all_comms.reserve(to_sizet(comm.size() - 1) * 2);
        const comm::IndexT_MPI rank = comm.rank();
        for (comm::IndexT_MPI rank_partner = 0; rank_partner < comm.size(); ++rank_partner) {
          if (rank == rank_partner)
            continue;

          const auto rank_partner_index = to_sizet(rank_partner);

          if (send_counts[rank_partner_index])
            all_comms.push_back(
                ex::just() | dlaf::comm::internal::transformMPI([=](MPI_Request* req) {
                  const SizeType nperms = send_counts[rank_partner_index];
                  auto message = dlaf::comm::make_message(
                      DataDescriptor<const T>(send_ptr +
                                                  send_displs[rank_partner_index] * send_perm_stride,
                                              C == Coord::Col ? nperms : vec_size,
                                              C == Coord::Col ? vec_size : nperms, send_ld));

                  DLAF_MPI_CHECK_ERROR(MPI_Isend(message.data(), message.count(), message.mpi_type(),
                                                 rank_partner, 0, comm, req));
                }));
          if (recv_counts[rank_partner_index])
            all_comms.push_back(
                ex::just() | dlaf::comm::internal::transformMPI([=](MPI_Request* req) {
                  const SizeType nperms = recv_counts[rank_partner_index];
                  auto message = dlaf::comm::make_message(
                      DataDescriptor<T>(recv_ptr + recv_displs[rank_partner_index] * recv_perm_stride,
                                        C == Coord::Col ? nperms : vec_size,
                                        C == Coord::Col ? vec_size : nperms, recv_ld));

                  DLAF_MPI_CHECK_ERROR(MPI_Irecv(message.data(), message.count(), message.mpi_type(),
                                                 rank_partner, 0, comm, req));
                }));
        }

        pika::this_thread::experimental::sync_wait(ex::when_all_vector(std::move(all_comms)));
      };

  ex::when_all(sub_task_chain(), std::forward<SendCountsSender>(send_counts_sender),
               ex::just(std::vector<int>(to_sizet(nranks))), whenAllReadOnlyTilesArray(send_mat),
               std::forward<RecvCountsSender>(recv_counts_sender),
               ex::just(std::vector<int>(to_sizet(nranks))), whenAllReadWriteTilesArray(recv_mat)) |
      dlaf::internal::transformDetach(dlaf::internal::Policy<Backend::MC>(), sendrecv_f);
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

  auto counts_fn = [nranks, offset_sub, dist,
                    nperms = packing_index.size().rows()](const auto& loc2gl_index_tiles,
                                                          const auto& packing_index_tiles) {
    const SizeType* loc2sub = loc2gl_index_tiles[0].get().ptr();
    SizeType* out = packing_index_tiles[0].ptr();

    std::vector<int> counts(to_sizet(nranks));

    for (int rank = 0, rank_displacement = 0; rank < nranks; ++rank) {
      int& nperms_local = counts[to_sizet(rank)] = 0;

      for (SizeType perm_index_local = 0; perm_index_local < nperms; ++perm_index_local) {
        const SizeType perm_index_global = offset_sub + loc2sub[perm_index_local];
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

  return ex::ensure_started(
      ex::when_all(whenAllReadOnlyTilesArray(loc2sub_index), whenAllReadWriteTilesArray(packing_index)) |
      di::transform(di::Policy<Backend::MC>{}, std::move(counts_fn)));
}

// Copies index tiles belonging to the current process from the complete index @p global_index into the
// partial index containing only the local parts @p local_index.
template <Device D, Coord C>
void copyLocalPartsFromGlobalIndex(SizeType i_loc_begin, const matrix::Distribution& dist,
                                   Matrix<const SizeType, D>& global_index,
                                   Matrix<SizeType, D>& local_index) {
  namespace ex = pika::execution::experimental;

  for (const LocalTileIndex i : common::iterate_range2d(local_index.distribution().localNrTiles())) {
    const GlobalTileIndex i_global(dist.globalTileFromLocalTile<C>(i_loc_begin + i.row()), 0);
    ex::start_detached(
        ex::when_all(global_index.read_sender(i_global), local_index.readwrite_sender(i)) |
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

  auto permute_fn = [subm_dist](const auto& index_tile_futs, const auto& mat_in_tiles,
                                const auto& mat_out_tiles, auto&&... ts) {
    const SizeType* i_ptr = index_tile_futs[0].get().ptr();
    applyPermutations<T, D, C>(GlobalElementIndex(0, 0), subm_dist.size(), 0, subm_dist, i_ptr,
                               mat_in_tiles, mat_out_tiles, std::forward<decltype(ts)>(ts)...);
  };
  ex::start_detached(
      di::transform(di::Policy<DefaultBackend_v<D>>(), std::move(permute_fn), std::move(sender)));
}

// Tranposes two tiles of compatible dimensions
template <class InTileSender, class OutTileSender>
void transposeTileSenders(InTileSender&& in, OutTileSender&& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto sender = ex::when_all(std::forward<InTileSender>(in), std::forward<OutTileSender>(out));

  auto transpose_fn = [](const auto& in_tile, const auto& out_tile) {
    for (TileElementIndex idx : common::iterate_range2d(out_tile.size())) {
      out_tile(idx) = in_tile(transposed(idx));
    }
  };

  ex::start_detached(
      di::transform(di::Policy<Backend::MC>(), std::move(transpose_fn), std::move(sender)));
}

// Transposes a local matrix @p mat_in into the local part of the distributed matrix @p mat_out.
template <class T>
void transposeFromLocalToDistributedMatrix(LocalTileIndex i_loc_begin,
                                           Matrix<const T, Device::CPU>& mat_in,
                                           Matrix<T, Device::CPU>& mat_out) {
  for (auto i_in_tile : common::iterate_range2d(mat_in.distribution().localNrTiles())) {
    LocalTileIndex i_out_tile(i_loc_begin.row() + i_in_tile.col(), i_loc_begin.col() + i_in_tile.row());
    transposeTileSenders(mat_in.read_sender(i_in_tile), mat_out.readwrite_sender(i_out_tile));
  }
}

// Transposes the local part of the distributed matrix @p mat_in into the local matrix @p mat_out.
template <class T>
void transposeFromDistributedToLocalMatrix(LocalTileIndex i_loc_begin,
                                           Matrix<const T, Device::CPU>& mat_in,
                                           Matrix<T, Device::CPU>& mat_out) {
  for (auto i_out_tile : common::iterate_range2d(mat_out.distribution().localNrTiles())) {
    LocalTileIndex i_in_tile(i_loc_begin.row() + i_out_tile.col(), i_loc_begin.col() + i_out_tile.row());
    transposeTileSenders(mat_in.read_sender(i_in_tile), mat_out.readwrite_sender(i_out_tile));
  }
}

// Inverts the the subset of tiles [ @p i_begin, @p i_end ) of the index map @p in and saves
// the result into @p out.
// TODO: duplicated?
inline void invertIndex(SizeType i_begin, SizeType i_end, Matrix<const SizeType, Device::CPU>& in,
                        Matrix<SizeType, Device::CPU>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  namespace ut = matrix::util;

  const matrix::Distribution& dist = in.distribution();
  SizeType nb = dist.blockSize().rows();
  SizeType nbr = dist.tileSize(GlobalTileIndex(i_end - 1, 0)).rows();
  SizeType n = (i_end - i_begin - 1) * nb + nbr;
  auto inv_fn = [n](const auto& in_tiles_futs, const auto& out_tiles) {
    TileElementIndex zero(0, 0);
    const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero);
    SizeType* out_ptr = out_tiles[0].ptr(zero);
    for (SizeType i = 0; i < n; ++i) {
      out_ptr[in_ptr[i]] = i;
    }
  };

  LocalTileIndex begin{i_begin, 0};
  LocalTileSize sz{i_end - i_begin, 1};
  auto sender = ex::when_all(ex::when_all_vector(ut::collectReadTiles(begin, sz, in)),
                             ex::when_all_vector(ut::collectReadWriteTiles(begin, sz, out)));
  ex::start_detached(di::transform(di::Policy<Backend::MC>(), std::move(inv_fn), std::move(sender)));
}

template <class T, Coord C>
void permuteOnCPU(common::Pipeline<comm::Communicator>& sub_task_chain, SizeType i_begin,
                  SizeType i_last, Matrix<const SizeType, Device::CPU>& perms,
                  Matrix<const T, Device::CPU>& mat_in, Matrix<T, Device::CPU>& mat_out) {
  constexpr Device D = Device::CPU;

  using namespace dlaf::matrix;

  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const Distribution& dist = mat_in.distribution();
  const comm::IndexT_MPI nranks = to_int(dist.commGridSize().get<C>());

  const SizeType i_end = i_last + 1;

  // Local size and index of subproblem [i_begin, i_last]
  const SizeType offset_sub = dist.globalElementFromGlobalTileAndTileElement<C>(i_begin, 0);
  const TileElementSize blk = dist.blockSize();

  const LocalTileIndex i_loc_begin{dist.nextLocalTileFromGlobalTile<Coord::Row>(i_begin),
                                   dist.nextLocalTileFromGlobalTile<Coord::Col>(i_begin)};
  const LocalTileIndex i_loc_last{dist.nextLocalTileFromGlobalTile<Coord::Row>(i_end) - 1,
                                 dist.nextLocalTileFromGlobalTile<Coord::Col>(i_end) - 1};
  // Note: the local shape of the permutation region may not be square if the process grid is not square
  const LocalElementSize sz_loc{dist.localElementDistanceFromGlobalTile<Coord::Row>(i_begin, i_end),
                                dist.localElementDistanceFromGlobalTile<Coord::Col>(i_begin, i_end)};

  // If there are no tiles in this rank, nothing to do here
  if (sz_loc.isEmpty())
    return;

  // Create a map from send indices to receive indices (inverse of perms)
  Matrix<SizeType, D> inverse_perms(perms.distribution());
  invertIndex(i_begin, i_end, perms, inverse_perms);

  // Local distribution used for packing and unpacking
  const Distribution subm_dist(sz_loc, blk);

  // Local single tile column matrices representing index maps used for packing and unpacking of
  // communication data
  const SizeType nvecs = sz_loc.get<C>();
  const Distribution index_dist(LocalElementSize(nvecs, 1), TileElementSize(blk.rows(), 1));
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
                             whenAllReadOnlyTilesArray(i_loc_begin, i_loc_last, mat_in),
                             whenAllReadWriteTilesArray(mat_send));

  // Unpacking
  // separate unpacking:
  // - locals
  // - communicated
  // and then start two different tasks:
  // - the first depends on mat_send instead of mat_recv (no dependency on comm)
  // - the last is the same, but it has to skip the part already done for local

  // LOCAL
  auto unpack_local_f = [subm_dist, rank = dist.rankIndex().get<C>()](const auto& send_counts,
                                                                      const auto& recv_counts,
                                                                      const auto& index_tile_futs,
                                                                      const auto& mat_in_tiles,
                                                                      const auto& mat_out_tiles) {
    const size_t rank_index = to_sizet(rank);

    const SizeType* perm_arr = index_tile_futs[0].get().ptr();
    const GlobalElementSize sz = subm_dist.size();

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
    std::vector<SizeType> perm_offseted(perm_arr, perm_arr + subm_dist.size().get<C>());
    std::transform(perm_offseted.begin(), perm_offseted.end(), perm_offseted.begin(),
                   [offset](const SizeType perm) { return perm + offset; });

    // [a, b)
    applyPermutationsFiltered<T, D, C>({0, 0}, sz, 0, subm_dist, perm_offseted.data(), mat_in_tiles,
                                       mat_out_tiles,
                                       [a, b](SizeType i_perm) { return i_perm >= a && i_perm < b; });
  };

  ex::when_all(send_counts_sender, recv_counts_sender, whenAllReadOnlyTilesArray(unpacking_index),
               whenAllReadOnlyTilesArray(mat_send),
               whenAllReadWriteTilesArray(i_loc_begin, i_loc_last, mat_out)) |
      di::transformDetach(di::Policy<DefaultBackend_v<D>>(), std::move(unpack_local_f));

  // COMMUNICATION-dependent
  all2allData<T, D, C>(sub_task_chain, nranks, sz_loc, send_counts_sender, mat_send, recv_counts_sender,
                       mat_recv);

  auto unpack_others_f = [subm_dist, rank = dist.rankIndex().get<C>()](const auto& recv_counts,
                                                                       const auto& index_tile_futs,
                                                                       const auto& mat_in_tiles,
                                                                       const auto& mat_out_tiles) {
    const size_t rank_index = to_sizet(rank);
    const int a = std::accumulate(recv_counts.cbegin(), recv_counts.cbegin() + rank, 0);
    const int b = a + recv_counts[rank_index];

    const SizeType* perm_arr = index_tile_futs[0].get().ptr();
    const GlobalElementSize sz = subm_dist.size();

    // [0, a)
    applyPermutationsFiltered<T, D, C>({0, 0}, sz, 0, subm_dist, perm_arr, mat_in_tiles, mat_out_tiles,
                                       [a](SizeType i_perm) { return i_perm < a; });

    // [b, end)
    applyPermutationsFiltered<T, D, C>({0, 0}, sz, 0, subm_dist, perm_arr, mat_in_tiles, mat_out_tiles,
                                       [b](SizeType i_perm) { return i_perm >= b; });
  };

  ex::when_all(recv_counts_sender, whenAllReadOnlyTilesArray(unpacking_index),
               whenAllReadOnlyTilesArray(mat_recv),
               whenAllReadWriteTilesArray(i_loc_begin, i_loc_last, mat_out)) |
      di::transformDetach(di::Policy<DefaultBackend_v<D>>(), std::move(unpack_others_f));
}

template <Backend B, Device D, class T, Coord C>
void Permutations<B, D, T, C>::call(common::Pipeline<comm::Communicator>& sub_task_chain,
                                    SizeType i_begin, SizeType i_last, Matrix<const SizeType, D>& perms,
                                    Matrix<const T, D>& mat_in, Matrix<T, D>& mat_out) {
  if constexpr (D == Device::GPU) {
    // This is a temporary placeholder which avoids diverging GPU API:
    DLAF_UNIMPLEMENTED("GPU implementation not available yet");
    dlaf::internal::silenceUnusedWarningFor(sub_task_chain, i_begin, i_last, perms, mat_in, mat_out);
    return;
  }
  else {
    permuteOnCPU<T, C>(sub_task_chain, i_begin, i_last, perms, mat_in, mat_out);
  }
}
}
