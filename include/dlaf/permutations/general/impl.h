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
#include "dlaf/sender/transform.h"
#include "dlaf/sender/transform_mpi.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#include <mpi.h>
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
                       const std::vector<matrix::Tile<T, D>>& in_tiles,
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
        auto& tile_in = in_tiles[to_sizet(distr.globalTileLinearIndex(i_split_gl_in))];
        TileElementIndex i_subtile_out = distr.tileElementIndex(i_split_gl_out);
        auto& tile_out = out_tiles[to_sizet(distr.globalTileLinearIndex(i_split_gl_out))];

        dlaf::tile::lacpy<T>(region, i_subtile_in, tile_in, i_subtile_out, tile_out);
      }
    });
  }
  else if constexpr (D == Device::GPU) {
#if defined(DLAF_WITH_GPU)
    applyPermutationsOnDevice<T, coord>(out_begin, sz, in_offset, distr, perm_arr, in_tiles, out_tiles,
                                        args...);
#endif
  }
}

template <Backend B, Device D, class T, Coord C>
void Permutations<B, D, T, C>::call(SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& perms,
                                    Matrix<T, D>& mat_in, Matrix<T, D>& mat_out) {
  namespace ut = matrix::util;
  namespace ex = pika::execution::experimental;

  const matrix::Distribution& distr = mat_in.distribution();
  TileElementSize sz_last_tile = distr.tileSize(GlobalTileIndex(i_end, i_end));
  SizeType m = distr.globalTileElementDistance<Coord::Row>(i_begin, i_end) + sz_last_tile.rows();
  SizeType n = distr.globalTileElementDistance<Coord::Col>(i_begin, i_end) + sz_last_tile.cols();
  matrix::Distribution subm_distr(LocalElementSize(m, n), distr.blockSize());
  SizeType ntiles = i_end - i_begin + 1;

  auto sender =
      ex::when_all(ex::when_all_vector(
                       matrix::selectRead(perms, common::iterate_range2d(LocalTileIndex(i_begin, 0),
                                                                         LocalTileSize(ntiles, 1)))),
                   ex::when_all_vector(
                       matrix::select(mat_in, common::iterate_range2d(LocalTileIndex(i_begin, i_begin),
                                                                      LocalTileSize(ntiles, ntiles)))),
                   ex::when_all_vector(
                       matrix::select(mat_out, common::iterate_range2d(LocalTileIndex(i_begin, i_begin),
                                                                       LocalTileSize(ntiles, ntiles)))));

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
  return ex::when_all_vector(matrix::select(matrix, common::iterate_range2d(begin, sz)));
}

template <class T, Device D>
auto whenAllReadWriteTilesArray(Matrix<T, D>& matrix) {
  namespace ex = pika::execution::experimental;
  namespace ut = matrix::util;
  return ex::when_all_vector(
      matrix::select(matrix, common::iterate_range2d(LocalTileIndex(0, 0),
                                                     matrix.distribution().localNrTiles())));
}

template <class T, Device D>
auto whenAllReadOnlyTilesArray(LocalTileIndex begin, LocalTileIndex end, Matrix<const T, D>& matrix) {
  LocalTileSize sz{end.row() - begin.row() + 1, end.col() - begin.col() + 1};
  namespace ex = pika::execution::experimental;
  namespace ut = matrix::util;
  return ex::when_all_vector(matrix::selectRead(matrix, common::iterate_range2d(begin, sz)));
}

template <class T, Device D>
auto whenAllReadOnlyTilesArray(Matrix<const T, D>& matrix) {
  namespace ex = pika::execution::experimental;
  namespace ut = matrix::util;
  return ex::when_all_vector(
      matrix::selectRead(matrix, common::iterate_range2d(LocalTileIndex(0, 0),
                                                         matrix.distribution().localNrTiles())));
}

// No data is sent or received but the processes participates in the collective call
template <class T>
void all2allEmptyData(common::Pipeline<comm::Communicator>& sub_task_chain, int nranks) {
  namespace ex = pika::execution::experimental;

  auto all2all_f = [](const comm::Communicator& comm, const std::vector<int>& arr, MPI_Request* req) {
    MPI_Datatype dtype = dlaf::comm::mpi_datatype<std::remove_pointer_t<T>>::type;
    // Avoid buffer aliasing error reported when send_ptr and recv_ptr are the same.
    int irrelevant = 42;
    DLAF_MPI_CHECK_ERROR(MPI_Ialltoallv(nullptr, arr.data(), arr.data(), dtype, &irrelevant, arr.data(),
                                        arr.data(), dtype, comm, req));
  };
  dlaf::comm::internal::transformMPIDetach(std::move(all2all_f),
                                           ex::when_all(sub_task_chain(),
                                                        ex::just(std::vector<int>(to_sizet(nranks), 0))));
}

// Note: Matrices used for communication @p send_mat and @p recv_mat are assumed to be in column-major layout!
//
template <class T, Device D, Coord C, class SendCountsSender, class RecvCountsSender>
void all2allData(common::Pipeline<comm::Communicator>& sub_task_chain, int nranks,
                 LocalElementSize sz_loc, SendCountsSender&& send_counts_sender, Matrix<T, D>& send_mat,
                 RecvCountsSender&& recv_counts_sender, Matrix<T, D>& recv_mat) {
  namespace ex = pika::execution::experimental;

  auto all2all_f =
      [len = sz_loc.get<orthogonal(C)>()](const comm::Communicator& comm, std::vector<int>& send_counts,
                                          std::vector<int>& send_displs,
                                          const std::vector<matrix::Tile<T, D>>& send_tiles,
                                          std::vector<int>& recv_counts, std::vector<int>& recv_displs,
                                          const std::vector<matrix::Tile<T, D>>& recv_tiles,
                                          MPI_Request* req) {
        // datatype to be sent to each rank
        MPI_Datatype dtype = dlaf::comm::mpi_datatype<std::remove_pointer_t<T>>::type;

        // scale by the length of the row or column vectors to get the number of elements sent to each process
        auto mul_const = [len](int num) { return to_int(len) * num; };
        std::transform(send_counts.cbegin(), send_counts.cend(), send_counts.begin(), mul_const);
        std::transform(recv_counts.cbegin(), recv_counts.cend(), recv_counts.begin(), mul_const);

        // Note: that was guaranteed to be contiguous on allocation
        T* send_ptr = send_tiles[0].ptr();
        T* recv_ptr = recv_tiles[0].ptr();

        // send displacements
        std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), 0);

        // recv displacements
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), 0);

        // All-to-All communication
        DLAF_MPI_CHECK_ERROR(MPI_Ialltoallv(send_ptr, send_counts.data(), send_displs.data(), dtype,
                                            recv_ptr, recv_counts.data(), recv_displs.data(), dtype,
                                            comm, req));
      };

  auto sender =
      ex::when_all(sub_task_chain(), std::forward<SendCountsSender>(send_counts_sender),
                   ex::just(std::vector<int>(to_sizet(nranks))), whenAllReadWriteTilesArray(send_mat),
                   std::forward<RecvCountsSender>(recv_counts_sender),
                   ex::just(std::vector<int>(to_sizet(nranks))), whenAllReadWriteTilesArray(recv_mat));
  dlaf::comm::internal::transformMPIDetach(std::move(all2all_f), std::move(sender));
}

// @param nranks number of ranks
// @param loc2gl_index  a column matrix that represents a map from local to global indices
// @param packing_index a column matrix that represents a map from packed indices to local indices
//        that is used for packing columns or rows on each rank
//
// Note: the order of the packed rows or columns on the send side must match the expected order at
// unpacking on the receive side
template <Coord C, bool PackBasedOnGlobalIndex>
auto initPackingIndex(int nranks, SizeType i_el_begin, const matrix::Distribution& dist,
                      Matrix<const SizeType, Device::CPU>& loc2gl_index,
                      Matrix<SizeType, Device::CPU>& packing_index) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  auto counts_fn = [nranks, i_el_begin, dist,
                    len = packing_index.size().rows()](const auto& loc2gl_index_tiles,
                                                       const auto& packing_index_tiles) {
    const SizeType* in = loc2gl_index_tiles[0].get().ptr();
    SizeType* out = packing_index_tiles[0].ptr();

    std::vector<int> counts(to_sizet(nranks));
    int offset = 0;
    for (int rank = 0; rank < nranks; ++rank) {
      int& count = counts[to_sizet(rank)];
      count = 0;
      for (SizeType i = 0; i < len; ++i) {
        if (dist.rankGlobalElement<C>(i_el_begin + in[i]) == rank) {
          if constexpr (PackBasedOnGlobalIndex) {
            out[offset + count] = i;
          }
          else {
            out[i] = offset + count;
          }
          ++count;
        }
      }
      if constexpr (PackBasedOnGlobalIndex) {
        std::sort(out + offset, out + offset + count,
                  [in](SizeType i1, SizeType i2) { return in[i1] < in[i2]; });
      }
      offset += count;
    }
    return counts;
  };

  auto sender =
      ex::when_all(whenAllReadOnlyTilesArray(loc2gl_index), whenAllReadWriteTilesArray(packing_index));
  return ex::ensure_started(
      di::transform(di::Policy<Backend::MC>{}, std::move(counts_fn), std::move(sender)));
}

// Copies index tiles belonging to the current process from the complete index @p global_index into the
// partial index containing only the local parts @p local_index.
template <Device D, Coord C>
void copyLocalPartsFromGlobalIndex(SizeType i_loc_begin, const matrix::Distribution& dist,
                                   Matrix<const SizeType, D>& global_index,
                                   Matrix<SizeType, D>& local_index) {
  namespace ex = pika::execution::experimental;

  for (auto tile_wrt_local : common::iterate_range2d(local_index.distribution().localNrTiles())) {
    SizeType i_gl_tile = dist.globalTileFromLocalTile<C>(i_loc_begin + tile_wrt_local.row());
    ex::start_detached(ex::when_all(global_index.read(GlobalTileIndex(i_gl_tile, 0)),
                                    local_index.readwrite(tile_wrt_local)) |
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
    transposeTileSenders(mat_in.read(i_in_tile), mat_out.readwrite(i_out_tile));
  }
}

// Transposes the local part of the distributed matrix @p mat_in into the local matrix @p mat_out.
template <class T>
void transposeFromDistributedToLocalMatrix(LocalTileIndex i_loc_begin,
                                           Matrix<const T, Device::CPU>& mat_in,
                                           Matrix<T, Device::CPU>& mat_out) {
  for (auto i_out_tile : common::iterate_range2d(mat_out.distribution().localNrTiles())) {
    LocalTileIndex i_in_tile(i_loc_begin.row() + i_out_tile.col(), i_loc_begin.col() + i_out_tile.row());
    transposeTileSenders(mat_in.read(i_in_tile), mat_out.readwrite(i_out_tile));
  }
}

// Inverts the the subset of tiles [ @p i_begin, @p i_end (including)] of the index map @p in and saves
// the result into @p out.
inline void invertIndex(SizeType i_begin, SizeType i_end, Matrix<const SizeType, Device::CPU>& in,
                        Matrix<SizeType, Device::CPU>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;
  namespace ut = matrix::util;

  const matrix::Distribution& dist = in.distribution();
  SizeType nb = dist.blockSize().rows();
  SizeType nbr = dist.tileSize(GlobalTileIndex(i_end, 0)).rows();
  SizeType n = (i_end - i_begin) * nb + nbr;
  auto inv_fn = [n](const auto& in_tiles_futs, const auto& out_tiles) {
    TileElementIndex zero(0, 0);
    const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero);
    SizeType* out_ptr = out_tiles[0].ptr(zero);
    for (SizeType i = 0; i < n; ++i) {
      out_ptr[in_ptr[i]] = i;
    }
  };

  LocalTileIndex begin{i_begin, 0};
  LocalTileSize sz{i_end - i_begin + 1, 1};
  auto sender =
      ex::when_all(ex::when_all_vector(matrix::selectRead(in, common::iterate_range2d(begin, sz))),
                   ex::when_all_vector(matrix::select(out, common::iterate_range2d(begin, sz))));
  ex::start_detached(di::transform(di::Policy<Backend::MC>(), std::move(inv_fn), std::move(sender)));
}

template <class T, Coord C>
void permuteOnCPU(common::Pipeline<comm::Communicator>& sub_task_chain, SizeType i_begin, SizeType i_end,
                  Matrix<const SizeType, Device::CPU>& perms, Matrix<T, Device::CPU>& mat_in,
                  Matrix<T, Device::CPU>& mat_out) {
  constexpr Device D = Device::CPU;

  const matrix::Distribution& dist = mat_in.distribution();
  int nranks = to_int(dist.commGridSize().get<C>());

  // Local size and index of subproblem [i_begin, i_end]
  SizeType i_el_begin = dist.globalElementFromGlobalTileAndTileElement<C>(i_begin, 0);
  TileElementSize blk = dist.blockSize();
  LocalTileIndex i_loc_begin{dist.nextLocalTileFromGlobalTile<Coord::Row>(i_begin),
                             dist.nextLocalTileFromGlobalTile<Coord::Col>(i_begin)};
  LocalTileIndex i_loc_end{dist.nextLocalTileFromGlobalTile<Coord::Row>(i_end + 1) - 1,
                           dist.nextLocalTileFromGlobalTile<Coord::Col>(i_end + 1) - 1};
  // Note: the local shape of the permutation region may not be square if the process grid is not square
  LocalElementSize sz_loc{dist.localElementDistanceFromGlobalTile<Coord::Row>(i_begin, i_end + 1),
                          dist.localElementDistanceFromGlobalTile<Coord::Col>(i_begin, i_end + 1)};

  // If there are no tiles in this rank, participate in the all2all call and return
  if (sz_loc.isEmpty()) {
    all2allEmptyData<T>(sub_task_chain, nranks);
    return;
  }

  // Create a map from send indices to receive indices (inverse of perms)
  Matrix<SizeType, D> inverse_perms(perms.distribution());
  invertIndex(i_begin, i_end, perms, inverse_perms);

  // Local distribution used for packing and unpacking
  matrix::Distribution subm_dist(sz_loc, blk);

  // Local single tile column matrices representing index maps used for packing and unpacking of
  // communication data
  matrix::Distribution index_dist(LocalElementSize(sz_loc.get<C>(), 1), TileElementSize(blk.rows(), 1));
  Matrix<SizeType, D> ws_index(index_dist);
  Matrix<SizeType, D> packing_index(index_dist);
  Matrix<SizeType, D> unpacking_index(index_dist);

  // Local matrices used for packing data for communication. Both matrices are in column-major order.
  // The particular constructor is used on purpose to guarantee that columns are stored contiguosly,
  // such that there is no padding and gaps between them.
  LocalElementSize comm_sz = (C == Coord::Col) ? sz_loc : transposed(sz_loc);
  matrix::Distribution comm_dist(comm_sz, blk);
  matrix::LayoutInfo comm_layout = matrix::colMajorLayout(comm_sz, blk, comm_sz.rows());
  Matrix<T, D> mat_send(comm_dist, comm_layout);
  Matrix<T, D> mat_recv(comm_dist, comm_layout);

  // Initialize the unpacking index
  copyLocalPartsFromGlobalIndex<D, C>(i_loc_begin.get<C>(), dist, perms, ws_index);
  auto recv_counts_sender =
      initPackingIndex<C, false>(nranks, i_el_begin, dist, ws_index, unpacking_index);

  // Initialize the packing index
  // Here `true` is specified so that the send side matches the order of columns/rows on the receive side
  copyLocalPartsFromGlobalIndex<D, C>(i_loc_begin.get<C>(), dist, inverse_perms, ws_index);
  auto send_counts_sender = initPackingIndex<C, true>(nranks, i_el_begin, dist, ws_index, packing_index);

  // Pack local rows or columns to be sent from this rank
  applyPackingIndex<T, D, C>(subm_dist, whenAllReadOnlyTilesArray(packing_index),
                             whenAllReadWriteTilesArray(i_loc_begin, i_loc_end, mat_in),
                             (C == Coord::Col)
                                 ? whenAllReadWriteTilesArray(mat_send)
                                 : whenAllReadWriteTilesArray(i_loc_begin, i_loc_end, mat_out));

  // Pack data into the contiguous column-major send buffer by transposing
  if constexpr (C == Coord::Row) {
    transposeFromDistributedToLocalMatrix(i_loc_begin, mat_out, mat_send);
  }

  // Communicate data
  all2allData<T, D, C>(sub_task_chain, nranks, sz_loc, std::move(send_counts_sender), mat_send,
                       std::move(recv_counts_sender), mat_recv);

  // Unpack data from the contiguous column-major receive buffer by transposing
  if constexpr (C == Coord::Row) {
    transposeFromLocalToDistributedMatrix(i_loc_begin, mat_recv, mat_in);
  }

  // Unpack local rows or columns received on this rank
  applyPackingIndex<T, D, C>(subm_dist, whenAllReadOnlyTilesArray(unpacking_index),
                             (C == Coord::Col)
                                 ? whenAllReadWriteTilesArray(mat_recv)
                                 : whenAllReadWriteTilesArray(i_loc_begin, i_loc_end, mat_in),
                             whenAllReadWriteTilesArray(i_loc_begin, i_loc_end, mat_out));
}

template <Backend B, Device D, class T, Coord C>
void Permutations<B, D, T, C>::call(common::Pipeline<comm::Communicator>& sub_task_chain,
                                    SizeType i_begin, SizeType i_end, Matrix<const SizeType, D>& perms,
                                    Matrix<T, D>& mat_in, Matrix<T, D>& mat_out) {
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
