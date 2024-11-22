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

#include <blas.hh>

#include <pika/barrier.hpp>
#include <pika/execution.hpp>
#include <pika/execution/algorithms/when_all_vector.hpp>

#include <dlaf/blas/tile_extensions.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/data.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/eigensolver/internal/get_red2band_panel_nworkers.h>
#include <dlaf/factorization/qr/api.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/matrix/views.h>
#include <dlaf/types.h>
#include <dlaf/util_matrix.h>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>

#include <dlaf/blas/tile.h>
#include <dlaf/lapack/gpu/larft.h>
#endif

namespace dlaf::factorization::internal {

namespace tfactor_l {
template <Backend backend, Device device, class T>
struct Helpers {};

template <class T>
struct Helpers<Backend::MC, Device::CPU, T> {
  template <class TSender>
  static auto set0(TSender&& t) {
    namespace ex = pika::execution::experimental;

    return dlaf::internal::transform(
        dlaf::internal::Policy<Backend::MC>(pika::execution::thread_priority::high),
        [](matrix::Tile<T, Device::CPU>&& tile) {
          tile::internal::set0<T>(tile);
          return std::move(tile);
        },
        std::forward<TSender>(t));
  }

  static matrix::Tile<T, Device::CPU> gemv_func(const SizeType first_row_tile,
                                                const matrix::Tile<const T, Device::CPU>& tile_v,
                                                const matrix::Tile<const T, Device::CPU>& taus,
                                                matrix::Tile<T, Device::CPU>&& tile_t) noexcept {
    const SizeType k = tile_t.size().cols();
    DLAF_ASSERT(tile_v.size().cols() == k, tile_v.size().cols(), k);
    DLAF_ASSERT(taus.size().rows() == k, taus.size().rows(), k);

    common::internal::SingleThreadedBlasScope single;
    for (SizeType j = 0; j < k; ++j) {
      const T tau = taus({j, 0});

      const TileElementIndex t_start{0, j};

      // Position of the 1 in the diagonal in the current column.
      const SizeType i_diag = j - first_row_tile;

      // Break if the reflector starts in the next tile.
      if (i_diag >= tile_v.size().rows())
        break;

      const SizeType first_element_in_col = std::max<SizeType>(0, i_diag);

      // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
      // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
      const TileElementIndex va_start{first_element_in_col, 0};
      const TileElementIndex vb_start{first_element_in_col, j};
      const TileElementSize va_size{tile_v.size().rows() - first_element_in_col, j};

      if (i_diag >= 0) {
        tile_t({j, j}) = tau;
      }

      blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, va_size.rows(), va_size.cols(), -tau,
                 tile_v.ptr(va_start), tile_v.ld(), tile_v.ptr(vb_start), 1, T(1), tile_t.ptr(t_start),
                 1);
    }
    return std::move(tile_t);
  }

  static auto gemvColumnT(const SizeType first_row_tile,
                          matrix::ReadOnlyTileSender<T, Device::CPU> tile_vi,
                          matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                          matrix::ReadWriteTileSender<T, Device::CPU>&& tile_t) {
    namespace ex = pika::execution::experimental;

    return ex::when_all(ex::just(first_row_tile), tile_vi, std::move(taus), std::move(tile_t)) |
           dlaf::internal::transform(
               dlaf::internal::Policy<Backend::MC>(pika::execution::thread_priority::high), gemv_func);
  }

  // Update each column (in order) t = T . t
  // remember that T is upper triangular, so it is possible to use TRMV
  static matrix::Tile<T, Device::CPU> trmv_func(matrix::Tile<T, Device::CPU>&& tile_t) {
    common::internal::SingleThreadedBlasScope single;
    for (SizeType j = 0; j < tile_t.size().cols(); ++j) {
      const TileElementIndex t_start{0, j};
      const TileElementSize t_size{j, 1};

      blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
                 t_size.rows(), tile_t.ptr(), tile_t.ld(), tile_t.ptr(t_start), 1);
    }
    // TODO: Why return if the tile is unused?
    return std::move(tile_t);
  }

  template <typename TSender>
  static auto trmvUpdateColumn(TSender&& tile_t) noexcept {
    return std::forward<TSender>(tile_t) |
           dlaf::internal::transform(
               dlaf::internal::Policy<Backend::MC>(pika::execution::thread_priority::high), trmv_func);
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct Helpers<Backend::GPU, Device::GPU, T> {
  template <class TSender>
  static auto set0(TSender&& t) {
    namespace ex = pika::execution::experimental;

    return dlaf::internal::transform(
        dlaf::internal::Policy<Backend::GPU>(pika::execution::thread_priority::high),
        [](matrix::Tile<T, Device::GPU>& tile, whip::stream_t stream) {
          tile::internal::set0<T>(tile, stream);
          return std::move(tile);
        },
        std::forward<TSender>(t));
  }

  static matrix::Tile<T, Device::GPU> gemv_func(cublasHandle_t handle, const SizeType first_row_tile,
                                                const matrix::Tile<const T, Device::GPU>& tile_v,
                                                const matrix::Tile<const T, Device::CPU>& taus,
                                                matrix::Tile<T, Device::GPU>& tile_t, SizeType begin = 0,
                                                SizeType end = -1) noexcept {
    const SizeType k = tile_t.size().cols();
    if (end == -1)
      end = k;
    DLAF_ASSERT(tile_v.size().cols() == k, tile_v.size().cols(), k);
    DLAF_ASSERT(taus.size().rows() == k, taus.size().rows(), k);
    DLAF_ASSERT(taus.size().cols() == 1, taus.size().cols());

    for (SizeType j = begin; j < end; ++j) {
      const auto mtau = util::blasToCublasCast(-taus({j, 0}));
      const auto one = util::blasToCublasCast(T{1});

      const TileElementIndex t_start{0, j};

      // Position of the 1 in the diagonal in the current column.
      SizeType i_diag = j - first_row_tile;
      const SizeType first_element_in_col = std::max<SizeType>(0, i_diag);

      // Break if the reflector starts in the next tile.
      if (i_diag >= tile_v.size().rows())
        break;

      // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
      // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
      TileElementIndex va_start{first_element_in_col, 0};
      TileElementIndex vb_start{first_element_in_col, j};
      TileElementSize va_size{tile_v.size().rows() - first_element_in_col, j};

      gpublas::internal::Gemv<T>::call(handle, CUBLAS_OP_C, to_int(va_size.rows()),
                                       to_int(va_size.cols()), &mtau,
                                       util::blasToCublasCast(tile_v.ptr(va_start)), to_int(tile_v.ld()),
                                       util::blasToCublasCast(tile_v.ptr(vb_start)), 1, &one,
                                       util::blasToCublasCast(tile_t.ptr(t_start)), 1);
    }
    return std::move(tile_t);
  }

  static auto gemvColumnT(const SizeType first_row_tile,
                          matrix::ReadOnlyTileSender<T, Device::GPU> tile_vi,
                          matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                          matrix::ReadWriteTileSender<T, Device::GPU>&& tile_t) noexcept {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    return ex::when_all(ex::just(first_row_tile), std::move(tile_vi), std::move(taus), std::move(tile_t),
                        ex::just(0), ex::just(-1)) |
           di::transform<di::TransformDispatchType::Blas>(
               di::Policy<Backend::GPU>(pika::execution::thread_priority::high), gemv_func);
  }

  // Update each column (in order) t = T . t
  // remember that T is upper triangular, so it is possible to use TRMV
  static matrix::Tile<T, Device::GPU> trmv_func(cublasHandle_t handle,
                                                matrix::Tile<T, Device::GPU>& tile_t) {
    for (SizeType j = 0; j < tile_t.size().cols(); ++j) {
      const TileElementIndex t_start{0, j};
      const TileElementSize t_size{j, 1};

      gpublas::internal::Trmv<T>::call(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                       to_int(t_size.rows()), util::blasToCublasCast(tile_t.ptr()),
                                       to_int(tile_t.ld()), util::blasToCublasCast(tile_t.ptr(t_start)),
                                       1);
    }
    return std::move(tile_t);
  }

  static auto trmvUpdateColumn(matrix::ReadWriteTileSender<T, Device::GPU>&& tile_t) noexcept {
    namespace ex = pika::execution::experimental;

    return std::move(tile_t) |
           dlaf::internal::transform<dlaf::internal::TransformDispatchType::Blas>(
               dlaf::internal::Policy<Backend::GPU>(pika::execution::thread_priority::high), trmv_func);
  }
};
#endif
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                                          matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                                          matrix::ReadWriteTileSender<T, device> t,
                                          std::vector<matrix::ReadWriteTileSender<T, device>> ws_t) {
  using pika::execution::thread_priority;
  using Helpers = tfactor_l::Helpers<backend, device, T>;

  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  // Fast return in case of no reflectors
  if (hh_panel.getWidth() == 0)
    return;

  const SizeType bs = hh_panel.parentDistribution().blockSize().rows();
  const SizeType offset_lc = (bs - hh_panel.tile_size_of_local_head().rows());

  std::vector<matrix::ReadOnlyTileSender<T, device>> hh_tiles;
  std::vector<SizeType> first_rows;

  for (const auto& v_i : hh_panel.iteratorLocal()) {
    hh_tiles.emplace_back(hh_panel.read(v_i));

    const SizeType first_row_tile =
        std::max<SizeType>(0, (v_i.row() - hh_panel.rangeStartLocal()) * bs - offset_lc);
    first_rows.push_back(first_row_tile);
  }
  if constexpr (backend == Backend::MC) {
    const auto hp_scheduler = di::getBackendScheduler<Backend::MC>(thread_priority::high);
    ex::start_detached(
        di::whenAllLift(std::move(first_rows), ex::when_all_vector(std::move(hh_tiles)), taus,
                        std::move(t), ex::when_all_vector(std::move(ws_t))) |
        di::continues_on(hp_scheduler) |
        ex::let_value([hp_scheduler](const std::vector<SizeType>& first_rows, auto&& hh_tiles,
                                     auto&& taus, auto&& t, auto&& ws_t) {
          const std::size_t nworkers = eigensolver::internal::getReductionToBandPanelNWorkers();
          const std::chrono::duration<double> barrier_busy_wait = std::chrono::microseconds(1000);

          DLAF_ASSERT(nworkers == ws_t.size(), nworkers, ws_t.size());

          return ex::just(std::make_unique<pika::barrier<>>(nworkers)) | di::continues_on(hp_scheduler) |
                 ex::bulk(nworkers, [=, &first_rows, &hh_tiles, &taus, &t,
                                     &ws_t](const std::size_t worker_id, auto& barrier_ptr) mutable {
                   const std::size_t batch_size = util::ceilDiv(hh_tiles.size(), nworkers);
                   const std::size_t begin = worker_id * batch_size;
                   const std::size_t end =
                       std::min(worker_id * batch_size + batch_size, hh_tiles.size());

                   auto&& ws_worker = ws_t[worker_id];
                   tile::internal::set0<T>(ws_worker);

                   // make it work on worker_id section of tiles
                   for (std::size_t index = begin; index < end; ++index) {
                     auto&& tile_snd = hh_tiles[index];
                     ws_worker = Helpers::gemv_func(first_rows[index], tile_snd.get(), taus.get(),
                                                    std::move(ws_worker));
                   }
                   barrier_ptr->arrive_and_wait(barrier_busy_wait);

                   // reduce ws_T in t
                   if (worker_id == 0) {
                     tile::internal::set0<T>(t);
                     for (std::size_t other_worker = 0; other_worker < nworkers; ++other_worker) {
                       tile::internal::add(T(1), ws_t[other_worker], t);
                     }
                     Helpers::trmv_func(std::move(t));
                   }
                 });
        }));
  }
  else if constexpr (backend == Backend::GPU) {
    dlaf::internal::silenceUnusedWarningFor(first_rows);

    const auto hp_scheduler = di::getBackendScheduler<Backend::GPU>(thread_priority::high);
    t = di::whenAllLift(ex::when_all_vector(std::move(hh_tiles)), taus, std::move(t),
                        ex::when_all_vector(std::move(ws_t))) |
        di::transform<dlaf::internal::TransformDispatchType::Plain>(
            di::Policy<Backend::GPU>(thread_priority::high),
            [](auto&& hh_tiles, auto&& taus, matrix::Tile<T, Device::GPU>& tile_t, auto&& /*ws_t*/,
               whip::stream_t stream) {
              const SizeType k = tile_t.size().cols();

              // Note:
              // prepare the diagonal of taus in t
              whip::memset_2d_async(tile_t.ptr(), sizeof(T) * to_sizet(tile_t.ld()), 0,
                                    sizeof(T) * to_sizet(k), to_sizet(k), stream);
              whip::memcpy_2d_async(tile_t.ptr(), to_sizet(tile_t.ld() + 1) * sizeof(T), taus.ptr(),
                                    sizeof(T), sizeof(T), to_sizet(k), whip::memcpy_host_to_device,
                                    stream);

              // Note:
              // - call one gemv per tile
              // - being on the same stream, they are already serialised on GPU
              for (std::size_t index = 0; index < hh_tiles.size(); ++index) {
                const matrix::Tile<const T, Device::GPU>& tile_v = hh_tiles[index].get();
                gpulapack::larft_gemv1202(tile_v.size().rows(), k, tile_v.ptr(), tile_v.ld(),
                                          tile_t.ptr(), tile_t.ld(), stream);
              }

              return std::move(tile_t);
            });

    // 2nd step: compute the T factor, by performing the last step on each column
    // each column depends on the previous part (all reflectors that comes before)
    // so it is performed sequentially
    ex::start_detached(Helpers::trmvUpdateColumn(std::move(t)));
  }
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(
    matrix::Panel<Coord::Col, T, device>& hh_panel, matrix::ReadOnlyTileSender<T, Device::CPU> taus,
    matrix::ReadWriteTileSender<T, device> t,
    comm::CommunicatorPipeline<comm::CommunicatorType::Col>& mpi_col_task_chain) {
  namespace ex = pika::execution::experimental;

  using Helpers = tfactor_l::Helpers<backend, device, T>;

  // Fast return in case of no reflectors
  if (hh_panel.getWidth() == 0)
    return;

  const auto v_start = hh_panel.offsetElement();
  auto dist = hh_panel.parentDistribution();

  matrix::ReadWriteTileSender<T, device> t_local = Helpers::set0(std::move(t));

  // TODO FIXME this has been removed in the GPU during refactoring
  //  const SizeType k = t.size().cols();
  //  whip::memcpy_2d_async(t.ptr(), to_sizet(t.ld() + 1) * sizeof(T), taus.ptr(), sizeof(T), sizeof(T),
  //                        to_sizet(k), whip::memcpy_host_to_device, stream);

  // Note:
  // T factor is an upper triangular square matrix, built column by column
  // with taus values on the diagonal
  //
  // T(j,j) = tau(j)
  //
  // and in the upper triangular part the following formula applies
  //
  // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
  //
  //
  // The result is achieved in two main steps:
  // 1) t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) T(0:j, j) = T(0:j, 0:j) . t

  // 1st step: compute the column partial result `t`
  // First we compute the matrix vector multiplication for each column
  // -tau(j) . V(j:, 0:j)* . V(j:, j)
  for (const auto& v_i_loc : hh_panel.iteratorLocal()) {
    const SizeType v_i = dist.template globalTileFromLocalTile<Coord::Row>(v_i_loc.row());
    const SizeType first_row_tile = std::max<SizeType>(0, v_i * dist.blockSize().rows() - v_start);

    // TODO
    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    t_local = Helpers::gemvColumnT(first_row_tile, hh_panel.read(v_i_loc), taus, std::move(t_local));
  }

  // at this point each rank has its partial result for each column
  // so, let's reduce the results (on all ranks, so that everyone can independently compute T factor)
  if (mpi_col_task_chain.size() > 1)
    t_local = dlaf::comm::schedule_all_reduce_in_place(mpi_col_task_chain.exclusive(), MPI_SUM,
                                                       std::move(t_local));

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  ex::start_detached(Helpers::trmvUpdateColumn(std::move(t_local)));
}

}
