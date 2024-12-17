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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <utility>
#include <vector>

#include <blas.hh>

#include <pika/barrier.hpp>
#include <pika/execution.hpp>

#include <dlaf/blas/tile_extensions.h>
#include <dlaf/common/assert.h>
#include <dlaf/common/data.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/factorization/qr/api.h>
#include <dlaf/factorization/qr/internal/get_tfactor_nworkers.h>
#include <dlaf/lapack/gpu/larft.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/matrix/views.h>
#include <dlaf/types.h>
#include <dlaf/util_math.h>
#include <dlaf/util_matrix.h>

#ifdef DLAF_WITH_GPU
#include <whip.hpp>

#include <dlaf/blas/tile.h>
#endif

namespace dlaf::factorization::internal {

namespace tfactor_l {

inline std::size_t num_workers_gemv(const std::size_t nrtiles) {
  const std::size_t min_workers = 1;
  const std::size_t available_workers = get_tfactor_nworkers();
  const std::size_t ideal_workers = util::ceilDiv(to_sizet(nrtiles), to_sizet(2));
  return std::clamp(ideal_workers, min_workers, available_workers);
}

template <Backend backend, Device device, class T>
struct Helpers {};

template <class T>
struct Helpers<Backend::MC, Device::CPU, T> {
  static auto set0_and_return(matrix::ReadWriteTileSender<T, Device::CPU> tile_t) {
    namespace di = dlaf::internal;
    return std::move(tile_t) |
           di::transform(di::Policy<Backend::MC>(pika::execution::thread_priority::high),
                         [](matrix::Tile<T, Device::CPU>&& tile_t) {
                           tile::internal::set0<T>(tile_t);
                           return std::move(tile_t);
                         });
  }

  static void loop_gemv(const matrix::Tile<const T, Device::CPU>& tile_v,
                        const matrix::Tile<const T, Device::CPU>& taus,
                        const matrix::Tile<T, Device::CPU>& tile_t) noexcept {
    const SizeType k = tile_t.size().cols();

    DLAF_ASSERT(tile_v.size().cols() == k, tile_v.size().cols(), k);
    DLAF_ASSERT(taus.size().rows() == k, taus.size().rows(), k);

    common::internal::SingleThreadedBlasScope single;
    for (SizeType j = 0; j < k; ++j) {
      // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
      // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
      const TileElementIndex t_start{0, j};
      const TileElementIndex va_start{0, 0};
      const TileElementIndex vb_start{0, j};
      const TileElementSize va_size{tile_v.size().rows(), j};
      const T tau = tile_t({j, j});

      blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, va_size.rows(), va_size.cols(), -tau,
                 tile_v.ptr(va_start), tile_v.ld(), tile_v.ptr(vb_start), 1, 1, tile_t.ptr(t_start), 1);
    }
  }

  static matrix::ReadWriteTileSender<T, Device::CPU> step_gemv(
      matrix::Panel<Coord::Col, T, Device::CPU>& hh_panel,
      matrix::ReadOnlyTileSender<T, Device::CPU> taus,
      matrix::ReadWriteTileSender<T, Device::CPU> tile_t,
      std::vector<matrix::ReadWriteTileSender<T, Device::CPU>> workspaces) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    using pika::execution::thread_priority;

    std::vector<matrix::ReadOnlyTileSender<T, Device::CPU>> hh_tiles =
        selectRead(hh_panel, hh_panel.iteratorLocal());

    const std::size_t nworkers = num_workers_gemv(hh_tiles.size());
    DLAF_ASSERT(workspaces.size() >= nworkers - 1, workspaces.size(), nworkers - 1);

    const auto hp_scheduler = di::getBackendScheduler<Backend::MC>(thread_priority::high);
    return ex::when_all(ex::when_all_vector(std::move(hh_tiles)), std::move(taus), std::move(tile_t),
                        ex::when_all_vector(std::move(workspaces))) |
           di::continues_on(hp_scheduler) |
           ex::let_value([hp_scheduler, nworkers](auto&& hh_tiles, auto&& taus, auto&& tile_t,
                                                  auto&& workspaces) {
             const std::chrono::duration<double> barrier_busy_wait = std::chrono::microseconds(1000);

             return ex::just(std::make_unique<pika::barrier<>>(nworkers)) |
                    di::continues_on(hp_scheduler) |
                    ex::bulk(nworkers,
                             [=, &hh_tiles, &taus, &tile_t, &workspaces](const std::size_t worker_id,
                                                                         auto& barrier_ptr) mutable {
                               const SizeType k = taus.get().size().rows();

                               const std::size_t batch_size = util::ceilDiv(hh_tiles.size(), nworkers);
                               const std::size_t begin = worker_id * batch_size;
                               const std::size_t end =
                                   std::min(worker_id * batch_size + batch_size, hh_tiles.size());

                               const matrix::Tile<T, Device::CPU>& ws_worker =
                                   worker_id == 0 ? tile_t : workspaces[worker_id - 1];

                               tile::internal::set0<T>(ws_worker);
                               lapack::lacpy(blas::Uplo::General, 1, k, taus.get().ptr(), 1,
                                             ws_worker.ptr(), ws_worker.ld() + 1);

                               // make it work on worker_id section of tiles
                               for (std::size_t index = begin; index < end; ++index) {
                                 const matrix::Tile<const T, Device::CPU>& tile_v =
                                     hh_tiles[index].get();
                                 loop_gemv(tile_v, taus.get(), ws_worker);
                               }

                               barrier_ptr->arrive_and_wait(barrier_busy_wait);

                               // reduce ws_T in tile_t
                               if (worker_id == 0) {
                                 for (std::size_t other_worker = 1; other_worker < nworkers;
                                      ++other_worker) {
                                   tile::internal::add(T(1), workspaces[other_worker - 1], tile_t);
                                 }
                               }
                             }) |
                    // Note: drop the barrier sent by the bulk and return tile_t
                    ex::then([&tile_t](auto&&) mutable { return std::move(tile_t); });
           });
  }

  static void loop_trmv(const matrix::Tile<T, Device::CPU>& tile_t) {
    common::internal::SingleThreadedBlasScope single;

    const SizeType k = tile_t.size().cols();

    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    for (SizeType j = 0; j < k; ++j) {
      const TileElementIndex t_start{0, j};
      const TileElementSize t_size{j, 1};

      blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
                 t_size.rows(), tile_t.ptr(), tile_t.ld(), tile_t.ptr(t_start), 1);
    }
  }

  static auto step_copy_diag_and_trmv(matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                                      matrix::ReadWriteTileSender<T, Device::CPU> tile_t) noexcept {
    auto tausdiag_trmvloop = [](const matrix::Tile<const T, Device::CPU>& taus,
                                matrix::Tile<T, Device::CPU> tile_t) {
      common::internal::SingleThreadedBlasScope single;

      const SizeType k = tile_t.size().cols();
      lapack::lacpy(blas::Uplo::General, 1, k, taus.ptr(), 1, tile_t.ptr(), tile_t.ld() + 1);

      loop_trmv(tile_t);
    };

    namespace di = dlaf::internal;
    namespace ex = pika::execution::experimental;

    return ex::when_all(std::move(taus), std::move(tile_t)) |
           di::transform(di::Policy<Backend::MC>(pika::execution::thread_priority::high),
                         std::move(tausdiag_trmvloop));
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct Helpers<Backend::GPU, Device::GPU, T> {
  static auto set0_and_return(matrix::ReadWriteTileSender<T, Device::GPU> tile_t) {
    namespace di = dlaf::internal;
    return std::move(tile_t) |
           di::transform(di::Policy<Backend::GPU>(pika::execution::thread_priority::high),
                         [](matrix::Tile<T, Device::GPU>& tile_t, whip::stream_t stream) {
                           tile::internal::set0<T>(tile_t, stream);
                           return std::move(tile_t);
                         });
  }
  static void loop_gemv(cublasHandle_t handle, const matrix::Tile<const T, Device::GPU>& tile_v,
                        const matrix::Tile<const T, Device::CPU>& taus,
                        const matrix::Tile<T, Device::GPU>& tile_t) noexcept {
    const SizeType m = tile_v.size().rows();
    const SizeType k = tile_t.size().cols();
    DLAF_ASSERT(tile_v.size().cols() == k, tile_v.size().cols(), k);
    DLAF_ASSERT(taus.size().rows() == k, taus.size().rows(), k);
    DLAF_ASSERT(taus.size().cols() == 1, taus.size().cols());

    gpulapack::larft_gemv0(handle, m, k, tile_v.ptr(), tile_v.ld(), taus.ptr(), tile_t.ptr(),
                           tile_t.ld());
  }

  static matrix::ReadWriteTileSender<T, Device::GPU> step_gemv(
      matrix::Panel<Coord::Col, T, Device::GPU>& hh_panel,
      matrix::ReadOnlyTileSender<T, Device::CPU> taus,
      matrix::ReadWriteTileSender<T, Device::GPU> tile_t,
      std::vector<matrix::ReadWriteTileSender<T, Device::GPU>> workspaces) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    using pika::execution::thread_priority;

    std::vector<matrix::ReadOnlyTileSender<T, Device::GPU>> hh_tiles =
        selectRead(hh_panel, hh_panel.iteratorLocal());

    const std::size_t nworkers = num_workers_gemv(hh_tiles.size());
    DLAF_ASSERT(workspaces.size() >= nworkers - 1, workspaces.size(), nworkers - 1);

    const std::size_t batch_size = util::ceilDiv(hh_tiles.size(), nworkers);
    for (std::size_t id_worker = 0; id_worker < nworkers; ++id_worker) {
      const std::size_t begin = id_worker * batch_size;
      const std::size_t end = std::min(hh_tiles.size(), (id_worker + 1) * batch_size);

      if (end - begin <= 0)
        continue;

      std::vector<matrix::ReadOnlyTileSender<T, Device::GPU>> input_tiles;
      for (std::size_t sub = begin; sub < end; ++sub)
        input_tiles.emplace_back(std::move(hh_tiles[sub]));

      matrix::ReadWriteTileSender<T, Device::GPU>& workspace =
          id_worker == 0 ? tile_t : workspaces[id_worker - 1];

      workspace =
          di::whenAllLift(ex::when_all_vector(std::move(input_tiles)), taus, std::move(workspace)) |
          di::transform<dlaf::internal::TransformDispatchType::Blas>(
              di::Policy<Backend::GPU>(thread_priority::high),
              [](cublasHandle_t handle, auto&& hh_tiles, auto&& taus,
                 matrix::Tile<T, Device::GPU>& tile_t) {
                const SizeType k = tile_t.size().cols();

                // Note:
                // prepare the diagonal of taus in t and reset the rest
                whip::stream_t stream;
                DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));

                whip::memset_2d_async(tile_t.ptr(), sizeof(T) * to_sizet(tile_t.ld()), 0,
                                      sizeof(T) * to_sizet(k), to_sizet(k), stream);
                gpulapack::lacpy(blas::Uplo::General, 1, k, taus.ptr(), 1, tile_t.ptr(), tile_t.ld() + 1,
                                 stream);

                // Note:
                // - call one gemv per tile
                // - being on the same stream, they are already serialised on GPU
                for (std::size_t index = 0; index < hh_tiles.size(); ++index) {
                  const matrix::Tile<const T, Device::GPU>& tile_v = hh_tiles[index].get();
                  Helpers::loop_gemv(handle, tile_v, taus, tile_t);
                }

                return std::move(tile_t);
              });

      if (id_worker == 0)
        tile_t = std::move(workspace);
      else
        workspaces[id_worker - 1] = std::move(workspace);
    }

    if (nworkers > 1)
      tile_t =
          di::whenAllLift(std::move(tile_t), ex::when_all_vector(std::move(workspaces))) |
          di::transform<dlaf::internal::TransformDispatchType::Plain>(
              di::Policy<Backend::GPU>(thread_priority::high),
              [nworkers](auto&& tile_t, auto&& workspaces, whip::stream_t stream) {
                for (std::size_t index = 0; index < nworkers - 1; ++index) {
                  matrix::Tile<T, Device::GPU>& ws = workspaces[index];
                  gpulapack::add(blas::Uplo::Upper, tile_t.size().rows() - 1, tile_t.size().cols() - 1,
                                 T(1), ws.ptr({0, 1}), ws.ld(), tile_t.ptr({0, 1}), tile_t.ld(), stream);
                }
                return std::move(tile_t);
              });

    return tile_t;
  }

  static void loop_trmv(cublasHandle_t handle, const matrix::Tile<T, Device::GPU>& tile_t) {
    const SizeType k = tile_t.size().cols();

    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    for (SizeType j = 0; j < k; ++j) {
      const TileElementIndex t_start{0, j};
      const TileElementSize t_size{j, 1};

      gpublas::internal::Trmv<T>::call(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                       to_int(t_size.rows()), util::blasToCublasCast(tile_t.ptr()),
                                       to_int(tile_t.ld()), util::blasToCublasCast(tile_t.ptr(t_start)),
                                       1);
    }
  }

  static auto step_copy_diag_and_trmv(matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                                      matrix::ReadWriteTileSender<T, Device::GPU> tile_t) noexcept {
    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    auto trmv_func = [](cublasHandle_t handle, const matrix::Tile<const T, Device::CPU>& taus,
                        matrix::Tile<T, Device::GPU>& tile_t) {
      whip::stream_t stream;
      DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));

      const SizeType k = tile_t.size().cols();
      gpulapack::lacpy(blas::Uplo::General, 1, k, taus.ptr(), 1, tile_t.ptr(), tile_t.ld() + 1, stream);

      loop_trmv(handle, tile_t);
    };

    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    return ex::when_all(std::move(taus), std::move(tile_t)) |
           di::transform<di::TransformDispatchType::Blas>(
               di::Policy<Backend::GPU>(pika::execution::thread_priority::high), std::move(trmv_func));
  }
};
#endif
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(
    matrix::Panel<Coord::Col, T, device>& hh_panel, matrix::ReadOnlyTileSender<T, Device::CPU> taus,
    matrix::ReadWriteTileSender<T, device> tile_t,
    std::vector<matrix::ReadWriteTileSender<T, device>> workspaces) {
  namespace ex = pika::execution::experimental;

  using Helpers = tfactor_l::Helpers<backend, device, T>;

  // Fast return in case of no reflectors
  if (hh_panel.getWidth() == 0)
    return;

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
  tile_t = Helpers::step_gemv(hh_panel, taus, std::move(tile_t), std::move(workspaces));

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  ex::start_detached(Helpers::step_copy_diag_and_trmv(taus, std::move(tile_t)));
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(
    matrix::Panel<Coord::Col, T, device>& hh_panel, matrix::ReadOnlyTileSender<T, Device::CPU> taus,
    matrix::ReadWriteTileSender<T, device> tile_t,
    std::vector<matrix::ReadWriteTileSender<T, device>> workspaces,
    comm::CommunicatorPipeline<comm::CommunicatorType::Col>& mpi_col_task_chain) {
  namespace ex = pika::execution::experimental;

  using Helpers = tfactor_l::Helpers<backend, device, T>;

  // Fast return in case of no reflectors
  if (hh_panel.getWidth() == 0)
    return;

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

  // Note:
  // reset is needed because not all ranks might have computations to do, but they will participate
  // to mpi reduction anyway.
  tile_t = Helpers::set0_and_return(std::move(tile_t));

  // 1st step: compute the column partial result `t`
  // First we compute the matrix vector multiplication for each column
  // -tau(j) . V(j:, 0:j)* . V(j:, j)
  tile_t = Helpers::step_gemv(hh_panel, taus, std::move(tile_t), std::move(workspaces));

  // at this point each rank has its partial result for each column
  // so, let's reduce the results (on all ranks, so that everyone can independently compute T factor)
  if (mpi_col_task_chain.size() > 1)
    tile_t = schedule_all_reduce_in_place(mpi_col_task_chain.exclusive(), MPI_SUM, std::move(tile_t));

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  ex::start_detached(Helpers::step_copy_diag_and_trmv(taus, std::move(tile_t)));
}

}
