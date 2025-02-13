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
#include <iterator>
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
#include <dlaf/factorization/qr/internal/get_tfactor_barrier_busy_wait.h>
#include <dlaf/factorization/qr/internal/get_tfactor_num_workers.h>
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

template <Backend B>
auto split_tfactor_work(const std::size_t nrtiles) {
  const std::size_t min_workers = 1;
  const std::size_t available_workers = get_tfactor_num_workers<B>();
  const std::size_t ideal_workers = util::ceilDiv(to_sizet(nrtiles), to_sizet(1));

  struct {
    std::size_t nworkers;
    std::size_t batch_size;
  } params;

  params.batch_size = util::ceilDiv(nrtiles, std::clamp(ideal_workers, min_workers, available_workers));
  DLAF_ASSERT_MODERATE(params.batch_size > 0, params.batch_size, nrtiles);
  params.nworkers = std::max<std::size_t>(1, util::ceilDiv(nrtiles, params.batch_size));

  return params;
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
      matrix::Panel<Coord::Col, T, Device::CPU>& workspaces) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    using pika::execution::thread_priority;

    std::vector<matrix::ReadOnlyTileSender<T, Device::CPU>> hh_tiles =
        selectRead(hh_panel, hh_panel.iteratorLocal());

    const SizeType k = hh_panel.getWidth();

    if (hh_tiles.size() == 0)
      return tile_t;

    const auto workers_params = split_tfactor_work<Backend::MC>(hh_tiles.size());
    const std::size_t nworkers = workers_params.nworkers;
    const std::size_t batch_size = workers_params.batch_size;

    const auto range_workspaces = common::iterate_range2d(LocalTileSize{to_SizeType(nworkers - 1), 1});
    DLAF_ASSERT(to_sizet(std::distance(range_workspaces.begin(), range_workspaces.end())) >=
                    nworkers - 1,
                std::distance(range_workspaces.begin(), range_workspaces.end()), nworkers - 1);

    const auto hp_scheduler = di::getBackendScheduler<Backend::MC>(thread_priority::high);
    return ex::when_all(ex::when_all_vector(std::move(hh_tiles)), std::move(taus), std::move(tile_t),
                        ex::when_all_vector(select(workspaces, range_workspaces))) |
           ex::continues_on(hp_scheduler) |
           ex::let_value([hp_scheduler, k, nworkers, batch_size](auto& hh_tiles, auto& taus,
                                                                 auto& tile_t, auto& workspaces) {
             return ex::just(std::make_unique<pika::barrier<>>(nworkers)) |
                    ex::continues_on(hp_scheduler) |
                    ex::bulk(
                        nworkers,
                        [=, &hh_tiles, &taus, &tile_t, &workspaces](const std::size_t worker_id,
                                                                    auto& barrier_ptr) mutable {
                          const auto barrier_busy_wait = getTFactorBarrierBusyWait();
                          DLAF_ASSERT_HEAVY(k == taus.get().size().rows(), k, taus.get().size().rows());

                          const std::size_t begin = worker_id * batch_size;
                          const std::size_t end =
                              std::min(worker_id * batch_size + batch_size, hh_tiles.size());

                          const matrix::Tile<T, Device::CPU> ws_worker =
                              (worker_id == 0 ? tile_t : workspaces[worker_id - 1])
                                  .subTileReference({{0, 0}, tile_t.size()});

                          tile::internal::set0<T>(ws_worker);
                          lapack::lacpy(blas::Uplo::General, 1, k, taus.get().ptr(), 1, ws_worker.ptr(),
                                        ws_worker.ld() + 1);

                          // make it work on worker_id section of tiles
                          for (std::size_t index = begin; index < end; ++index) {
                            const matrix::Tile<const T, Device::CPU>& tile_v = hh_tiles[index].get();
                            loop_gemv(tile_v, taus.get(), ws_worker);
                          }

                          barrier_ptr->arrive_and_wait(barrier_busy_wait);

                          // reduce ws_T in tile_t
                          if (worker_id == 0) {
                            for (std::size_t other_worker = 1; other_worker < nworkers; ++other_worker) {
                              matrix::Tile<T, Device::CPU> ws =
                                  workspaces[other_worker - 1].subTileReference({{0, 0}, tile_t.size()});
                              tile::internal::add(T(1), ws, tile_t);
                            }
                          }
                        }) |
                    // Note: ignore the barrier sent by the bulk and just return tile_t
                    ex::then([&tile_t](auto) mutable { return std::move(tile_t); });
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

    gpulapack::larft_gemv1_notau(handle, m, k, tile_v.ptr(), tile_v.ld(), tile_t.ptr(), tile_t.ld());
  }

  static matrix::ReadWriteTileSender<T, Device::GPU> step_gemv(
      matrix::Panel<Coord::Col, T, Device::GPU>& hh_panel,
      matrix::ReadOnlyTileSender<T, Device::CPU> taus,
      matrix::ReadWriteTileSender<T, Device::GPU> tile_t,
      matrix::Panel<Coord::Col, T, Device::GPU>& workspaces) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    using pika::execution::thread_priority;

    std::vector<matrix::ReadOnlyTileSender<T, Device::GPU>> hh_tiles =
        selectRead(hh_panel, hh_panel.iteratorLocal());

    const SizeType k = hh_panel.getWidth();

    if (hh_tiles.size() == 0)
      return tile_t;

    const auto workers_params = split_tfactor_work<Backend::GPU>(hh_tiles.size());
    const std::size_t nworkers = workers_params.nworkers;
    const std::size_t batch_size = workers_params.batch_size;

    const auto range_workspaces = common::iterate_range2d(LocalTileSize{to_SizeType(nworkers - 1), 1});
    DLAF_ASSERT(to_sizet(std::distance(range_workspaces.begin(), range_workspaces.end())) >=
                    nworkers - 1,
                std::distance(range_workspaces.begin(), range_workspaces.end()), nworkers - 1);

    for (std::size_t worker_id = 0; worker_id < nworkers; ++worker_id) {
      const std::size_t begin = worker_id * batch_size;
      const std::size_t end = std::min(hh_tiles.size(), (worker_id + 1) * batch_size);

      DLAF_ASSERT(end >= begin, begin, end);

      if (end == begin)
        continue;

      std::vector<matrix::ReadOnlyTileSender<T, Device::GPU>> input_tiles;
      input_tiles.reserve(end - begin);
      for (std::size_t sub = begin; sub < end; ++sub)
        input_tiles.emplace_back(std::move(hh_tiles[sub]));

      matrix::ReadWriteTileSender<T, Device::GPU> workspace =
          worker_id == 0 ? std::move(tile_t)
                         : workspaces.readwrite(LocalTileIndex{to_SizeType(worker_id - 1), 0});

      workspace =
          di::whenAllLift(ex::when_all_vector(std::move(input_tiles)), taus, std::move(workspace)) |
          di::transform<dlaf::internal::TransformDispatchType::Blas>(
              di::Policy<Backend::GPU>(thread_priority::high),
              [k](cublasHandle_t handle, auto&& hh_tiles, auto&& taus,
                  matrix::Tile<T, Device::GPU>& tile_t_full) {
                DLAF_ASSERT_MODERATE(k == taus.size().rows(), k, taus.size().rows());

                matrix::Tile<T, Device::GPU> tile_t = tile_t_full.subTileReference({{0, 0}, {k, k}});

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

                return std::move(tile_t_full);
              });

      if (worker_id == 0)
        tile_t = std::move(workspace);
      else
        ex::start_detached(std::move(workspace));
    }

    if (nworkers > 1 and k > 1) {
      tile_t =
          di::whenAllLift(std::move(tile_t),
                          ex::when_all_vector(selectRead(workspaces, range_workspaces))) |
          di::transform<dlaf::internal::TransformDispatchType::Plain>(
              di::Policy<Backend::GPU>(thread_priority::high),
              [nworkers](auto&& tile_t, auto&& workspaces, whip::stream_t stream) {
                for (std::size_t index = 0; index < nworkers - 1; ++index) {
                  matrix::Tile<const T, Device::GPU> ws =
                      workspaces[index].get().subTileReference({{0, 0}, tile_t.size()});
                  DLAF_ASSERT(equal_size(ws, tile_t), ws, tile_t);
                  gpulapack::add(blas::Uplo::Upper, tile_t.size().rows() - 1, tile_t.size().cols() - 1,
                                 T(1), ws.ptr({0, 1}), ws.ld(), tile_t.ptr({0, 1}), tile_t.ld(), stream);
                }
                return std::move(tile_t);
              });
    }

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
      gpulapack::larft_gemv1_fixtau(k, tile_t.ptr(), tile_t.ld() + 1, tile_t.ptr(), tile_t.ld(), stream);

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
void QR_Tfactor<backend, device, T>::call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                                          matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                                          matrix::ReadWriteTileSender<T, device> tile_t,
                                          matrix::Panel<Coord::Col, T, device>& workspaces) {
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
  // The result is achieved in two main steps:
  // 1) "GEMV" t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) "TRMV" T(0:j, j) = T(0:j, 0:j) . t

  tile_t = Helpers::step_gemv(hh_panel, taus, std::move(tile_t), workspaces);
  ex::start_detached(Helpers::step_copy_diag_and_trmv(taus, std::move(tile_t)));
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(
    matrix::Panel<Coord::Col, T, device>& hh_panel, matrix::ReadOnlyTileSender<T, Device::CPU> taus,
    matrix::ReadWriteTileSender<T, device> tile_t, matrix::Panel<Coord::Col, T, device>& workspaces,
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
  // The result is achieved in two main steps:
  // 1) "GEMV" t = -tau(j) . V(j:, 0:j)* . V(j:, j)
  // 2) "TRMV" T(0:j, j) = T(0:j, 0:j) . t

  // Note:
  // reset is needed because not all ranks might have computations to do, but they will participate
  // to mpi reduction anyway.
  tile_t = Helpers::set0_and_return(std::move(tile_t));
  tile_t = Helpers::step_gemv(hh_panel, taus, std::move(tile_t), workspaces);

  // Note: at this point each rank has its partial result for each column
  // so, let's reduce the results (on all ranks, so that everyone can independently compute T factor)
  if (mpi_col_task_chain.size() > 1)
    tile_t = schedule_all_reduce_in_place(mpi_col_task_chain.exclusive(), MPI_SUM, std::move(tile_t));

  ex::start_detached(Helpers::step_copy_diag_and_trmv(taus, std::move(tile_t)));
}

}
