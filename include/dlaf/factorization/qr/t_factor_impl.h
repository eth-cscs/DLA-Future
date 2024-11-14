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
template <Backend backend, Device device, class T>
struct Helpers {};

template <class T>
struct Helpers<Backend::MC, Device::CPU, T> {
  static auto prepareT(matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                       matrix::ReadWriteTileSender<T, Device::CPU> tile_t) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;
    return ex::when_all(std::move(taus), std::move(tile_t)) |
           di::transform(di::Policy<Backend::MC>(pika::execution::thread_priority::high),
                         [](const matrix::Tile<const T, Device::CPU>& taus,
                            matrix::Tile<T, Device::CPU>&& tile_t) {
                           tile::internal::set0<T>(tile_t);
                           return std::move(tile_t);
                         });
  }

  static matrix::Tile<T, Device::CPU> gemvLoop(const matrix::Tile<const T, Device::CPU>& tile_v,
                                               const matrix::Tile<const T, Device::CPU>& taus,
                                               matrix::Tile<T, Device::CPU> tile_t) noexcept {
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
    return tile_t;
  }

  static auto stepGEMV(matrix::ReadOnlyTileSender<T, Device::CPU> tile_vi,
                       matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                       matrix::ReadWriteTileSender<T, Device::CPU> tile_t) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    return ex::when_all(tile_vi, std::move(taus), std::move(tile_t)) |
           di::transform(di::Policy<Backend::MC>(pika::execution::thread_priority::high), gemvLoop);
  }

  static matrix::ReadWriteTileSender<T, Device::CPU> stepGEMVAll(
      std::vector<matrix::ReadOnlyTileSender<T, Device::CPU>> hh_tiles,
      matrix::ReadOnlyTileSender<T, Device::CPU> taus,
      matrix::ReadWriteTileSender<T, Device::CPU> tile_t,
      std::vector<matrix::ReadWriteTileSender<T, Device::CPU>> workspaces) {
    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    using pika::execution::thread_priority;

    const auto hp_scheduler = di::getBackendScheduler<Backend::MC>(thread_priority::high);
    return ex::when_all(ex::when_all_vector(std::move(hh_tiles)), std::move(taus), std::move(tile_t),
                        ex::when_all_vector(std::move(workspaces))) |
           di::continues_on(hp_scheduler) |
           ex::let_value([hp_scheduler](auto&& hh_tiles, auto&& taus, auto&& tile_t, auto&& workspaces) {
             const std::size_t nworkers = eigensolver::internal::getReductionToBandPanelNWorkers();
             const std::chrono::duration<double> barrier_busy_wait = std::chrono::microseconds(1000);

             DLAF_ASSERT(nworkers == workspaces.size(), nworkers, workspaces.size());

             return ex::just(std::make_unique<pika::barrier<>>(nworkers)) |
                    di::continues_on(hp_scheduler) |
                    ex::bulk(nworkers,
                             [=, &hh_tiles, &taus, &tile_t, &workspaces](const std::size_t worker_id,
                                                                         auto& barrier_ptr) mutable {
                               const SizeType k = tile_t.size().cols();

                               const std::size_t batch_size = util::ceilDiv(hh_tiles.size(), nworkers);
                               const std::size_t begin = worker_id * batch_size;
                               const std::size_t end =
                                   std::min(worker_id * batch_size + batch_size, hh_tiles.size());

                               auto&& ws_worker = workspaces[worker_id];
                               tile::internal::set0<T>(ws_worker);
                               lapack::lacpy(blas::Uplo::General, 1, k, taus.get().ptr(), 1,
                                             ws_worker.ptr(), ws_worker.ld() + 1);

                               // make it work on worker_id section of tiles
                               for (std::size_t index = begin; index < end; ++index) {
                                 auto&& tile_snd = hh_tiles[index];
                                 ws_worker = gemvLoop(tile_snd.get(), taus.get(), std::move(ws_worker));
                               }
                               barrier_ptr->arrive_and_wait(barrier_busy_wait);

                               // reduce ws_T in tile_t
                               if (worker_id == 0) {
                                 tile::internal::set0<T>(tile_t);
                                 for (std::size_t other_worker = 0; other_worker < nworkers;
                                      ++other_worker) {
                                   tile::internal::add(T(1), workspaces[other_worker], tile_t);
                                 }
                               }
                             }) |
                    // Note: drop the barrier sent by the bulk and return tile_t
                    ex::then([&tile_t](auto&&) mutable { return std::move(tile_t); });
           });
  }

  static void trmvLoop(const matrix::Tile<T, Device::CPU>& tile_t) {
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

  static auto stepTRMV(matrix::ReadWriteTileSender<T, Device::CPU> tile_t) noexcept {
    namespace di = dlaf::internal;

    return std::move(tile_t) |
           di::transform(di::Policy<Backend::MC>(pika::execution::thread_priority::high), trmvLoop);
  }

  static auto stepCopyDiagAndTRMV(matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                                  matrix::ReadWriteTileSender<T, Device::CPU> tile_t) noexcept {
    auto tausdiag_trmvloop = [](const matrix::Tile<const T, Device::CPU>& taus,
                                matrix::Tile<T, Device::CPU> tile_t) {
      common::internal::SingleThreadedBlasScope single;

      const SizeType k = tile_t.size().cols();
      lapack::lacpy(blas::Uplo::General, 1, k, taus.ptr(), 1, tile_t.ptr(), tile_t.ld() + 1);

      trmvLoop(tile_t);
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
  static auto prepareT(matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                       matrix::ReadWriteTileSender<T, Device::GPU> tile_t) {
    namespace di = dlaf::internal;
    namespace ex = pika::execution::experimental;

    return ex::when_all(std::move(taus), std::move(tile_t)) |
           di::transform(di::Policy<Backend::GPU>(pika::execution::thread_priority::high),
                         [](const matrix::Tile<const T, Device::CPU>& taus,
                            matrix::Tile<T, Device::GPU>& tile_t, whip::stream_t stream) {
                           tile::internal::set0<T>(tile_t, stream);

                           const SizeType k = tile_t.size().cols();
                           whip::memcpy_2d_async(tile_t.ptr(), to_sizet(tile_t.ld() + 1) * sizeof(T),
                                                 taus.ptr(), sizeof(T), sizeof(T), to_sizet(k),
                                                 whip::memcpy_host_to_device, stream);

                           return std::move(tile_t);
                         });
  }

  static auto stepGEMV(matrix::ReadOnlyTileSender<T, Device::GPU> tile_vi,
                       matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                       matrix::ReadWriteTileSender<T, Device::GPU> tile_t) noexcept {
    auto gemv_func = [](cublasHandle_t handle, const matrix::Tile<const T, Device::GPU>& tile_v,
                        const matrix::Tile<const T, Device::CPU>& taus,
                        matrix::Tile<T, Device::GPU>& tile_t) noexcept {
      const SizeType m = tile_v.size().rows();
      const SizeType k = tile_t.size().cols();
      DLAF_ASSERT(tile_v.size().cols() == k, tile_v.size().cols(), k);
      DLAF_ASSERT(taus.size().rows() == k, taus.size().rows(), k);
      DLAF_ASSERT(taus.size().cols() == 1, taus.size().cols());

      gpulapack::larft_gemv0(handle, m, k, tile_v.ptr(), tile_v.ld(), taus.ptr(), tile_t.ptr(),
                             tile_t.ld());

      return std::move(tile_t);
    };

    namespace ex = pika::execution::experimental;
    namespace di = dlaf::internal;

    return ex::when_all(std::move(tile_vi), std::move(taus), std::move(tile_t)) |
           di::transform<di::TransformDispatchType::Blas>(
               di::Policy<Backend::GPU>(pika::execution::thread_priority::high), std::move(gemv_func));
  }

  static void trmvLoop(cublasHandle_t handle, const matrix::Tile<T, Device::GPU>& tile_t) {
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

  static auto stepTRMV(matrix::ReadWriteTileSender<T, Device::GPU> tile_t) noexcept {
    namespace di = dlaf::internal;

    return std::move(tile_t) |
           di::transform<di::TransformDispatchType::Blas>(
               di::Policy<Backend::GPU>(pika::execution::thread_priority::high), trmvLoop);
  }

  static auto stepCopyDiagAndTRMV(matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                                  matrix::ReadWriteTileSender<T, Device::GPU> tile_t) noexcept {
    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    auto trmv_func = [](cublasHandle_t handle, const matrix::Tile<const T, Device::CPU>& taus,
                        matrix::Tile<T, Device::GPU>& tile_t) {
      whip::stream_t stream;
      DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));

      const SizeType k = tile_t.size().cols();
      whip::memcpy_2d_async(tile_t.ptr(), to_sizet(tile_t.ld() + 1) * sizeof(T), taus.ptr(), sizeof(T),
                            sizeof(T), to_sizet(k), whip::memcpy_host_to_device, stream);

      trmvLoop(handle, tile_t);
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

  tile_t = Helpers::prepareT(taus, std::move(tile_t));

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

  if constexpr (backend == Backend::MC) {
    tile_t = Helpers::stepGEMVAll(matrix::selectRead(hh_panel, hh_panel.iteratorLocal()), taus,
                                  std::move(tile_t), std::move(workspaces));
  }

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  ex::start_detached(Helpers::stepTRMV(std::move(tile_t)));
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(
    matrix::Panel<Coord::Col, T, device>& hh_panel, matrix::ReadOnlyTileSender<T, Device::CPU> taus,
    matrix::ReadWriteTileSender<T, device> tile_t,
    comm::CommunicatorPipeline<comm::CommunicatorType::Col>& mpi_col_task_chain) {
  namespace ex = pika::execution::experimental;

  using Helpers = tfactor_l::Helpers<backend, device, T>;

  // Fast return in case of no reflectors
  if (hh_panel.getWidth() == 0)
    return;

  tile_t = Helpers::prepareT(taus, std::move(tile_t));

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
  for (const auto& i_lc : hh_panel.iteratorLocal()) {
    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    tile_t = Helpers::stepGEMV(hh_panel.read(i_lc), taus, std::move(tile_t));
  }

  // at this point each rank has its partial result for each column
  // so, let's reduce the results (on all ranks, so that everyone can independently compute T factor)
  if (mpi_col_task_chain.size() > 1)
    tile_t = schedule_all_reduce_in_place(mpi_col_task_chain.exclusive(), MPI_SUM, std::move(tile_t));

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  ex::start_detached(Helpers::stepCopyDiagAndTRMV(taus, std::move(tile_t)));
}

}
