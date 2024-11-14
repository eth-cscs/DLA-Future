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

#include <pika/execution.hpp>

#include <dlaf/common/assert.h>
#include <dlaf/common/data.h>
#include <dlaf/common/index2d.h>
#include <dlaf/common/range2d.h>
#include <dlaf/common/single_threaded_blas.h>
#include <dlaf/communication/communicator_pipeline.h>
#include <dlaf/communication/kernels/all_reduce.h>
#include <dlaf/factorization/qr/api.h>
#include <dlaf/lapack/gpu/larft.h>
#include <dlaf/lapack/tile.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf/matrix/tile.h>
#include <dlaf/matrix/views.h>
#include <dlaf/types.h>
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

                           const SizeType k = tile_t.size().cols();
                           lapack::lacpy(blas::Uplo::General, 1, k, taus.ptr(), 1, tile_t.ptr(),
                                         tile_t.ld() + 1);

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
void QR_Tfactor<backend, device, T>::call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                                          matrix::ReadOnlyTileSender<T, Device::CPU> taus,
                                          matrix::ReadWriteTileSender<T, device> tile_t) {
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
