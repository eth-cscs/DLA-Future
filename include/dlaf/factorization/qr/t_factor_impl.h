//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <pika/future.hpp>
#include <pika/unwrap.hpp>

#include <blas.hh>

#include "dlaf/factorization/qr/api.h"

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/kernels/all_reduce.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/views.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#ifdef DLAF_WITH_GPU
#include "dlaf/gpu/blas/api.h"
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
        dlaf::internal::Policy<Backend::MC>(pika::threads::thread_priority::high),
        [](matrix::Tile<T, Device::CPU>&& tile) {
          tile::internal::set0<T>(tile);
          return std::move(tile);
        },
        std::forward<TSender>(t));
  }

  template <class TSender>
  static auto gemvColumnT(SizeType first_row_tile,
                          pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_vi,
                          pika::shared_future<common::internal::vector<T>>& taus, TSender&& tile_t) {
    namespace ex = pika::execution::experimental;

    auto gemv_func = [first_row_tile](const auto& tile_v, const auto& taus, auto&& tile_t) noexcept {
      const SizeType k = tile_t.size().cols();
      DLAF_ASSERT(tile_v.size().cols() == k, tile_v.size().cols(), k);
      DLAF_ASSERT(taus.size() == k, taus.size(), k);

      for (SizeType j = 0; j < k; ++j) {
        const T tau = taus[j];

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

        if (i_diag >= 0) {
          tile_t({j, j}) = tau;
        }

        blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, va_size.rows(), va_size.cols(), -tau,
                   tile_v.ptr(va_start), tile_v.ld(), tile_v.ptr(vb_start), 1, 1, tile_t.ptr(t_start),
                   1);
      }
      return std::move(tile_t);
    };
    return dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(
                                         pika::threads::thread_priority::high),
                                     std::move(gemv_func),
                                     ex::when_all(ex::keep_future(tile_vi), ex::keep_future(taus),
                                                  std::forward<TSender>(tile_t)));
  }

  template <typename TSender>
  static auto trmvUpdateColumn(TSender&& tile_t) noexcept {
    namespace ex = pika::execution::experimental;

    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    auto trmv_func = [](matrix::Tile<T, Device::CPU>&& tile_t) {
      for (SizeType j = 0; j < tile_t.size().cols(); ++j) {
        const TileElementIndex t_start{0, j};
        const TileElementSize t_size{j, 1};

        blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
                   t_size.rows(), tile_t.ptr(), tile_t.ld(), tile_t.ptr(t_start), 1);
      }
      // TODO: Why return if the tile is unused?
      return std::move(tile_t);
    };
    return dlaf::internal::transform(dlaf::internal::Policy<Backend::MC>(
                                         pika::threads::thread_priority::high),
                                     std::move(trmv_func), std::forward<TSender>(tile_t));
  }
};

#ifdef DLAF_WITH_GPU
template <class T>
struct Helpers<Backend::GPU, Device::GPU, T> {
  template <class TSender>
  static auto set0(TSender&& t) {
    namespace ex = pika::execution::experimental;

    return dlaf::internal::transform(
        dlaf::internal::Policy<Backend::GPU>(pika::threads::thread_priority::high),
        [](matrix::Tile<T, Device::GPU>& tile, cudaStream_t stream) {
          tile::internal::set0<T>(tile, stream);
          return std::move(tile);
        },
        std::forward<TSender>(t));
  }

  template <class TSender>
  static auto gemvColumnT(SizeType first_row_tile,
                          pika::shared_future<matrix::Tile<const T, Device::GPU>> tile_vi,
                          pika::shared_future<common::internal::vector<T>>& taus,
                          TSender&& tile_t) noexcept {
    namespace ex = pika::execution::experimental;

    auto gemv_func = [first_row_tile](cublasHandle_t handle, const auto& tile_v, const auto& taus,
                                      auto& tile_t) noexcept {
      const SizeType k = tile_t.size().cols();
      DLAF_ASSERT(tile_v.size().cols() == k, tile_v.size().cols(), k);
      DLAF_ASSERT(taus.size() == k, taus.size(), k);

      if (first_row_tile == 0) {
        cudaStream_t stream;
        DLAF_CUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));

        DLAF_CUDA_CHECK_ERROR(cudaMemcpy2DAsync(tile_t.ptr(), to_sizet(tile_t.ld() + 1) * sizeof(T),
                                                taus.data(), sizeof(T), sizeof(T), to_sizet(k),
                                                cudaMemcpyDefault, stream));
      }

      for (SizeType j = 0; j < k; ++j) {
        const auto mtau = util::blasToCublasCast(-taus[j]);
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

        gpublas::Gemv<T>::call(handle, CUBLAS_OP_C, to_int(va_size.rows()), to_int(va_size.cols()),
                               &mtau, util::blasToCublasCast(tile_v.ptr(va_start)), to_int(tile_v.ld()),
                               util::blasToCublasCast(tile_v.ptr(vb_start)), 1, &one,
                               util::blasToCublasCast(tile_t.ptr(t_start)), 1);
      }
      return std::move(tile_t);
    };
    return dlaf::internal::transform<
        dlaf::internal::TransformDispatchType::Blas>(dlaf::internal::Policy<Backend::GPU>(
                                                         pika::threads::thread_priority::high),
                                                     std::move(gemv_func),
                                                     ex::when_all(ex::keep_future(tile_vi),
                                                                  ex::keep_future(taus),
                                                                  std::forward<TSender>(tile_t)));
  }

  template <class TSender>
  static auto trmvUpdateColumn(TSender&& tile_t) noexcept {
    namespace ex = pika::execution::experimental;

    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    auto trmv_func = [](cublasHandle_t handle, matrix::Tile<T, Device::GPU>& tile_t) {
      for (SizeType j = 0; j < tile_t.size().cols(); ++j) {
        const TileElementIndex t_start{0, j};
        const TileElementSize t_size{j, 1};

        gpublas::Trmv<T>::call(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                               to_int(t_size.rows()), util::blasToCublasCast(tile_t.ptr()),
                               to_int(tile_t.ld()), util::blasToCublasCast(tile_t.ptr(t_start)), 1);
      }
      return std::move(tile_t);
    };

    return dlaf::internal::transform<
        dlaf::internal::TransformDispatchType::Blas>(dlaf::internal::Policy<Backend::GPU>(
                                                         pika::threads::thread_priority::high),
                                                     std::move(trmv_func),
                                                     std::forward<TSender>(tile_t));
  }
};
#endif
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                                          pika::shared_future<common::internal::vector<T>> taus,
                                          pika::future<matrix::Tile<T, device>> t) {
  namespace ex = pika::execution::experimental;

  using Helpers = tfactor_l::Helpers<backend, device, T>;
  // Fast return in case of no reflectors
  if (hh_panel.getWidth() == 0)
    return;

  const auto v_start = hh_panel.offsetElement();

  ex::unique_any_sender<matrix::Tile<T, device>> t_local = Helpers::set0(std::move(t));

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
  for (const auto& v_i : hh_panel.iteratorLocal()) {
    const SizeType first_row_tile =
        std::max<SizeType>(0, v_i.row() * hh_panel.parentDistribution().blockSize().rows() - v_start);

    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    t_local = Helpers::gemvColumnT(first_row_tile, hh_panel.read(v_i), taus, std::move(t_local));
  }

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  ex::start_detached(Helpers::trmvUpdateColumn(std::move(t_local)));
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(matrix::Panel<Coord::Col, T, device>& hh_panel,
                                          pika::shared_future<common::internal::vector<T>> taus,
                                          pika::future<matrix::Tile<T, device>> t,
                                          common::Pipeline<comm::Communicator>& mpi_col_task_chain) {
  namespace ex = pika::execution::experimental;

  using Helpers = tfactor_l::Helpers<backend, device, T>;
  if constexpr (backend != Backend::MC) {
    DLAF_STATIC_UNIMPLEMENTED(T);
  }

  // Fast return in case of no reflectors
  if (hh_panel.getWidth() == 0)
    return;

  const auto v_start = hh_panel.offsetElement();
  auto dist = hh_panel.parentDistribution();

  ex::unique_any_sender<matrix::Tile<T, device>> t_local = Helpers::set0(std::move(t));

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
  if (true)  // TODO if the column communicator has more than 1 tile...but I just have the pipeline
    t_local = dlaf::comm::scheduleAllReduceInPlace(mpi_col_task_chain(), MPI_SUM, std::move(t_local));

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  ex::start_detached(Helpers::trmvUpdateColumn(std::move(t_local)));
}

}
