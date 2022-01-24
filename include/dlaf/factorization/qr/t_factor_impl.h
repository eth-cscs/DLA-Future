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
#include "dlaf/factorization/qr/t_factor_kernels.h"

#include "dlaf/common/assert.h"
#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/kernels/all_reduce.h"
#include "dlaf/executors.h"
#include "dlaf/lapack/tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

#ifdef DLAF_WITH_CUDA
#include "dlaf/cublas/template_cublas.h"
#endif

namespace dlaf::factorization::internal {

namespace tfactor_l {
template <Backend backend, Device device, class T>
struct Helpers {};

template <class T>
struct Helpers<Backend::MC, Device::CPU, T> {
  static pika::future<matrix::Tile<T, Device::CPU>> set0(pika::future<matrix::Tile<T, Device::CPU>>& t) {
    return t.then(getHpExecutor<Backend::MC>(), pika::unwrapping([](auto&& tile) {
                    tile::internal::set0<T>(tile);
                    return std::move(tile);
                  }));
  }

  static pika::future<matrix::Tile<T, Device::CPU>> gemvColumnT(
      bool is_v0, pika::shared_future<matrix::Tile<const T, Device::CPU>> tile_vi,
      pika::shared_future<common::internal::vector<T>>& taus,
      pika::future<matrix::Tile<T, Device::CPU>>& tile_t) {
    auto gemv_func =
        pika::unwrapping([is_v0](const auto& tile_v, const auto& taus, auto&& tile_t) noexcept {
          const SizeType k = tile_t.size().cols();
          DLAF_ASSERT(taus.size() == k, taus.size(), k);

          for (SizeType j = 0; j < k; ++j) {
            const T tau = taus[j];
            // this is the x0 element of the reflector j
            const TileElementIndex x0{j, j};

            const TileElementIndex t_start{0, x0.col()};

            const SizeType first_element_in_tile = is_v0 ? x0.row() : 0;

            // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
            // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
            TileElementSize va_size{tile_v.size().rows() - first_element_in_tile, x0.col()};
            TileElementIndex va_start{first_element_in_tile, 0};
            TileElementIndex vb_start{first_element_in_tile, x0.col()};

            if (is_v0) {
              tile_t(x0) = tau;

              // use implicit 1 for the 2nd operand
              for (SizeType r = 0; !va_size.isEmpty() && r < va_size.cols(); ++r) {
                const TileElementIndex i_v{va_start.row(), r + va_start.col()};
                const TileElementIndex i_t{r + t_start.row(), t_start.col()};

                tile_t(i_t) = -tau * dlaf::conj(tile_v(i_v));
              }

              // skip already managed computations with implicit 1
              va_start = va_start + TileElementSize{1, 0};
              vb_start = vb_start + TileElementSize{1, 0};
              va_size = {va_size.rows() - 1, va_size.cols()};
            }

            if (!va_size.isEmpty()) {
              blas::gemv(blas::Layout::ColMajor, blas::Op::ConjTrans, va_size.rows(), va_size.cols(),
                         -tau, tile_v.ptr(va_start), tile_v.ld(), tile_v.ptr(vb_start), 1, 1,
                         tile_t.ptr(t_start), 1);
            }
          }
          return std::move(tile_t);
        });
    return pika::dataflow(getHpExecutor<Backend::MC>(), gemv_func, tile_vi, taus, tile_t);
  }

  static pika::future<matrix::Tile<T, Device::CPU>> trmvUpdateColumn(
      pika::future<matrix::Tile<T, Device::CPU>>& tile_t) noexcept {
    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    auto trmv_func = pika::unwrapping([](auto&& tile_t) {
      for (SizeType j = 0; j < tile_t.size().cols(); ++j) {
        const TileElementIndex t_start{0, j};
        const TileElementSize t_size{j, 1};

        blas::trmv(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
                   t_size.rows(), tile_t.ptr(), tile_t.ld(), tile_t.ptr(t_start), 1);
      }
      return std::move(tile_t);
    });
    return pika::dataflow(getHpExecutor<Backend::MC>(), trmv_func, tile_t);
  }
};

#ifdef DLAF_WITH_CUDA
template <class T>
struct Helpers<Backend::GPU, Device::GPU, T> {
  static pika::future<matrix::Tile<T, Device::GPU>> set0(pika::future<matrix::Tile<T, Device::GPU>>& t) {
    return pika::dataflow(dlaf::cuda::Executor{dlaf::internal::getHpCudaStreamPool()},
                         pika::unwrapping([](auto&& tile, cudaStream_t stream) {
                           tile::internal::set0<T>(tile, stream);
                           return std::move(tile);
                         }),
                         std::move(t));
  }

  static pika::future<matrix::Tile<T, Device::GPU>> gemvColumnT(
      bool is_v0, pika::shared_future<matrix::Tile<const T, Device::GPU>> tile_vi,
      pika::shared_future<common::internal::vector<T>>& taus,
      pika::future<matrix::Tile<T, Device::GPU>>& tile_t) noexcept {
    auto gemv_func = pika::unwrapping(
        [is_v0](cublasHandle_t handle, const auto& tile_v, const auto& taus, auto&& tile_t) {
          const SizeType k = tile_t.size().cols();
          DLAF_ASSERT(taus.size() == k, taus.size(), k);

          if (is_v0) {
            cudaStream_t stream;
            DLAF_CUBLAS_CALL(cublasGetStream(handle, &stream));

            memory::MemoryView<T, Device::GPU> taus_d(taus.size());
            DLAF_CUDA_CALL(cudaMemcpyAsync(taus_d(), taus.data(), to_sizet(taus.size()) * sizeof(T),
                                           cudaMemcpyDefault, stream));

            // manage computations with implicit 1 for the whole j loop
            tfactorImplicit1(k, taus_d(), tile_v.ptr(), tile_v.ld(), tile_t.ptr(), tile_t.ld(), stream);
          }

          for (SizeType j = 0; j < k; ++j) {
            const auto mtau = util::blasToCublasCast(-taus[j]);
            const auto one = util::blasToCublasCast(T{1});
            // this is the x0 element of the reflector j
            const TileElementIndex x0{j, j};

            const TileElementIndex t_start{0, x0.col()};

            const SizeType first_element_in_tile = is_v0 ? x0.row() : 0;

            // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
            // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
            TileElementSize va_size{tile_v.size().rows() - first_element_in_tile, x0.col()};
            TileElementIndex va_start{first_element_in_tile, 0};
            TileElementIndex vb_start{first_element_in_tile, x0.col()};

            if (is_v0) {
              // skip already managed computations with implicit 1
              va_start = va_start + TileElementSize{1, 0};
              vb_start = vb_start + TileElementSize{1, 0};
              va_size = va_size - TileElementSize{1, 0};
            }

            if (!va_size.isEmpty()) {
              cublas::Gemv<T>::call(handle, CUBLAS_OP_C, to_int(va_size.rows()), to_int(va_size.cols()),
                                    &mtau, util::blasToCublasCast(tile_v.ptr(va_start)),
                                    to_int(tile_v.ld()), util::blasToCublasCast(tile_v.ptr(vb_start)), 1,
                                    &one, util::blasToCublasCast(tile_t.ptr(t_start)), 1);
            }
          }
          return std::move(tile_t);
        });
    return pika::dataflow(getHpExecutor<Backend::GPU>(), gemv_func, tile_vi, taus, tile_t);
  }

  static pika::future<matrix::Tile<T, Device::GPU>> trmvUpdateColumn(
      pika::future<matrix::Tile<T, Device::GPU>>& tile_t) noexcept {
    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    auto trmv_func = pika::unwrapping([](cublasHandle_t handle, auto&& tile_t) {
      for (SizeType j = 0; j < tile_t.size().cols(); ++j) {
        const TileElementIndex t_start{0, j};
        const TileElementSize t_size{j, 1};

        cublas::Trmv<T>::call(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                              to_int(t_size.rows()), util::blasToCublasCast(tile_t.ptr()),
                              to_int(tile_t.ld()), util::blasToCublasCast(tile_t.ptr(t_start)), 1);
      }
      return std::move(tile_t);
    });
    return pika::dataflow(getHpExecutor<Backend::GPU>(), trmv_func, tile_t);
  }
};
#endif
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(const SizeType k, Matrix<const T, device>& v,
                                          const GlobalTileIndex v_start,
                                          pika::shared_future<common::internal::vector<T>> taus,
                                          pika::future<matrix::Tile<T, device>> t) {
  using Helpers = tfactor_l::Helpers<backend, device, T>;
  // Fast return in case of no reflectors
  if (k == 0)
    return;

  t = splitTile(t, {{0, 0}, {k, k}});
  const auto panel_width = v.tileSize(v_start).cols();

  DLAF_ASSERT(k <= panel_width, k, panel_width);

  const GlobalTileIndex v_end{v.nrTiles().rows(), std::min(v.nrTiles().cols(), v_start.col() + 1)};

  // TODO S/R
  t = Helpers::set0(t);

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
  for (const auto& v_i : iterate_range2d(v_start, v_end)) {
    const bool is_v0 = (v_i.row() == v_start.row());

    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    t = Helpers::gemvColumnT(is_v0, v.read(v_i), taus, t);
  }

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  t = Helpers::trmvUpdateColumn(t);
}

template <Backend backend, Device device, class T>
void QR_Tfactor<backend, device, T>::call(const SizeType k, Matrix<const T, device>& v,
                                          const GlobalTileIndex v_start,
                                          pika::shared_future<common::internal::vector<T>> taus,
                                          pika::future<matrix::Tile<T, device>> t,
                                          common::Pipeline<comm::Communicator>& mpi_col_task_chain) {
  using Helpers = tfactor_l::Helpers<backend, device, T>;
  if constexpr (backend != Backend::MC) {
    DLAF_STATIC_UNIMPLEMENTED(T);
  }
  t = splitTile(t, {{0, 0}, {k, k}});

  // Fast return in case of no reflectors
  if (k == 0)
    return;

  const auto panel_width = v.tileSize(v_start).cols();
  DLAF_ASSERT(0 <= k && k <= panel_width, k, panel_width);

  DLAF_ASSERT_MODERATE(v_start.isIn(v.nrTiles()), v_start, v.nrTiles());
  const auto& dist = v.distribution();
  const comm::Index2D rank = dist.rankIndex();
  const comm::Index2D rank_v0 = dist.rankGlobalTile(v_start);

  // Just the column of ranks with the reflectors participates
  if (rank.col() != rank_v0.col())
    return;

  const LocalTileIndex v_start_loc{
      dist.template nextLocalTileFromGlobalTile<Coord::Row>(v_start.row()),
      dist.template nextLocalTileFromGlobalTile<Coord::Col>(v_start.col()),
  };
  const LocalTileIndex v_end_loc{dist.localNrTiles().rows(), v_start_loc.col() + 1};

  t = t.then(getHpExecutor<Backend::MC>(), pika::unwrapping([](auto&& tile) {
               tile::internal::set0<T>(tile);
               return std::move(tile);
             }));

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
  for (const auto& v_i_loc : iterate_range2d(v_start_loc, v_end_loc)) {
    const SizeType v_i = dist.template globalTileFromLocalTile<Coord::Row>(v_i_loc.row());
    const bool is_v0 = (v_i == v_start.row());

    // TODO
    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    t = Helpers::gemvColumnT(is_v0, v.read(v_i_loc), taus, t);
  }

  // at this point each rank has its partial result for each column
  // so, let's reduce the results (on all ranks, so that everyone can independently compute T factor)
  if (true)  // TODO if the column communicator has more than 1 tile...but I just have the pipeline
    t = scheduleAllReduceInPlace(getMPIExecutor<Backend::MC>(), mpi_col_task_chain(), MPI_SUM,
                                 std::move(t));

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  t = Helpers::trmvUpdateColumn(t);
}

}
