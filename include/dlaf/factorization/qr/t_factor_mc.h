//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <hpx/future.hpp>
#include <hpx/include/util.hpp>

#include <blas.hh>

#include "dlaf/factorization/qr/api.h"

#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf {
namespace factorization {
namespace internal {

template <class T>
struct QR_Tfactor<Backend::MC, Device::CPU, T> {
  /// Forms the triangular factor T of a block of reflectors H, which is defined as a product of k
  /// elementary reflectors.
  ///
  /// It is similar to what xLARFT in LAPACK does.
  /// Given @p k elementary reflectors stored in the column of @p v starting at tile @p v_start,
  /// together with related tau values in @p taus, in @p t will be formed the triangular factor for the H
  /// block of reflector, such that
  ///
  /// H = I - V . T . V*
  ///
  /// where H = H1 . H2 . ... . Hk
  ///
  /// in which Hi represents a single elementary reflector transformation
  ///
  /// @param k the number of elementary reflectors to use (from the beginning of the tile)
  /// @param v where the elementary reflectors are stored
  /// @param v_start tile in @p v where the column of reflectors starts
  /// @param taus array of taus, associated with the related elementary reflector
  /// @param t a matrix where the triangular factor will be stored (in tile {0, 0}) for all ranks in the
  /// column of the reflectors
  /// @param serial_comm where internal communications are issued
  static void call(const SizeType k, Matrix<const T, Device::CPU>& v, const GlobalTileIndex v_start,
                   common::internal::vector<hpx::shared_future<T>> taus, Matrix<T, Device::CPU>& t,
                   common::Pipeline<comm::CommunicatorGrid>& serial_comm);
};

template <class T>
void QR_Tfactor<Backend::MC, Device::CPU, T>::call(
    const SizeType k, Matrix<const T, Device::CPU>& v, const GlobalTileIndex v_start,
    common::internal::vector<hpx::shared_future<T>> taus, Matrix<T, Device::CPU>& t,
    common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using hpx::util::unwrapping;
  using common::make_data;
  using namespace comm::sync;

  const auto& dist = v.distribution();
  const comm::Index2D rank = dist.rankIndex();
  const comm::Index2D rank_v0 = dist.rankGlobalTile(v_start);

  const auto panel_width = v.tileSize(v_start).cols();

  DLAF_ASSERT(k <= panel_width, k, panel_width);
  DLAF_ASSERT(taus.size() == k, taus.size(), k);

  const GlobalTileIndex t_idx(0, 0);
  const auto t_size = t.tileSize(t_idx);
  DLAF_ASSERT(k <= t_size.rows(), k, t_size);
  DLAF_ASSERT(k <= t_size.cols(), k, t_size);

  if (rank.col() != rank_v0.col())
    return;

  const LocalTileIndex v_start_loc{
      dist.template nextLocalTileFromGlobalTile<Coord::Row>(v_start.row()),
      dist.template nextLocalTileFromGlobalTile<Coord::Col>(v_start.col()),
  };
  const LocalTileIndex v_end_loc{dist.localNrTiles().rows(), v_start_loc.col() + 1};

  t(t_idx).then(unwrapping([](auto&& tile) {
    lapack::laset(lapack::MatrixType::General, tile.size().rows(), tile.size().cols(), 0, 0, tile.ptr(),
                  tile.ld());
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

    auto gemv_func = unwrapping([=](const auto& tile_v, const auto& taus, auto&& tile_t) {
      for (SizeType j = 0; j < k; ++j) {
        const T tau = taus[j];
        // this is the x0 element of the reflector j
        const TileElementIndex x0{j, j};

        const TileElementIndex t_start{0, x0.col()};
        const TileElementSize t_size{x0.row(), 1};

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

            DLAF_ASSERT_HEAVY(i_t.isIn(tile_t.size()), i_t, t_size);
            DLAF_ASSERT_HEAVY(i_v.isIn(tile_v.size()), i_v, tile_v.size());

            tile_t(i_t) = -tau * dlaf::conj(tile_v(i_v));
          }

          // skip alredy managed computations with implicit 1
          va_start = va_start + TileElementSize{1, 0};
          vb_start = vb_start + TileElementSize{1, 0};
          va_size = {va_size.rows() - 1, va_size.cols()};
        }

        if (!va_size.isEmpty()) {
          // clang-format off
          blas::gemv(blas::Layout::ColMajor,
              blas::Op::ConjTrans,
              va_size.rows(), va_size.cols(),
              -tau,
              tile_v.ptr(va_start), tile_v.ld(),
              tile_v.ptr(vb_start), 1,
              1, tile_t.ptr(t_start), 1);
          // clang-format on
        }
      }
    });

    // TODO
    // Note:
    // Since we are writing always on the same t, the gemv are serialized
    // A possible solution to this would be to have multiple places where to store partial
    // results, and then locally reduce them just before the reduce over ranks
    hpx::dataflow(gemv_func, v.read(v_i_loc), taus, t(t_idx));
  }

  // at this point each rank has its partial result for each column
  // so, let's reduce the results (on all ranks, so that everyone can independently compute T factor)
  if (true) {  // TODO if the column communicator has more than 1 tile...but I just have the pipeline
    auto reduce_t_func = unwrapping([=](auto&& tile_t, auto&& comm_wrapper) {
      auto&& input_t = make_data(tile_t);
      all_reduce(comm_wrapper.ref().colCommunicator(), MPI_SUM, input_t, input_t);
    });

    hpx::dataflow(reduce_t_func, t(t_idx), serial_comm());
  }

  // 2nd step: compute the T factor, by performing the last step on each column
  // each column depends on the previous part (all reflectors that comes before)
  // so it is performed sequentially
  for (SizeType j = 0; j < k; ++j) {
    const TileElementIndex t_start{0, j};
    const TileElementSize t_size{j, 1};

    // Update each column (in order) t = T . t
    // remember that T is upper triangular, so it is possible to use TRMV
    auto trmv_func = unwrapping([](auto&& tile_t, TileElementIndex t_start, TileElementSize t_size) {
      // clang-format off
      blas::trmv(blas::Layout::ColMajor,
          blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
          t_size.rows(),
          tile_t.ptr(), tile_t.ld(),
          tile_t.ptr(t_start), 1);
      // clang-format on
    });

    hpx::dataflow(trmv_func, t(t_idx), t_start, t_size);
  }
}
}
}
}
