//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <hpx/future.hpp>
#include <hpx/include/util.hpp>

#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/types.h"

namespace dlaf {
namespace internal {
namespace mc {

template <class Type>
void computeTFactor(Matrix<Type, Device::CPU>& t, Matrix<const Type, Device::CPU>& a,
                    const LocalTileIndex ai_start_loc,
                    const GlobalTileIndex ai_start, const SizeType last_reflector,
                    common::internal::vector<hpx::shared_future<Type>> taus,
                    common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using hpx::util::unwrapping;
  using common::make_data;
  using namespace comm::sync;

  // TODO assumption: intra-tile reflector
  // check that last_reflector < block_size

  const auto& dist = a.distribution();
  const comm::Index2D rank = dist.rankIndex();
  const comm::Index2D rank_v0 = dist.rankGlobalTile(ai_start);

  const LocalTileIndex ai_bound_index{dist.localNrTiles().rows(), ai_start_loc.col() + 1};

  // 2. CALCULATE T-FACTOR
  // T(0:j, j) = T(0:j, 0:j) . -tau(j) . V(j:, 0:j)* . V(j:, j)
  for (SizeType j_reflector = 0; j_reflector <= last_reflector; ++j_reflector) {
    const TileElementIndex index_el_x0{j_reflector, j_reflector};

    // 2A First step GEMV
    const TileElementIndex t_start{0, index_el_x0.col()};
    const TileElementSize t_size{index_el_x0.row(), 1};

    for (const auto& index_tile_v : iterate_range2d(ai_start_loc, ai_bound_index)) {
      const SizeType index_tile_v_global =
          dist.template globalTileFromLocalTile<Coord::Row>(index_tile_v.row());

      const bool has_first_component = (index_tile_v_global == ai_start.row());

      // GEMV t = V(j:mV; 0:j)* . V(j:mV;j)
      auto gemv_func = unwrapping([=](auto&& tile_t, const Type tau, const auto& tile_v) {
        const SizeType first_element_in_tile = has_first_component ? index_el_x0.row() : 0;

        // T(0:j, j) = -tau . V(j:, 0:j)* . V(j:, j)
        // [j x 1] = [(n-j) x j]* . [(n-j) x 1]
        TileElementSize va_size{tile_v.size().rows() - first_element_in_tile, index_el_x0.col()};
        TileElementIndex va_start{first_element_in_tile, 0};
        TileElementIndex vb_start{first_element_in_tile, index_el_x0.col()};

        // if it is the "head" tile...
        if (has_first_component) {
          // set the tau on the diagonal
          tile_t(index_el_x0) = tau;

          // use implicit 1 for the 2nd operand
          for (SizeType r = 0; !va_size.isEmpty() && r < va_size.cols(); ++r) {
            const TileElementIndex i_v{va_start.row(), r + va_start.col()};
            const TileElementIndex i_t{r + t_start.row(), t_start.col()};

            DLAF_ASSERT_HEAVY(i_t.isIn(tile_t.size()), i_t, t_size);
            DLAF_ASSERT_HEAVY(i_v.isIn(tile_v.size()), i_v, tile_v.size());

            tile_t(i_t) += -tau * dlaf::conj(tile_v(i_v));
          }

          // and update the geometries/indices to skip the element managed separately
          va_start = va_start + TileElementSize{1, 0};
          vb_start = vb_start + TileElementSize{1, 0};

          va_size = {va_size.rows() - 1, va_size.cols()};
        }

        if (va_size.isValid() && !va_size.isEmpty()) {
          // t = -tau . V* . V
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
      });

      hpx::dataflow(gemv_func, t(LocalTileIndex{0, 0}), taus[j_reflector], a.read(index_tile_v));
    }

    // REDUCE after GEMV
    if (!t_size.isEmpty()) {
      auto reduce_t_func = unwrapping([=](auto&& tile_t, auto&& comm_wrapper) {
        auto&& input_t = make_data(tile_t.ptr(t_start), t_size.rows());
        std::vector<Type> out_data(to_sizet(t_size.rows()));
        auto&& output_t = make_data(out_data.data(), t_size.rows());
        // TODO reduce just the current, otherwise reduce all together
        reduce(rank_v0.row(), comm_wrapper().colCommunicator(), MPI_SUM, input_t, output_t);
        common::copy(output_t, input_t);
      });

      // TODO just reducer needs RW
      hpx::dataflow(reduce_t_func, t(LocalTileIndex{0, 0}), serial_comm());
    }

    // 2B Second Step TRMV
    if (rank_v0 == rank) {
      // TRMV t = T . t
      auto trmv_func = unwrapping([=](auto&& tile_t) {
        // clang-format off
        blas::trmv(blas::Layout::ColMajor,
            blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            t_size.rows(),
            tile_t.ptr(), tile_t.ld(),
            tile_t.ptr(t_start), 1);
        // clang-format on
      });

      hpx::dataflow(trmv_func, t(LocalTileIndex{0, 0}));
    }
  }
}

}
}
}
