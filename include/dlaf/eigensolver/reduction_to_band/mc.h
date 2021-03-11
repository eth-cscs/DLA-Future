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

#include <cmath>
#include <string>

#include <hpx/future.hpp>
#include <hpx/include/util.hpp>
#include <hpx/tuple.hpp>

#include "dlaf/blas_tile.h"
#include "dlaf/common/data.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/pipeline.h"
#include "dlaf/common/range2d.h"
#include "dlaf/common/vector.h"
#include "dlaf/communication/communicator.h"
#include "dlaf/communication/communicator_grid.h"
#include "dlaf/communication/functions_sync.h"
#include "dlaf/lapack_tile.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/matrix/panel.h"
#include "dlaf/util_matrix.h"

#include "dlaf/eigensolver/reduction_to_band/api.h"
#include "dlaf/factorization/qr.h"

namespace dlaf {
namespace eigensolver {
namespace internal {

template <class T>
struct ReductionToBand<Backend::MC, Device::CPU, T> {
  static std::vector<hpx::shared_future<std::vector<T>>> call(comm::CommunicatorGrid grid,
                                                              Matrix<T, Device::CPU>& mat_a);
};

namespace {

using matrix::Matrix;
using matrix::Tile;

template <class Type>
using MatrixT = Matrix<Type, Device::CPU>;
template <class Type>
using ConstMatrixT = MatrixT<const Type>;
template <class Type>
using TileT = Tile<Type, Device::CPU>;
template <class Type>
using ConstTileT = TileT<const Type>;

template <Coord panel_type, class T>
using PanelT = matrix::Panel<panel_type, T, Device::CPU>;
template <Coord panel_type, class T>
using ConstPanelT = PanelT<panel_type, const T>;

template <class Type>
using FutureTile = hpx::future<TileT<Type>>;
template <class Type>
using FutureConstTile = hpx::shared_future<ConstTileT<Type>>;

template <class Type>
using MemViewT = memory::MemoryView<Type, Device::CPU>;

template <class Type>
void set_to_zero(MatrixT<Type>& matrix) {
  dlaf::matrix::util::set(matrix, [](...) { return 0; });
}

template <class T>
hpx::shared_future<T> compute_reflector(
    MatrixT<T>& a, const common::IterableRange2D<SizeType, dlaf::matrix::LocalTile_TAG> ai_panel,
    const GlobalTileIndex ai_start, const TileElementIndex index_el_x0,
    common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using hpx::util::unwrapping;
  using common::make_data;

  using namespace comm::sync;

  struct params_reflector_t {
    T x0;
    T y;
  };

  using x0_and_squares_t = std::pair<T, T>;

  const auto& dist = a.distribution();
  const comm::Index2D rank = dist.rankIndex();
  const comm::Index2D rank_v0 = dist.rankGlobalTile(ai_start);

  // 1A/1 COMPUTING NORM
  // Extract x0 and compute local cumulative sum of squares of the reflector column
  auto x0_and_squares = hpx::make_ready_future<x0_and_squares_t>(static_cast<T>(0), static_cast<T>(0));

  for (const LocalTileIndex& index_x_loc : ai_panel) {
    const SizeType index_x_row = dist.template globalTileFromLocalTile<Coord::Row>(index_x_loc.row());

    const bool has_first_component = (index_x_row == ai_start.row());

    if (has_first_component) {
      auto compute_x0_and_squares_func =
          unwrapping([index_el_x0](const auto& tile_x, x0_and_squares_t&& data) {
            data.first = tile_x(index_el_x0);

            const T* x_ptr = tile_x.ptr(index_el_x0);
            data.second = blas::dot(tile_x.size().rows() - index_el_x0.row(), x_ptr, 1, x_ptr, 1);

            return std::move(data);
          });

      x0_and_squares = hpx::dataflow(compute_x0_and_squares_func, a.read(index_x_loc), x0_and_squares);
    }
    else {
      auto cumsum_squares_func = unwrapping([index_el_x0](const auto& tile_x, x0_and_squares_t&& data) {
        const T* x_ptr = tile_x.ptr({0, index_el_x0.col()});
        data.second += blas::dot(tile_x.size().rows(), x_ptr, 1, x_ptr, 1);
        return std::move(data);
      });

      x0_and_squares = hpx::dataflow(cumsum_squares_func, a.read(index_x_loc), x0_and_squares);
    }
  }

  /*
   * reduce local cumulative sums
   * rank_v0 will have the x0 and the total cumulative sum of squares
   */
  auto reduce_norm_func = unwrapping([rank_v0](x0_and_squares_t&& local_data, auto&& comm_wrapper) {
    const T local_sum = local_data.second;
    T norm = local_data.second;
    reduce(rank_v0.row(), comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(&local_sum, 1),
           make_data(&norm, 1));
    local_data.second = norm;
    return std::move(local_data);
  });

  x0_and_squares = hpx::dataflow(reduce_norm_func, x0_and_squares, serial_comm());

  /*
   * rank_v0 will compute params that will be used for next computation of reflector components
   * FIXME in this case just one compute and the other will receive it
   * it may be better to compute on each one, in order to avoid a communication of few values
   * but it would benefit if all_reduce of the norm and x0 is faster than communicating params
   */
  // 1A/2 COMPUTE PARAMS
  hpx::shared_future<T> tau;
  hpx::shared_future<params_reflector_t> reflector_params;
  if (rank_v0 == rank) {
    auto compute_parameters_func = unwrapping([](const x0_and_squares_t& x0_and_norm) {
      const T norm = std::sqrt(x0_and_norm.second);

      // clang-format off
      const params_reflector_t params{
        x0_and_norm.first,
        std::signbit(std::real(params.x0)) ? norm : -norm
      };
      // clang-format on

      const T tau = (params.y - params.x0) / params.y;

      return hpx::make_tuple(params, tau);
    });

    hpx::tie(reflector_params, tau) =
        hpx::split_future(hpx::dataflow(compute_parameters_func, x0_and_squares));

    auto bcast_params_func = unwrapping([](const auto& params, const T tau, auto&& comm_wrapper) {
      const T data[3] = {params.x0, params.y, tau};
      broadcast::send(comm_wrapper.ref().colCommunicator(), make_data(data, 3));
    });

    hpx::dataflow(bcast_params_func, reflector_params, tau, serial_comm());
  }
  else {
    auto bcast_params_func = unwrapping([rank = rank_v0.row()](auto&& comm_wrapper) {
      T data[3];
      broadcast::receive_from(rank, comm_wrapper.ref().colCommunicator(), make_data(data, 3));
      params_reflector_t params;
      params.x0 = data[0];
      params.y = data[1];
      const T tau = data[2];
      return hpx::make_tuple(params, tau);
    });

    hpx::tie(reflector_params, tau) = hpx::split_future(hpx::dataflow(bcast_params_func, serial_comm()));
  }

  // 1A/3 COMPUTE REFLECTOR COMPONENTs
  for (const LocalTileIndex& index_v_loc : ai_panel) {
    const SizeType index_v_row = dist.template globalTileFromLocalTile<Coord::Row>(index_v_loc.row());

    const bool has_first_component = (index_v_row == ai_start.row());

    auto compute_reflector_func = unwrapping([=](auto&& tile_v, const params_reflector_t& params) {
      if (has_first_component)
        tile_v(index_el_x0) = params.y;

      const SizeType first_tile_element = has_first_component ? index_el_x0.row() + 1 : 0;

      if (first_tile_element > tile_v.size().rows() - 1)
        return;

      T* v = tile_v.ptr({first_tile_element, index_el_x0.col()});
      blas::scal(tile_v.size().rows() - first_tile_element,
                 typename TypeInfo<T>::BaseType(1) / (params.x0 - params.y), v, 1);
    });

    hpx::dataflow(compute_reflector_func, a(index_v_loc), reflector_params);
  }

  return tau;
}

template <class T>
void update_trailing_panel(MatrixT<T>& a,
                           const common::IterableRange2D<SizeType, dlaf::matrix::LocalTile_TAG> ai_panel,
                           const GlobalTileIndex ai_start, const TileElementIndex index_el_x0,
                           hpx::shared_future<T> tau,
                           common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using hpx::util::unwrapping;
  using common::make_data;
  using namespace comm::sync;

  const auto& dist = a.distribution();

  const SizeType nb = a.blockSize().rows();

  // 1B UPDATE TRAILING PANEL
  // for each tile in the panel, consider just the trailing panel
  // i.e. all rows (height = reflector), just columns to the right of the current reflector
  if (!(index_el_x0.col() + 1 < nb))
    return;

  // 1B/1 Compute W
  // TODO it should be panel.cols() instead of nb
  MatrixT<T> w({nb, 1}, dist.blockSize());
  set_to_zero(w);

  for (const LocalTileIndex& index_a_loc : ai_panel) {
    const SizeType index_a_row = dist.template globalTileFromLocalTile<Coord::Row>(index_a_loc.row());

    const bool has_first_component = (index_a_row == ai_start.row());

    // GEMV w = Pt* . V
    auto compute_w_func = unwrapping([=](auto&& tile_w, const auto& tile_a) {
      const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

      // clang-format off
      TileElementIndex        pt_start  {first_element, index_el_x0.col() + 1};
      TileElementSize         pt_size   {tile_a.size().rows() - pt_start.row(), tile_a.size().cols() - pt_start.col()};

      TileElementIndex        v_start   {first_element, index_el_x0.col()};
      const TileElementIndex  w_start   {0, 0};
      // clang-format on

      if (has_first_component) {
        const TileElementSize offset{1, 0};

        const T fake_v = 1;
        // clang-format off
        blas::gemv(blas::Layout::ColMajor,
            blas::Op::ConjTrans,
            offset.rows(), pt_size.cols(),
            static_cast<T>(1),
            tile_a.ptr(pt_start), tile_a.ld(),
            &fake_v, 1,
            static_cast<T>(0),
            tile_w.ptr(w_start), 1);
        // clang-format on

        pt_start = pt_start + offset;
        v_start = v_start + offset;
        pt_size = pt_size - offset;
      }

      if (pt_start.isIn(tile_a.size())) {
        // W += 1 . A* . V
        // clang-format off
        blas::gemv(blas::Layout::ColMajor,
            blas::Op::ConjTrans,
            pt_size.rows(), pt_size.cols(),
            static_cast<T>(1),
            tile_a.ptr(pt_start), tile_a.ld(),
            tile_a.ptr(v_start), 1,
            1,
            tile_w.ptr(w_start), 1);
        // clang-format on
      }
    });

    hpx::dataflow(compute_w_func, w(LocalTileIndex{0, 0}), a.read(index_a_loc));
  }

  // all-reduce W
  auto reduce_w_func = unwrapping([](auto&& tile_w, auto&& comm_wrapper) {
    all_reduce(comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w));
  });

  hpx::dataflow(reduce_w_func, w(LocalTileIndex{0, 0}), serial_comm());

  // 1B/2 UPDATE TRAILING PANEL
  for (const LocalTileIndex& index_a_loc : ai_panel) {
    const SizeType index_a_row = dist.template globalTileFromLocalTile<Coord::Row>(index_a_loc.row());

    const bool has_first_component = (index_a_row == ai_start.row());

    // GER Pt = Pt - tau . v . w*
    auto apply_reflector_func = unwrapping([=](auto&& tile_a, const T tau, const auto& tile_w) {
      const SizeType first_element = has_first_component ? index_el_x0.row() : 0;

      // clang-format off
      TileElementIndex        pt_start{first_element, index_el_x0.col() + 1};
      TileElementSize         pt_size {tile_a.size().rows() - pt_start.row(), tile_a.size().cols() - pt_start.col()};

      TileElementIndex        v_start {first_element, index_el_x0.col()};
      const TileElementIndex  w_start {0, 0};
      // clang-format on

      if (has_first_component) {
        const TileElementSize offset{1, 0};

        // Pt = Pt - tau * v[0] * w*
        // clang-format off
        const T fake_v = 1;
        blas::ger(blas::Layout::ColMajor,
            1, pt_size.cols(),
            -dlaf::conj(tau),
            &fake_v, 1,
            tile_w.ptr(w_start), 1,
            tile_a.ptr(pt_start), tile_a.ld());
        // clang-format on

        pt_start = pt_start + offset;
        v_start = v_start + offset;
        pt_size = pt_size - offset;
      }

      if (pt_start.isIn(tile_a.size())) {
        // Pt = Pt - tau * v * w*
        // clang-format off
        blas::ger(blas::Layout::ColMajor,
            pt_size.rows(), pt_size.cols(),
            -dlaf::conj(tau),
            tile_a.ptr(v_start), 1,
            tile_w.ptr(w_start), 1,
            tile_a.ptr(pt_start), tile_a.ld());
        // clang-format on
      }
    });

    hpx::dataflow(apply_reflector_func, a(index_a_loc), tau, w.read(LocalTileIndex{0, 0}));
  }
}

template <class T, class MatrixLikeT>
void compute_w(PanelT<Coord::Col, T>& w, MatrixLikeT& v, ConstMatrixT<T>& t) {
  auto trmm_func =
      hpx::util::unwrapping([](auto&& tile_w, const auto& tile_v, const auto& tile_t) -> void {
        dlaf::tile::lacpy(tile_v, tile_w);

        // W = V . T
        // clang-format off
        blas::trmm(blas::Layout::ColMajor,
            blas::Side::Right, blas::Uplo::Upper, blas::Op::NoTrans, blas::Diag::NonUnit,
            tile_w.size().rows(), tile_w.size().cols(),
            static_cast<T>(1),
            tile_t.ptr(), tile_t.ld(),
            tile_w.ptr(), tile_w.ld());
        // clang-format on
      });

  for (const auto& index_tile_w : w) {
    // clang-format off
    FutureTile<T>      tile_w = w(index_tile_w);
    FutureConstTile<T> tile_v = v.read(index_tile_w);
    FutureConstTile<T> tile_t = t.read(LocalTileIndex{0, 0});
    // clang-format on

    hpx::dataflow(trmm_func, std::move(tile_w), std::move(tile_v), std::move(tile_t));
  }
}

// TODO document that it stores in Xcols just the ones for which he is not on the right row,
// otherwise it directly compute also gemm2 inside Xrows
template <class T>
void compute_x(comm::IndexT_MPI reducer_col, PanelT<Coord::Col, T>& x, PanelT<Coord::Row, T>& xt,
               const LocalTileSize at_offset, ConstMatrixT<T>& a, ConstPanelT<Coord::Col, T>& w,
               ConstPanelT<Coord::Row, T>& wt, common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using hpx::util::unwrapping;
  using dlaf::common::make_data;
  using dlaf::comm::sync::reduce;

  const auto dist = a.distribution();
  const auto rank = dist.rankIndex();

  for (SizeType i = at_offset.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = limit - 1; j >= at_offset.cols(); --j) {
      const LocalTileIndex index_a_loc{i, j};

      const GlobalTileIndex index_a = dist.globalTileIndex(index_a_loc);

      const bool is_first = j == limit - 1;
      const bool is_diagonal_tile = (index_a.row() == index_a.col());

      if (is_diagonal_tile) {
        const LocalTileIndex index_x{index_a_loc.row(), 0};
        const LocalTileIndex index_w{index_a_loc.row(), 0};

        // clang-format off
        FutureTile<T>       tile_x = x(index_x);
        FutureConstTile<T>  tile_a = a.read(index_a_loc);
        FutureConstTile<T>  tile_w = w.read(index_w);
        // clang-format on

        hpx::dataflow(unwrapping(dlaf::tile::hemm<T, Device::CPU>), blas::Side::Left, blas::Uplo::Lower,
                      static_cast<T>(1), std::move(tile_a), std::move(tile_w),
                      static_cast<T>(is_first ? 0 : 1), std::move(tile_x));
      }
      else {
        // A  . W
        {
          // Note:
          // Since it is not a diagonal tile, otherwise it would have been managed in the previous
          // branch, the second operand is not available in W but it is accessible through the
          // support panel Wt.
          // However, since we are still computing the "straight" part, the result can be stored
          // in the "local" panel X.
          const LocalTileIndex index_x{index_a_loc.row(), 0};
          const LocalTileIndex index_wt{0, index_a_loc.col()};

          // clang-format off
          FutureTile<T>       tile_x = x(index_x);
          FutureConstTile<T>  tile_a = a.read(index_a_loc);
          FutureConstTile<T>  tile_w = wt.read(index_wt);
          // clang-format on

          hpx::dataflow(unwrapping(dlaf::tile::gemm<T, Device::CPU>), blas::Op::NoTrans,
                        blas::Op::NoTrans, static_cast<T>(1), std::move(tile_a), std::move(tile_w),
                        static_cast<T>(is_first ? 0 : 1), std::move(tile_x));
        }

        // A* . W
        {
          // Note:
          // Here we are considering the hermitian part of A, so coordinates have to be "mirrored".
          // So, first step is identifying the mirrored cell coordinate, i.e. swap row/col, together
          // with realizing if the new coord lays on an owned row or not.
          // If yes, the result can be stored in the X, otherwise Xt support panel will be used.
          // For what concerns the second operand, it can be found for sure in W. In fact, the
          // multiplication requires matching col(A) == row(W), but since coordinates are mirrored,
          // we are mathing row(A) == row(W), so it is local by construction.
          const auto owner = dist.template rankGlobalTile<Coord::Row>(index_a.col());

          const LocalTileIndex index_x{dist.template localTileFromGlobalTile<Coord::Row>(index_a.col()),
                                       0};
          const LocalTileIndex index_xt{0, index_a_loc.col()};

          const LocalTileIndex index_w{index_a_loc.row(), 0};

          const bool is_first_xt = (dist.rankIndex().row() != owner) && is_first;

          // clang-format off
          FutureTile<T>       tile_x = (dist.rankIndex().row() == owner) ? x(index_x) : xt(index_xt);
          FutureConstTile<T>  tile_a = a.read(index_a_loc);
          FutureConstTile<T>  tile_w = w.read(index_w);
          // clang-format on

          hpx::dataflow(unwrapping(dlaf::tile::gemm<T, Device::CPU>), blas::Op::ConjTrans,
                        blas::Op::NoTrans, static_cast<T>(1), std::move(tile_a), std::move(tile_w),
                        static_cast<T>(is_first_xt ? 0 : 1), std::move(tile_x));
        }
      }
    }
  }

  // Note:
  // At this point, partial results of X and Xt are available in the panels, and they have to be reduced,
  // both row-wise and col-wise.
  // The final X result will be available just on Ai panel column.

  // Note:
  // The first step in reducing partial results distributed over X and Xt, it is to reduce the row
  // panel Xt col-wise, by collecting all Xt results on the rank which can "mirror" the result on its
  // rows (i.e. diagonal). So, for each tile of the row panel, select who is the "diagonal" rank that can
  // mirror and reduce on it.
  for (const auto& index_xt : xt) {
    const auto index_k = dist.template globalTileFromLocalTile<Coord::Col>(index_xt.col());
    const auto rank_owner_row = dist.template rankGlobalTile<Coord::Row>(index_k);

    if (rank_owner_row == rank.row()) {
      // Note:
      // Since it is the owner, it has to perform the "mirroring" of the results from columns to
      // rows.
      auto reduce_x_func = unwrapping([=](auto&& tile_x, auto&& comm_wrapper) {
        auto comm_grid = comm_wrapper.ref();

        // TODO here the IN-PLACE is happening (because Xt is not used for this tile on this rank)
        reduce(comm_grid.colCommunicator(), MPI_SUM, make_data(tile_x));
      });

      const auto i = dist.template localTileFromGlobalTile<Coord::Row>(index_k);
      FutureTile<T> tile_x = x({i, 0});

      hpx::dataflow(reduce_x_func, std::move(tile_x), serial_comm());
    }
    else {
      auto reduce_x_func = unwrapping([=](auto&& tile_x_conj, auto&& comm_wrapper) {
        auto comm_grid = comm_wrapper.ref();

        reduce(rank_owner_row, comm_grid.colCommunicator(), MPI_SUM, make_data(tile_x_conj));
      });

      FutureConstTile<T> tile_x = xt.read(index_xt);

      hpx::dataflow(reduce_x_func, std::move(tile_x), serial_comm());
    }
  }

  // Note:
  // At this point partial results are all collected in X (Xt has been embedded in previous step),
  // so the last step needed is to reduce these last partial results in the final results.
  // The result is needed just on the column with reflectors.
  auto reduce_x_func = unwrapping([reducer_col](auto&& tile_x, auto&& comm_wrapper) {
    // TODO Again, here the IN-PLACE is happening
    auto comm = comm_wrapper.ref().rowCommunicator();
    if (comm.rank() == reducer_col)
      reduce(comm, MPI_SUM, make_data(tile_x));
    else
      reduce(reducer_col, comm, MPI_SUM, make_data(tile_x));
  });

  // TODO readonly tile management
  for (const auto& index_x_loc : x)
    hpx::dataflow(reduce_x_func, x(index_x_loc), serial_comm());
}

template <class T>
void compute_w2(MatrixT<T>& w2, ConstPanelT<Coord::Col, T>& w, ConstPanelT<Coord::Col, T>& x,
                common::Pipeline<comm::CommunicatorGrid>& serial_comm) {
  using hpx::util::unwrapping;
  using common::make_data;
  using namespace comm::sync;

  // GEMM W2 = W* . X
  for (const auto& index_tile : w) {
    const T beta = (index_tile.row() == 0) ? 0 : 1;

    // clang-format off
    FutureTile<T>       tile_w2 = w2(LocalTileIndex{0, 0});
    FutureConstTile<T>  tile_w  = w.read(index_tile);
    FutureConstTile<T>  tile_x  = x.read(index_tile);
    // clang-format on

    hpx::dataflow(unwrapping(dlaf::tile::gemm<T, Device::CPU>), blas::Op::ConjTrans, blas::Op::NoTrans,
                  static_cast<T>(1), std::move(tile_w), std::move(tile_x), beta, std::move(tile_w2));
  }

  // all-reduce instead of computing it on each node, everyone in the panel should have it
  auto all_reduce_w2 = unwrapping([](auto&& tile_w2, auto&& comm_wrapper) {
    all_reduce(comm_wrapper.ref().colCommunicator(), MPI_SUM, make_data(tile_w2));
  });

  FutureTile<T> tile_w2 = w2(LocalTileIndex{0, 0});

  hpx::dataflow(std::move(all_reduce_w2), std::move(tile_w2), serial_comm());
}

template <class T, class MatrixLikeT>
void update_x(PanelT<Coord::Col, T>& x, ConstMatrixT<T>& w2, MatrixLikeT& v) {
  using hpx::util::unwrapping;

  // GEMM X = X - 0.5 . V . W2
  for (const auto& index_row : v) {
    // clang-format off
    FutureTile<T>       tile_x  = x(index_row);
    FutureConstTile<T>  tile_v  = v.read(index_row);
    FutureConstTile<T>  tile_w2 = w2.read(LocalTileIndex{0, 0});
    // clang-format on

    hpx::dataflow(unwrapping(dlaf::tile::gemm<T, Device::CPU>), blas::Op::NoTrans, blas::Op::NoTrans,
                  static_cast<T>(-0.5), std::move(tile_v), std::move(tile_w2), static_cast<T>(1),
                  std::move(tile_x));
  }
}

template <class T>
void update_a(const LocalTileSize& at_start, MatrixT<T>& a, ConstPanelT<Coord::Col, T>& x,
              ConstPanelT<Coord::Row, T>& vt, ConstPanelT<Coord::Col, T>& v,
              ConstPanelT<Coord::Row, T>& xt) {
  using hpx::util::unwrapping;

  const auto dist = a.distribution();

  for (SizeType i = at_start.rows(); i < dist.localNrTiles().rows(); ++i) {
    const auto limit = dist.template nextLocalTileFromGlobalTile<Coord::Col>(
        dist.template globalTileFromLocalTile<Coord::Row>(i) + 1);
    for (SizeType j = at_start.cols(); j < limit; ++j) {
      const LocalTileIndex index_at{i, j};

      const GlobalTileIndex index_a = dist.globalTileIndex(index_at);  // TODO possible FIXME
      const bool is_diagonal_tile = (index_a.row() == index_a.col());

      if (is_diagonal_tile) {
        // HER2K
        const LocalTileIndex index_x{index_at.row(), 0};

        // clang-format off
        FutureTile<T>       tile_a = a(index_at);
        FutureConstTile<T>  tile_v = v.read(index_x);
        FutureConstTile<T>  tile_x = x.read(index_x);
        // clang-format on

        const T alpha = -1;  // TODO T must be a signed type
        hpx::dataflow(unwrapping(dlaf::tile::her2k<T, Device::CPU>), blas::Uplo::Lower,
                      blas::Op::NoTrans, alpha, std::move(tile_v), std::move(tile_x),
                      static_cast<typename TypeInfo<T>::BaseType>(1), std::move(tile_a));
      }
      else {
        // GEMM A: X . V*
        {
          const LocalTileIndex index_x{index_at.row(), 0};

          // clang-format off
          FutureTile<T>       tile_a = a(index_at);
          FutureConstTile<T>  tile_x = x.read(index_x);
          FutureConstTile<T>  tile_v = vt.read({0, index_at.col()});
          // clang-format on

          const T alpha = -1;  // TODO T must be a sigend type
          hpx::dataflow(unwrapping(dlaf::tile::gemm<T, Device::CPU>), blas::Op::NoTrans,
                        blas::Op::ConjTrans, alpha, std::move(tile_x), std::move(tile_v),
                        static_cast<T>(1), std::move(tile_a));
        }

        // GEMM A: V . X*
        {
          // clang-format off
          FutureTile<T>       tile_a = a(index_at);
          FutureConstTile<T>  tile_v = v.read({index_at.row(), 0});
          FutureConstTile<T>  tile_x = xt.read({0, index_at.col()});
          // clang-format on

          const T alpha = -1;  // TODO T must be a sigend type
          hpx::dataflow(unwrapping(dlaf::tile::gemm<T, Device::CPU>), blas::Op::NoTrans,
                        blas::Op::ConjTrans, alpha, std::move(tile_v), std::move(tile_x),
                        static_cast<T>(1), std::move(tile_a));
        }
      }
    }
  }
}

}

/// Distributed implementation of reduction to band
/// @return a list of shared futures of vectors, where each vector contains a block of taus
template <class T>
std::vector<hpx::shared_future<std::vector<T>>> ReductionToBand<Backend::MC, Device::CPU, T>::call(
    comm::CommunicatorGrid grid, Matrix<T, Device::CPU>& mat_a) {
  using hpx::execution::parallel_executor;
  using hpx::resource::get_thread_pool;
  using hpx::resource::pool_exists;
  using hpx::threads::thread_priority;

  using namespace comm;
  using namespace comm::sync;

  using common::iterate_range2d;
  using common::make_data;
  using hpx::util::unwrapping;
  using matrix::Distribution;

  using factorization::internal::computeTFactor;

  DLAF_ASSERT(equal_process_grid(mat_a, grid), mat_a, grid);
  DLAF_ASSERT(square_size(mat_a), mat_a.size());
  DLAF_ASSERT(square_blocksize(mat_a), mat_a.blockSize());

  parallel_executor executor_hp(&get_thread_pool("default"), thread_priority::high);
  parallel_executor executor_normal(&get_thread_pool("default"), thread_priority::default_);

  auto executor_mpi = (pool_exists("mpi"))
                          ? parallel_executor(&get_thread_pool("mpi"), thread_priority::high)
                          : executor_hp;

  const auto& dist = mat_a.distribution();
  const comm::Index2D rank = dist.rankIndex();

  const SizeType nb = mat_a.blockSize().rows();
  // TODO not yet implemented for the moment the panel is tile-wide
  // const SizeType band_size = nb;

  const Distribution dist_block(LocalElementSize{nb, nb}, dist.blockSize());

  common::Pipeline<comm::CommunicatorGrid> serial_comm(std::move(grid));

  std::vector<hpx::shared_future<std::vector<T>>> taus;

  PanelT<Coord::Col, T> v(dist);
  PanelT<Coord::Row, T> vt(dist);
  PanelT<Coord::Col, T> w(dist);
  PanelT<Coord::Row, T> wt(dist);
  PanelT<Coord::Col, T> x(dist);
  PanelT<Coord::Row, T> xt(dist);

  for (SizeType j_panel = 0; j_panel < (dist.nrTiles().cols() - 1); ++j_panel) {
    MatrixT<T> t(dist_block);  // used just by the column

    const GlobalTileIndex ai_start{GlobalTileIndex{j_panel, j_panel} + GlobalTileSize{1, 0}};
    const GlobalTileIndex at_start{ai_start + GlobalTileSize{0, 1}};

    const comm::Index2D rank_v0 = dist.rankGlobalTile(ai_start);

    const LocalTileSize ai_offset{
        dist.template nextLocalTileFromGlobalTile<Coord::Row>(ai_start.row()),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(ai_start.col()),
    };
    const LocalTileSize at_offset{
        ai_offset.rows(),
        dist.template nextLocalTileFromGlobalTile<Coord::Col>(at_start.col()),
    };

    const auto ai_panel =
        iterate_range2d(LocalTileIndex{0, 0} + ai_offset,
                        LocalTileSize{dist.localNrTiles().rows() - ai_offset.rows(), 1});

    const bool is_panel_rank_col = rank_v0.col() == rank.col();

    v.set_offset(at_offset);

    // 1. PANEL
    if (is_panel_rank_col) {
      const SizeType k_reflectors = [v0_size = mat_a.tileSize(ai_start)]() {
        return std::min(v0_size.cols(), v0_size.rows());
      }();

      common::internal::vector<hpx::shared_future<T>> taus_panel;

      // Note:
      // for each column in the panel, compute reflector and update panel
      // if this block has the last reflector, that would be just the first 1, skip the last column
      for (SizeType j_reflector = 0; j_reflector < k_reflectors; ++j_reflector) {
        const TileElementIndex index_el_x0{j_reflector, j_reflector};

        auto tau = compute_reflector(mat_a, ai_panel, ai_start, index_el_x0, serial_comm);
        taus_panel.push_back(tau);
        update_trailing_panel(mat_a, ai_panel, ai_start, index_el_x0, tau, serial_comm);
      }

      taus.emplace_back(hpx::when_all(taus_panel)
                            .then(unwrapping(hpx::util::unwrap<std::vector<hpx::shared_future<T>>>)));

      // Prepare T and V for the next step

      computeTFactor<Backend::MC>(k_reflectors, mat_a, ai_start, taus_panel, t, serial_comm);

      // Note:
      // Reflectors are stored in the lower triangular part of the A matrix leading to sharing memory
      // between reflectors and results, which are in the upper triangular part. The problem exists only
      // for the first tile (of the V, i.e. band excluded). Since refelectors will be used in next
      // computations, they should be well-formed, i.e. a unit lower trapezoidal matrix. For this reason,
      // a support tile is used, where just the reflectors values are copied, the diagonal is set to 1
      // and the rest is zeroed out.
      if (rank_v0 == rank) {
        auto setup_V0_func = unwrapping([](auto&& tile_v, const auto& tile_a) {
          dlaf::tile::lacpy(tile_a, tile_v);

          // set upper part to zero and 1 on diagonal (reflectors)
          // clang-format off
          lapack::laset(lapack::MatrixType::Upper,
              tile_v.size().rows(), tile_v.size().cols(),
              T(0), // off diag
              T(1), // on  diag
              tile_v.ptr(), tile_v.ld());
          // clang-format on
        });

        const LocalTileIndex v0_index(ai_offset.rows(), ai_offset.cols());
        hpx::dataflow(setup_V0_func, v(v0_index), mat_a.read(v0_index));
      }

      // The rest of the V panel of reflectors can just point to the values in A, since they are
      // well formed in-place.
      for (auto row = dist.template nextLocalTileFromGlobalTile<Coord::Row>(ai_start.row() + 1);
           row < dist.localNrTiles().rows(); ++row) {
        const LocalTileIndex idx{row, ai_offset.cols()};
        v.set_tile(idx, mat_a.read(idx));
      }
    }

    vt.set_offset(at_offset);
    matrix::broadcast(executor_mpi, rank_v0.col(), v, vt, serial_comm);

    // UPDATE TRAILING MATRIX

    // COMPUTE W
    // W = V . T
    w.set_offset(at_offset);

    if (is_panel_rank_col)
      compute_w(w, v, t);

    wt.set_offset(at_offset);
    matrix::broadcast(executor_mpi, rank_v0.col(), w, wt, serial_comm);

    // COMPUTE X
    // X = At . W
    // Since At is hermitian, just the lower part is referenced.
    // When the tile is not part of the main diagonal, the same tile has to be used for two computations
    // that will contribute to two different rows of X: the ones indexed with row and col.
    // This is achieved by storing the two results in two different workspaces: X and X_conj respectively.

    // They have to be set to zero, because all tiles are going to be reduced, and some tiles may not get
    // "initialized" during computation, so they should not contribute with any spurious value to the final result
    x.set_offset(at_offset, true);
    xt.set_offset(at_offset, true);

    compute_x(rank_v0.col(), x, xt, at_offset, mat_a, w, wt, serial_comm);

    // Now the intermediate result for X is available on the panel column rank,
    // which has locally all the needed stuff for updating X and finalize the result

    // W2 can be computed by the panel column rank only, it is the only one that has the X
    if (is_panel_rank_col) {
      MatrixT<T> w2 = std::move(t);

      // Note:
      // Not all ranks in the column always hold at least a tile in the panel Ai, but all ranks in
      // the column are going to participate to the reduce. For them, it is important to set the
      // partial result W2 to zero.
      set_to_zero(w2);

      compute_w2(w2, w, x, serial_comm);

      update_x(x, w2, v);
    }

    matrix::broadcast(executor_mpi, rank_v0.col(), x, xt, serial_comm);

    // UPDATE
    // At = At - X . V* + V . X*
    update_a(at_offset, mat_a, x, vt, v, xt);

    w.reset();
    wt.reset();

    v.reset();
    vt.reset();
    x.reset();
    xt.reset();
  }

  return taus;
}

/// ---- ETI
#define DLAF_EIGENSOLVER_MC_ETI(KWORD, DATATYPE) \
  KWORD template struct ReductionToBand<Backend::MC, Device::CPU, DATATYPE>;

DLAF_EIGENSOLVER_MC_ETI(extern, float)
DLAF_EIGENSOLVER_MC_ETI(extern, double)
DLAF_EIGENSOLVER_MC_ETI(extern, std::complex<float>)
DLAF_EIGENSOLVER_MC_ETI(extern, std::complex<double>)

}
}
}
