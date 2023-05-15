//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2023, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <pika/algorithm.hpp>
#include <pika/execution.hpp>

#include "dlaf/eigensolver/tridiag_solver/kernels.h"
#include "dlaf/eigensolver/tridiag_solver/tile_collector.h"
#include "dlaf/matrix/distribution.h"
#include "dlaf/matrix/index.h"
#include "dlaf/matrix/matrix.h"
#include "dlaf/sender/policy.h"
#include "dlaf/sender/when_all_lift.h"
#include "dlaf/types.h"
#include "dlaf/util_matrix.h"

namespace dlaf::eigensolver::internal {

// Calculates the problem size in the tile range [i_begin, i_end)
inline SizeType problemSize(const SizeType i_begin, const SizeType i_end,
                            const matrix::Distribution& distr) {
  const SizeType nb = distr.blockSize().rows();
  const SizeType nbr = distr.tileSize(GlobalTileIndex(i_end - 1, 0)).rows();
  return (i_end - i_begin - 1) * nb + nbr;
}

// The index starts at `0` for tiles in the range [i_begin, i_end)
template <Device D>
void initIndex(const SizeType i_begin, const SizeType i_end, Matrix<SizeType, D>& index) {
  const SizeType nb = index.distribution().blockSize().rows();
  for (SizeType i = i_begin; i < i_end; ++i) {
    const GlobalTileIndex tile_idx(i, 0);
    const SizeType tile_row = (i - i_begin) * nb;
    initIndexTileAsync<D>(tile_row, index.readwrite(tile_idx));
  }
}

// Sorts an index `in_index_tiles` based on values in `vals_tiles` in ascending order into the index
// `out_index_tiles` where `vals_tiles` is composed of two pre-sorted ranges in ascending order that
// are merged, the first is [0, k) and the second is [k, n).
//
template <class T, Device D, class KSender>
void sortIndex(const SizeType i_begin, const SizeType i_end, KSender&& k, Matrix<const T, D>& vec,
               Matrix<const SizeType, D>& in_index, Matrix<SizeType, D>& out_index) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, vec.distribution());
  auto sort_fn = [n](const auto& k, const auto& vec_futs, const auto& in_index_futs,
                     const auto& out_index, [[maybe_unused]] auto&&... ts) {
    DLAF_ASSERT(k <= n, k, n);

    const TileElementIndex zero_idx(0, 0);
    const T* v_ptr = vec_futs[0].get().ptr(zero_idx);
    const SizeType* in_index_ptr = in_index_futs[0].get().ptr(zero_idx);
    SizeType* out_index_ptr = out_index[0].ptr(zero_idx);

    auto begin_it = in_index_ptr;
    auto split_it = in_index_ptr + k;
    auto end_it = in_index_ptr + n;
    if constexpr (D == Device::CPU) {
      auto cmp = [v_ptr](const SizeType i1, const SizeType i2) { return v_ptr[i1] < v_ptr[i2]; };
      pika::merge(pika::execution::par, begin_it, split_it, split_it, end_it, out_index_ptr,
                  std::move(cmp));
    }
    else {
#ifdef DLAF_WITH_GPU
      mergeIndicesOnDevice(begin_it, split_it, end_it, out_index_ptr, v_ptr, ts...);
#endif
    }
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(std::forward<KSender>(k), ex::when_all_vector(tc.read<T, D>(vec)),
                             ex::when_all_vector(tc.read<SizeType, D>(in_index)),
                             ex::when_all_vector(tc.readwrite<SizeType, D>(out_index)));

  ex::start_detached(
      di::transform(di::Policy<DefaultBackend_v<D>>(), std::move(sort_fn), std::move(sender)));
}

// Applies `index` to `in` to get `out`
template <class T, Device D>
void applyIndex(const SizeType i_begin, const SizeType i_end, Matrix<const SizeType, D>& index,
                Matrix<const T, D>& in, Matrix<T, D>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, index.distribution());
  auto applyIndex_fn = [n](const auto& index_futs, const auto& in_futs, const auto& out,
                           [[maybe_unused]] auto&&... ts) {
    const TileElementIndex zero_idx(0, 0);
    const SizeType* i_ptr = index_futs[0].get().ptr(zero_idx);
    const T* in_ptr = in_futs[0].get().ptr(zero_idx);
    T* out_ptr = out[0].ptr(zero_idx);

    if constexpr (D == Device::CPU) {
      for (SizeType i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i_ptr[i]];
      }
    }
    else {
#ifdef DLAF_WITH_GPU
      applyIndexOnDevice(n, i_ptr, in_ptr, out_ptr, ts...);
#endif
    }
  };

  TileCollector tc{i_begin, i_end};

  auto sender = ex::when_all(ex::when_all_vector(tc.read(index)), ex::when_all_vector(tc.read(in)),
                             ex::when_all_vector(tc.readwrite(out)));
  ex::start_detached(
      di::transform(di::Policy<DefaultBackend_v<D>>(), std::move(applyIndex_fn), std::move(sender)));
}

template <Device D>
void invertIndex(const SizeType i_begin, const SizeType i_end, Matrix<const SizeType, D>& in,
                 Matrix<SizeType, D>& out) {
  namespace ex = pika::execution::experimental;
  namespace di = dlaf::internal;

  const SizeType n = problemSize(i_begin, i_end, in.distribution());
  auto inv_fn = [n](const auto& in_tiles_futs, const auto& out_tiles, [[maybe_unused]] auto&&... ts) {
    const TileElementIndex zero(0, 0);
    const SizeType* in_ptr = in_tiles_futs[0].get().ptr(zero);
    SizeType* out_ptr = out_tiles[0].ptr(zero);

    if constexpr (D == Device::CPU) {
      for (SizeType i = 0; i < n; ++i) {
        out_ptr[in_ptr[i]] = i;
      }
    }
    else {
      invertIndexOnDevice(n, in_ptr, out_ptr, ts...);
    }
  };

  TileCollector tc{i_begin, i_end};
  auto sender = ex::when_all(ex::when_all_vector(tc.read(in)), ex::when_all_vector(tc.readwrite(out)));
  ex::start_detached(
      di::transform(di::Policy<DefaultBackend_v<D>>(), std::move(inv_fn), std::move(sender)));
}
}
