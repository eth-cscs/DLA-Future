//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

#include <blas.hh>

#include "dlaf/common/callable_object.h"
#include "dlaf/matrix/tile.h"

#ifdef DLAF_WITH_CUDA
#include <cublas_v2.h>

#include "dlaf/cublas/error.h"
#include "dlaf/util_cublas.h"
#endif

namespace dlaf {
namespace tile {
using matrix::Tile;

template <class T>
void add(T alpha, const matrix::Tile<const T, Device::CPU>& tile_b,
         const matrix::Tile<T, Device::CPU>& tile_a) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);
  for (auto j = 0; j < tile_a.size().cols(); ++j)
    blas::axpy(tile_a.size().rows(), alpha, tile_b.ptr({0, j}), 1, tile_a.ptr({0, j}), 1);
}

#ifdef DLAF_WITH_CUDA
template <class T>
void add(cublasHandle_t handle, T alpha, const matrix::Tile<const T, Device::GPU>& tile_b,
         const matrix::Tile<T, Device::GPU>& tile_a) {
  DLAF_ASSERT(equal_size(tile_a, tile_b), tile_a, tile_b);
  for (auto j = 0; j < tile_a.size().cols(); ++j)
    tile::internal::CublasAxpy<T>::call(handle, tile_a.size().rows(), util::blasToCublasCast(&alpha),
                                        util::blasToCublasCast(tile_b.ptr({0, j})), 1,
                                        util::blasToCublasCast(tile_a.ptr({0, j})), 1);
}
#endif

DLAF_MAKE_CALLABLE_OBJECT(add);
}
}
