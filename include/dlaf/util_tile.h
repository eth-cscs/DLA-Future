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

#include <cmath>
#include <exception>
#include <random>
#include <string>

#ifndef M_PI
constexpr double M_PI = 3.141592;
#endif

#include <blas.hh>
#include <hpx/local/future.hpp>
#include <hpx/local/unwrap.hpp>

#include "dlaf/common/assert.h"
#include "dlaf/common/index2d.h"
#include "dlaf/common/range2d.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

/// @file

namespace dlaf {
namespace tile {
using matrix::Tile;

/// Returns true if the tile is square.
template <class T, Device D>
bool square_size(const Tile<T, D>& t) noexcept {
  return t.size().rows() == t.size().cols();
}

template <class T>
bool tile_complex_trans(blas::Op op) noexcept {
  bool complextrans = false;
  if (!std::is_same_v<T, ComplexType<T>> || op != blas::Op::Trans)
    complextrans = true;

  return complextrans;
}
}
}
