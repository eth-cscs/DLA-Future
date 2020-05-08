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

/// @file

#include <complex>
#include <functional>

#include "dlaf/tile.h"
#include "dlaf/types.h"

#define DLAF_STD_FUNC_ETI(KWORD, DTYPE, DEVICE)                                   \
  KWORD template class std::function<void(dlaf::Tile<DTYPE, DEVICE> &&)>;         \
  KWORD template class std::function<void(dlaf::Tile<const DTYPE, DEVICE> const&, \
                                          dlaf::Tile<DTYPE, DEVICE>&&)>;          \
  KWORD template class std::function<void(dlaf::Tile<const DTYPE, DEVICE> const&, \
                                          dlaf::Tile<const DTYPE, DEVICE> const&, \
                                          dlaf::Tile<DTYPE, DEVICE>&&)>;

DLAF_STD_FUNC_ETI(extern, float, dlaf::Device::CPU)
DLAF_STD_FUNC_ETI(extern, double, dlaf::Device::CPU)
DLAF_STD_FUNC_ETI(extern, std::complex<float>, dlaf::Device::CPU)
DLAF_STD_FUNC_ETI(extern, std::complex<double>, dlaf::Device::CPU)

namespace dlaf {

template <class T, Device D>
using func_w = std::function<void(Tile<T, D>&&)>;

template <class T, Device D>
using func_rw = std::function<void(Tile<const T, D> const&, Tile<T, D>&&)>;

template <class T, Device D>
using func_rrw = std::function<void(Tile<const T, D> const&, Tile<const T, D> const&, Tile<T, D>&&)>;

}
