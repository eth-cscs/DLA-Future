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

#include <functional>
#include <utility>

#include <hpx/dataflow.hpp>
#include <hpx/util/unwrap.hpp>

#include "dlaf/eti/function.h"
#include "dlaf/tile.h"

/// @file

namespace dlaf {

template <class T, Device D>
hpx::future<void> dataflow(func_w<T, D>&& fn, hpx::future<Tile<T, D>>&& tile) {
  return hpx::dataflow(hpx::util::unwrapping(std::move(fn)), std::move(tile));
}

template <class T, Device D>
hpx::future<void> dataflow(func_rw<T, D>&& fn, hpx::shared_future<Tile<const T, D>>&& tr,
                           hpx::future<Tile<T, D>>&& tw) {
  return hpx::dataflow(hpx::util::unwrapping(fn), std::move(tr), std::move(tw));
}

template <class T, Device D>
hpx::future<void> dataflow(func_rrw<T, D>&& fn, hpx::shared_future<Tile<const T, D>>&& tr1,
                           hpx::shared_future<Tile<const T, D>>&& tr2, hpx::future<Tile<T, D>>&& tw) {
  return hpx::dataflow(hpx::util::unwrapping(std::move(fn)), std::move(tr1), std::move(tr2),
                       std::move(tw));
}

#define DLAF_DATAFLOW_ETI(KWORD, DTYPE, DEVICE)                                              \
  KWORD template hpx::future<void> dataflow(func_w<DTYPE, DEVICE>&&,                         \
                                            hpx::future<Tile<DTYPE, DEVICE>>&&);             \
  KWORD template hpx::future<void> dataflow(func_rw<DTYPE, DEVICE>&&,                        \
                                            hpx::shared_future<Tile<const DTYPE, DEVICE>>&&, \
                                            hpx::future<Tile<DTYPE, DEVICE>>&&);             \
  KWORD template hpx::future<void> dataflow(func_rrw<DTYPE, DEVICE>&&,                       \
                                            hpx::shared_future<Tile<const DTYPE, DEVICE>>&&, \
                                            hpx::shared_future<Tile<const DTYPE, DEVICE>>&&, \
                                            hpx::future<Tile<DTYPE, DEVICE>>&&);

DLAF_DATAFLOW_ETI(extern, float, Device::CPU)
DLAF_DATAFLOW_ETI(extern, double, Device::CPU)
DLAF_DATAFLOW_ETI(extern, std::complex<float>, Device::CPU)
DLAF_DATAFLOW_ETI(extern, std::complex<double>, Device::CPU)

}
