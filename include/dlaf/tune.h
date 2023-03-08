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

#include <pika/runtime.hpp>
#include <dlaf/types.h>

namespace dlaf {
/// DLA-Future tuning parameters.
///
/// Holds the value of the parameters that can be used to tune DLA-Future.
/// - red2band_panel_nworkers: number of threads to use for computing the panel in the reduction to band algorithm.
/// - eigensolver_min_band: The minimun value to start looking for a divisor of the block size.
///                         Set with --dlaf:eigensolver-min-band or env variable DLAF_EIGENSOLVER_MIN_BAND.
/// Note to developers: Users can change these values, therefore consistency has to be ensured by algorithms.
struct TuneParameters {
  size_t red2band_panel_nworkers =
      std::max<size_t>(1, pika::resource::get_thread_pool("default").get_os_thread_count() / 2);

  SizeType eigensolver_min_band = 100;
  SizeType band_to_tridiag_1d_block_size_base = 8192;
};

TuneParameters& getTuneParameters();

}
