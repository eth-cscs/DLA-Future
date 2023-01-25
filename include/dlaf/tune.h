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

#include <pika/program_options.hpp>

#include <dlaf/types.h>

namespace dlaf {
/// DLA-Future tuning parameters.
///
/// Holds the value of the parameters that can be used to tune DLA-Future.
/// - eigensolver_min_band: The minimun value to start looking for a divisor of the block size.
///                         Set with --dlaf:eigensolver-min-band or env variable DLAF_EIGENSOLVER_MIN_BAND.
/// Note to developers: Users can change these values, therefore consistency has to be ensured by algorithms.
struct TuneParameters {
  SizeType eigensolver_min_band = 100;
  SizeType band_to_tridiag_1d_block_size_base = 4096;
};

TuneParameters& getTuneParameters();

}
