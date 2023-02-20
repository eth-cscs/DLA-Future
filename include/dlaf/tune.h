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

#include <pika/program_options.hpp>

#include <dlaf/types.h>

namespace dlaf {
/// DLA-Future tuning parameters.
///
/// Holds the value of the parameters that can be used to tune DLA-Future.
/// - eigensolver_min_band: The minimun value to start looking for a divisor of the block size.
///                         Set with --dlaf:eigensolver-min-band or env variable DLAF_EIGENSOLVER_MIN_BAND.
/// - band_to_tridiag_1d_block_size_base:
///     The 1D block size for band_to_tridiagonal is computed as 1d_block_size_base / nb * nb. The input matrix
///     is distributed with a {nb x nb} block size.
///     Set with --dlaf:band-to-tridiag-1d-block-size-base or env variable DLAF_BAND_TO_TRIDIAG_1D_BLOCK_SIZE_BASE.
/// - bt_band_to_tridiag_hh_apply_batch_size:
///     The application of the HH reflector is splitted in smaller applications of the batch_size reflectors.
///     Set with --dlaf:bt-band-to-tridiag-hh-apply-batch-size or env variable
///     DLAF_BT_BAND_TO_TRIDIAG_HH_APPLY_BATCH_SIZE.
/// Note to developers: Users can change these values, therefore consistency has to be ensured by algorithms.
struct TuneParameters {
  SizeType eigensolver_min_band = 100;
  SizeType band_to_tridiag_1d_block_size_base = 8192;
  SizeType bt_band_to_tridiag_hh_apply_batch_size = 64;
};

TuneParameters& getTuneParameters();

}
