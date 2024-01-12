//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2024, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file

#include <cstdint>
#include <iosfwd>

#include <pika/runtime.hpp>

#include <dlaf/types.h>

namespace dlaf {
/// DLA-Future tuning parameters.
///
/// Holds the value of the parameters that can be used to tune DLA-Future.
/// - debug_dump_trisolver_data:
///     Enable dump of trisolver input/output data to "trid-ref.h5" file that will be created in the
///     working folder (it should not exist before the execution).
///     WARNING: just a single execution can be dumped on disk, and any subsequent call fails.
///     Set with environment variable DLAF_DEBUG_DUMP_TRISOLVER_DATA.
/// - red2band_panel_nworkers:
///     The maximum number of threads to use for computing the panel in the reduction to band algorithm.
///     Set with --dlaf:red2band-panel-nworkers or env variable DLAF_RED2BAND_PANEL_NWORKERS.
/// - red2band_barrier_busy_wait_us:
///     The duration in microseconds to busy-wait in barriers in the reduction to band algorithm.
///     Set with --dlaf:red2band-barrier-busy-wait-us or env variable DLAF_RED2BAND_BARRIER_BUSY_WAIT_US.
/// - tridiag_rank1_nworkers:
///     The maximum number of threads to use for computing rank1 problem solution in tridiagonal solver
///     algorithm. Set with --dlaf:tridiag-rank1-nworkers or env variable DLAF_TRIDIAG_RANK1_NWORKERS.
/// - tridiag_rank1_barrier_busy_wait_us:
///     The duration in microseconds to busy-wait in barriers when computing rank1 problem solution in
///     the tridiagonal solver algorithm. Set with --dlaf:tridiag-rank1-barrier-busy-wait-us or env
///     variable DLAF_TRIDIAG_RANK1_BARRIER_BUSY_WAIT_US.
/// - eigensolver_min_band:
///     The minimum value to start looking for a divisor of the block size.
///     Set with --dlaf:eigensolver-min-band or env variable DLAF_EIGENSOLVER_MIN_BAND.
/// - band_to_tridiag_1d_block_size_base:
///     The 1D block size for band_to_tridiagonal is computed as 1d_block_size_base / nb * nb. The input
///     matrix is distributed with a {nb x nb} block size. Set with
///     --dlaf:band-to-tridiag-1d-block-size-base or env variable
///     DLAF_BAND_TO_TRIDIAG_1D_BLOCK_SIZE_BASE.
/// - bt_band_to_tridiag_hh_apply_group_size:
///     The application of the HH reflector is splitted in smaller applications of the group size
///     reflectors. Set with --dlaf:bt-band-to-tridiag-hh-apply-group-size or env variable
///     DLAF_BT_BAND_TO_TRIDIAG_HH_APPLY_GROUP_SIZE.
/// - communicator_grid_num_pipelines:
///     The default number of row, column, and full communicator pipelins to initialize in
///     CommunicatorGrid. Set with --dlaf:communicator-grid-num-pipelines or env variable
///     DLAF_COMMUNICATOR_GRID_NUM_PIPELINES.
/// Note to developers: Users can change these values, therefore consistency has to be ensured by
/// algorithms.
///
/// Note: debug parameters should not be considered as part of the public API
struct TuneParameters {
  // NOTE: Remember to update the following if you add or change parameters below:
  // - Documentation in the docstring above
  // - The operator<< overload in tune.cpp
  // - updateConfiguration in init.cpp to update the value from command line options and environment
  //   values
  // - getOptionsDescription to add a corresponding command line option

  TuneParameters() {
    // Some parameters require the pika runtime to be initialized since they depend on the number of
    // threads used by the runtime. We initialize them separately in the constructor after checking that
    // pika is initialized.

    // TODO: Check upfront whether pika has been initialized to be able to give a better error message.
    // If pika has not been initialized getting the default thread pool will either trigger an assertion
    // or a segfault.
    const auto default_pool_thread_count =
        pika::resource::get_thread_pool("default").get_os_thread_count();
    red2band_panel_nworkers = std::max<std::size_t>(1, default_pool_thread_count / 2);
    tridiag_rank1_nworkers = default_pool_thread_count;
  }
  bool debug_dump_trisolver_data = false;
  std::size_t red2band_panel_nworkers = 1;
  std::size_t red2band_barrier_busy_wait_us = 1000;
  std::size_t tridiag_rank1_nworkers = 1;
  std::size_t tridiag_rank1_barrier_busy_wait_us = 0;

  SizeType eigensolver_min_band = 100;
  SizeType band_to_tridiag_1d_block_size_base = 8192;
  SizeType bt_band_to_tridiag_hh_apply_group_size = 64;

  std::size_t communicator_grid_num_pipelines = 3;
};

std::ostream& operator<<(std::ostream& os, const TuneParameters& params);
TuneParameters& getTuneParameters();

}
