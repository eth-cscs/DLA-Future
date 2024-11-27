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

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iosfwd>
#include <iostream>

#include <pika/init.hpp>
#include <pika/runtime.hpp>

#include <dlaf/types.h>

namespace dlaf {
/// DLA-Future tuning parameters.
///
/// Holds the value of the parameters that can be used to tune DLA-Future.
/// - debug_dump_cholesky_factorization_data:
///     Enable dump of Cholesky factorization input/output data to "cholesky-factorization.h5" file
///     that will be created in the working folder (it should not exist before the execution).
///     Set with environment variable DLAF_DEBUG_CHOLESKY_FACTORIZATION_DATA.
/// - debug_dump_generalized_to_standard_data:
///     Enable dump of gen_to_std input/output data to "generalized-to-standard.h5" file that will be
///     created in the working folder (it should not exist before the execution).
///     Set with environment variable DLAF_DEBUG_GENERALIZED_TO_STANDARD_DATA.
/// - debug_dump_generalized_eigensolver_data:
///     Enable dump of generalized eigensolver input/output data to "eigensolver.h5" file that will be
///     created in the working folder (it should not exist before the execution).
///     Set with environment variable DLAF_DEBUG_GENERALIZED_EIGENSOLVER_DATA.
/// - debug_dump_eigensolver_data:
///     Enable dump of eigensolver input/output data to "eigensolver.h5" file that will be created in the
///     working folder (it should not exist before the execution).
///     Set with environment variable DLAF_DEBUG_DUMP_EIGENSOLVER_DATA.
/// - debug_dump_reduction_to_band_data:
///     Enable dump of reduction_to_band input/output data to "reduction_to_band.h5" file that will be
///     created in the working folder (it should not exist before the execution).
///    environment variable
///     DLAF_DEBUG_DUMP_EIGENSOLVER_DATA.
/// - debug_dump_band_to_tridiagonal_data:
///     Enable dump of band_to_trigiagonal input/output data to "band_to_tridiagonal.h5" file that will
///     be created in the working folder (it should not exist before the execution).
///     environment variable DLAF_DEBUG_DUMP_BAND_TO_TRIDIAGONAL_DATA.
/// - debug_dump_tridiag_solver_data:
///     Enable dump of tridiagonal solver input/output data to "tridiagonal.h5" file that will before
///     created in the working folder (it should not exist before the execution).
///     Set with environment variable DLAF_DEBUG_DUMP_TRIDIAG_SOLVER_DATA.
/// - tfactor_nworkers:
///     The maximum number of threads/stream to use for computing tfactor (e.g. which is used for
///     instance in red2band and its backtransformation). Set with --dlaf:tfactor-nworkers or env
///     variable DLAF_TFACTOR_NWORKERS.
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
#if PIKA_VERSION_FULL >= 0x001600  // >= 0.22.0
    if (!pika::is_runtime_initialized()) {
      std::cerr
          << "[ERROR] Trying to initialize DLA-Future tune parameters but the pika runtime is not initialized. Make sure pika is initialized first.\n";
      std::terminate();
    }
#endif

    const auto default_pool_thread_count =
        pika::resource::get_thread_pool("default").get_os_thread_count();
    red2band_panel_nworkers = std::max<std::size_t>(1, default_pool_thread_count / 2);
    tridiag_rank1_nworkers = default_pool_thread_count;
  }
  bool debug_dump_cholesky_factorization_data = false;
  bool debug_dump_generalized_to_standard_data = false;
  bool debug_dump_generalized_eigensolver_data = false;
  bool debug_dump_eigensolver_data = false;
  bool debug_dump_reduction_to_band_data = false;
  bool debug_dump_band_to_tridiagonal_data = false;
  bool debug_dump_tridiag_solver_data = false;

  std::size_t tfactor_nworkers = 1;
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
