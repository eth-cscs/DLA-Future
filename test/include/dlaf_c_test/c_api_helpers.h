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

#include <utility>

#include <mpi.h>

#include <pika/execution.hpp>

#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/matrix.h>
#include <dlaf_c/grid.h>
#include <dlaf_c/init.h>
#include <dlaf_c_test/blacs.h>

enum class API { dlaf, scalapack };

template <API api>
int c_api_test_inititialize(int pika_argc, const char* pika_argv[], int dlaf_argc,
                            const char* dlaf_argv[], const dlaf::comm::CommunicatorGrid& grid) {
  dlaf_initialize(pika_argc, pika_argv, dlaf_argc, dlaf_argv);

  char grid_order = grid_ordering(MPI_COMM_WORLD, grid.size().rows(), grid.size().cols(),
                                  grid.rank().row(), grid.rank().col());

  int dlaf_context = -1;

  if constexpr (api == API::dlaf) {
    dlaf_context = dlaf_create_grid(MPI_COMM_WORLD, grid.size().rows(), grid.size().cols(), grid_order);
  }
#if DLAF_WITH_SCALAPACK
  else if constexpr (api == API::scalapack) {
    // Create BLACS grid
    Cblacs_get(0, 0, &dlaf_context);  // Default system context
    Cblacs_gridinit(&dlaf_context, &grid_order, grid.size().rows(), grid.size().cols());

    // Create DLAF grid from BLACS context
    dlaf_create_grid_from_blacs(dlaf_context);
  }
#endif
  return dlaf_context;
}

template <API api>
void c_api_test_finalize(int dlaf_context) {
  dlaf_free_grid(dlaf_context);
  dlaf_finalize();

#if DLAF_WITH_SCALAPACK
  if constexpr (api == API::scalapack) {
    Cblacs_gridexit(dlaf_context);
  }
#endif
}

template <typename T, dlaf::Device D>
std::pair<T*, int> top_left_tile(dlaf::matrix::Matrix<T, D>& mat) {
  T* local_ptr;
  int lld;

  if (dlaf::LocalTileIndex(0, 0).isIn(mat.distribution().localNrTiles())) {
    auto toplefttile =
        pika::this_thread::experimental::sync_wait(mat.readwrite(dlaf::LocalTileIndex(0, 0)));

    local_ptr = toplefttile.ptr();
    lld = static_cast<int>(toplefttile.ld());
  }
  else {
    local_ptr = nullptr;
    lld = 1;
  }

  return std::make_pair(local_ptr, lld);
}
