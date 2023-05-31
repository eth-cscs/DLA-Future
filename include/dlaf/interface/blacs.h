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

#include <dlaf/communication/communicator.h>
#include <dlaf/communication/communicator_grid.h>
#include <dlaf/matrix/distribution.h>
#include <dlaf/matrix/layout_info.h>

#include <mpi.h>

extern "C" {
MPI_Comm Cblacs2sys_handle(int ictxt);
void Cblacs_get(int ictxt, int inum, int* comm);
extern "C" void Cblacs_gridinfo(int ictxt, int* np, int* mp, int* px, int* py);
}

namespace dlaf::interface::blacs {

int get_grid_context(int* desc);

int get_communicator_context(const int grid_context);
MPI_Comm get_communicator(const int grid_context);

struct DlafSetup {
  dlaf::matrix::Distribution distribution;
  dlaf::matrix::LayoutInfo layout_info;
  dlaf::comm::CommunicatorGrid communicator_grid;
};

DlafSetup from_desc(int* desc);

}