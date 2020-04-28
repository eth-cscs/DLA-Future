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

#include <iostream>

#include <mpi.h>

#define MPI_CALL(x)                                                                       \
  do {                                                                                    \
    auto error_code = x;                                                                  \
    DLAF_ASSERT_WITH_ORIGIN(SOURCE_LOCATION(), (MPI_SUCCESS == error_code), "MPI error"); \
  } while (0)
