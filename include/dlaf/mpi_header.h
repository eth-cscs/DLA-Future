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

#include <mpi.h>

#include <iostream>

#define MPI_CALL(x)                                                \
  {                                                                \
    auto _ = x;                                                    \
    if (MPI_SUCCESS != _)                                          \
      std::cerr << "MPI ERROR [" << _ << "]: " << #x << std::endl; \
  }
