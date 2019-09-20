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

#if __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include <mpi.h>

#include <iostream>

namespace dlaf {
namespace comm {
namespace mpi {

constexpr MPI_Comm NULL_COMMUNICATOR = MPI_COMM_NULL;
constexpr MPI_Comm WORLD_COMMUNICATOR = MPI_COMM_WORLD;

}
}
}

#define MPI_CALL(x)                                                \
  {                                                                \
    auto _ = x;                                                    \
    if (MPI_SUCCESS != _)                                          \
      std::cerr << "MPI ERROR [" << _ << "]: " << #x << std::endl; \
  }

#if __GNUC__
#pragma GCC diagnostic pop
#endif
