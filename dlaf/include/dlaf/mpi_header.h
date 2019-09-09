//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2019, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#if __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#include <iostream>
#include <mpi.h>

#define MPI_CALL(x) {\
  auto _ = x;\
  if (MPI_SUCCESS != _)\
    std::cout << "MPI ERROR [" << _ << "]: " << #x << std::endl;\
  }

namespace dlaf {
namespace mpi {

// alias that hides old-style cast warning (since it doesn't need to be freed, it is ok)
constexpr MPI_Comm COMM_WORLD = MPI_COMM_WORLD;

}
}

#if __GNUC__
#pragma GCC diagnostic pop
#endif
