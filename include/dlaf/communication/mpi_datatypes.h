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

#include "dlaf/mpi_header.h"

namespace dlaf {
namespace comm {

template <typename T>
struct mpi_datatype;

template <>
struct mpi_datatype<int> {
  static constexpr MPI_Datatype type = MPI_INT;
};

}
}
