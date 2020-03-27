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

#include <complex>

namespace dlaf {
namespace comm {

/// @brief mapper between language types and basic MPI_Datatype
template <typename T>
struct mpi_datatype {
  static MPI_Datatype type;
};

}
}
