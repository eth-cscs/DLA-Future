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

#include "dlaf/communication/internal/datatypes.h"

namespace dlaf {
namespace comm {

extern template MPI_Datatype mpi_datatype<char>::type;
extern template MPI_Datatype mpi_datatype<short>::type;
extern template MPI_Datatype mpi_datatype<int>::type;
extern template MPI_Datatype mpi_datatype<long>::type;
extern template MPI_Datatype mpi_datatype<long long>::type;
extern template MPI_Datatype mpi_datatype<unsigned char>::type;
extern template MPI_Datatype mpi_datatype<unsigned short>::type;
extern template MPI_Datatype mpi_datatype<unsigned int>::type;
extern template MPI_Datatype mpi_datatype<unsigned long>::type;
extern template MPI_Datatype mpi_datatype<unsigned long long>::type;
extern template MPI_Datatype mpi_datatype<float>::type;
extern template MPI_Datatype mpi_datatype<double>::type;
extern template MPI_Datatype mpi_datatype<bool>::type;
extern template MPI_Datatype mpi_datatype<std::complex<float>>::type;
extern template MPI_Datatype mpi_datatype<std::complex<double>>::type;
}
}
