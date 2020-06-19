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

#include "dlaf/communication/error.h"

#include <complex>

namespace dlaf {
namespace comm {

/// Mapper between language types and basic MPI_Datatype.
template <typename T>
struct mpi_datatype {
  static MPI_Datatype type;
};

/// Helper for mapping also const types.
template <typename T>
struct mpi_datatype<const T> : public mpi_datatype<T> {};

// Forward declare specializations
// clang-format off
template<> MPI_Datatype mpi_datatype<char>                 ::type;
template<> MPI_Datatype mpi_datatype<short>                ::type;
template<> MPI_Datatype mpi_datatype<int>                  ::type;
template<> MPI_Datatype mpi_datatype<long>                 ::type;
template<> MPI_Datatype mpi_datatype<long long>            ::type;
template<> MPI_Datatype mpi_datatype<unsigned char>        ::type;
template<> MPI_Datatype mpi_datatype<unsigned short>       ::type;
template<> MPI_Datatype mpi_datatype<unsigned int>         ::type;
template<> MPI_Datatype mpi_datatype<unsigned long>        ::type;
template<> MPI_Datatype mpi_datatype<unsigned long long>   ::type;
template<> MPI_Datatype mpi_datatype<float>                ::type;
template<> MPI_Datatype mpi_datatype<double>               ::type;
template<> MPI_Datatype mpi_datatype<bool>                 ::type;
template<> MPI_Datatype mpi_datatype<std::complex<float>>  ::type;
template<> MPI_Datatype mpi_datatype<std::complex<double>> ::type;
// clang-format on

}
}
