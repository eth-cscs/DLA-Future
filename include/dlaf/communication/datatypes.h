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
  static MPI_Datatype type();
};

/// helper for mapping also const types
template <typename T>
struct mpi_datatype<const T> : public mpi_datatype<T> {};

// clang-format off
template<> MPI_Datatype mpi_datatype<char>::type() { return MPI_CHAR; }
template<> MPI_Datatype mpi_datatype<short>::type() { return MPI_SHORT; }
template<> MPI_Datatype mpi_datatype<int>::type() { return MPI_INT; }
template<> MPI_Datatype mpi_datatype<long>::type() { return MPI_LONG; }
template<> MPI_Datatype mpi_datatype<long long>::type() { return MPI_LONG_LONG; }
template<> MPI_Datatype mpi_datatype<unsigned char>::type() { return MPI_UNSIGNED_CHAR; }
template<> MPI_Datatype mpi_datatype<unsigned short>::type() { return MPI_UNSIGNED_SHORT; }
template<> MPI_Datatype mpi_datatype<unsigned int>::type() { return MPI_UNSIGNED; }
template<> MPI_Datatype mpi_datatype<unsigned long>::type() { return MPI_UNSIGNED_LONG; }
template<> MPI_Datatype mpi_datatype<unsigned long long>::type() { return MPI_UNSIGNED_LONG_LONG; }
template<> MPI_Datatype mpi_datatype<float>::type() { return MPI_FLOAT; };
template<> MPI_Datatype mpi_datatype<double>::type() { return MPI_DOUBLE; };
template<> MPI_Datatype mpi_datatype<bool>::type() { return MPI_CXX_BOOL; };
template<> MPI_Datatype mpi_datatype<std::complex<float>>::type() { return MPI_CXX_FLOAT_COMPLEX; }
template<> MPI_Datatype mpi_datatype<std::complex<double>>::type() { return MPI_CXX_DOUBLE_COMPLEX; }
// clang-format on

}
}
