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
struct mpi_datatype;

/// helper for mapping also custom types
template <typename T>
struct mpi_datatype<const T> : public mpi_datatype<T> {};

template <>
struct mpi_datatype<char> {
  static constexpr MPI_Datatype type = MPI_CHAR;
};

template <>
struct mpi_datatype<short> {
  static constexpr MPI_Datatype type = MPI_SHORT;
};

template <>
struct mpi_datatype<int> {
  static constexpr MPI_Datatype type = MPI_INT;
};

template <>
struct mpi_datatype<long> {
  static constexpr MPI_Datatype type = MPI_LONG;
};

template <>
struct mpi_datatype<long long> {
  static constexpr MPI_Datatype type = MPI_LONG_LONG;
};

template <>
struct mpi_datatype<unsigned char> {
  static constexpr MPI_Datatype type = MPI_UNSIGNED_CHAR;
};

template <>
struct mpi_datatype<unsigned short> {
  static constexpr MPI_Datatype type = MPI_UNSIGNED_SHORT;
};

template <>
struct mpi_datatype<unsigned int> {
  static constexpr MPI_Datatype type = MPI_UNSIGNED;
};

template <>
struct mpi_datatype<unsigned long> {
  static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG;
};

template <>
struct mpi_datatype<unsigned long long> {
  static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG_LONG;
};

template <>
struct mpi_datatype<float> {
  static constexpr MPI_Datatype type = MPI_FLOAT;
};

template <>
struct mpi_datatype<double> {
  static constexpr MPI_Datatype type = MPI_DOUBLE;
};

template <>
struct mpi_datatype<bool> {
  static constexpr MPI_Datatype type = MPI_CXX_BOOL;
};

template <>
struct mpi_datatype<std::complex<float>> {
  static constexpr MPI_Datatype type = MPI_CXX_FLOAT_COMPLEX;
};

template <>
struct mpi_datatype<std::complex<double>> {
  static constexpr MPI_Datatype type = MPI_CXX_DOUBLE_COMPLEX;
};

}
}
