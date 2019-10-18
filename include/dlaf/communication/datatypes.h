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

// TODO MPI_INT and MPI_INT32_T are different

template <typename T>
struct mpi_datatype;

/// get mapping also for const types
template <typename T>
struct mpi_datatype<const T> : public mpi_datatype<T> {};

template <>
struct mpi_datatype<int8_t> {
  static constexpr MPI_Datatype type = MPI_INT8_T;
};

template <>
struct mpi_datatype<int16_t> {
  static constexpr MPI_Datatype type = MPI_INT16_T;
};

template <>
struct mpi_datatype<int32_t> {
  static constexpr MPI_Datatype type = MPI_INT32_T;
};

template <>
struct mpi_datatype<int64_t> {
  static constexpr MPI_Datatype type = MPI_INT64_T;
};

template <>
struct mpi_datatype<uint8_t> {
  static constexpr MPI_Datatype type = MPI_UINT8_T;
};

template <>
struct mpi_datatype<uint16_t> {
  static constexpr MPI_Datatype type = MPI_UINT16_T;
};

template <>
struct mpi_datatype<uint32_t> {
  static constexpr MPI_Datatype type = MPI_UINT32_T;
};

template <>
struct mpi_datatype<uint64_t> {
  static constexpr MPI_Datatype type = MPI_UINT64_T;
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
