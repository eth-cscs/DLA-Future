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

/// @file

#include <cstdio>
#include <iostream>
#include <memory>

#include <mpi.h>

namespace dlaf {
namespace internal {

/// MPI provides two predefined error handlers
///
/// - `MPI_ERRORS_ARE_FATAL` (the default ?): causes MPI to abort
/// - `MPI_ERRORS_RETURN`: causes MPI to return an error values instead of aborting
///
/// This function is only relevant when the error handler is `MPI_ERRORS_RETURN`. To set the error
/// handler use `MPI_Errhandler_set`.
///
inline void mpi_call(int err, char const* file, int line) {
  if (err != MPI_SUCCESS) {
    std::unique_ptr<char[]> err_buff(new char[MPI_MAX_ERROR_STRING]);
    int err_str_len;
    MPI_Error_string(err, err_buff.get(), &err_str_len);
    std::printf("[MPI ERROR] %s:%d: '%s'\n", file, line, err_buff.get());
    // MPI_Abort(MPI_COMM_WORLD, err);
    // std::abort();
  }
}

#define DLAF_MPI_CALL(mpi_f) dlaf::internal::mpi_call((mpi_f), __FILE__, __LINE__)

}
}
