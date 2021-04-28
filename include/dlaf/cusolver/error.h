//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2021, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// @file

#include <exception>
#include <iostream>

#ifdef DLAF_WITH_CUDA
#include <cusolverDn.h>
#endif

#include "dlaf/common/source_location.h"

namespace dlaf {
namespace internal {

#ifdef DLAF_WITH_CUDA

/// CUSOLVER equivalent to `cudaGetErrorString()`
/// Reference: https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverSPstatus
inline std::string cusolverGetErrorString(cusolverStatus_t st) {
  // clang-format off
  switch (st) {
    case CUSOLVER_STATUS_SUCCESS:                                 return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:                         return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:                            return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:                           return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:                           return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:                        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:                          return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:               return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_MAPPING_ERROR:                           return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_NOT_SUPPORTED:                           return "CUSOLVER_STATUS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_ZERO_PIVOT:                              return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:                         return "CUSOLVER_STATUS_INVALID_LICENSE";
    case CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED:              return "CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID:                      return "CUSOLVER_STATUS_IRS_PARAMS_INVALID";
    case CUSOLVER_STATUS_IRS_INTERNAL_ERROR:                      return "CUSOLVER_STATUS_IRS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_IRS_NOT_SUPPORTED:                       return "CUSOLVER_STATUS_IRS_NOT_SUPPORTED";
    case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:                        return "CUSOLVER_STATUS_IRS_OUT_OF_RANGE";
    case CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES: return "CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES";
    case CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED:               return "CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC:                 return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC";
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE:               return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE";
    case CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER:              return "CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER";
    case CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED:                 return "CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED";
    case CUSOLVER_STATUS_IRS_MATRIX_SINGULAR:                     return "CUSOLVER_STATUS_IRS_MATRIX_SINGULAR";
    case CUSOLVER_STATUS_INVALID_WORKSPACE:                       return "CUSOLVER_STATUS_INVALID_WORKSPACE";
  }
  // clang-format on
  return "UNKNOWN";
}

inline void cusolverCall(cusolverStatus_t st, const dlaf::common::internal::source_location& info) noexcept {
  if (st != CUSOLVER_STATUS_SUCCESS) {
    std::cout << "[CUSOLVER ERROR] " << info << std::endl << cusolverGetErrorString(st) << std::endl;
    std::terminate();
  }
}

#define DLAF_CUSOLVER_CALL(cusolver_f) dlaf::internal::cusolverCall((cusolver_f), SOURCE_LOCATION())

#endif

}
}
