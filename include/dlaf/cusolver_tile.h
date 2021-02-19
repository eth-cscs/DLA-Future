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

#ifdef DLAF_WITH_CUDA

#include <cusolverDn.h>
#include <blas.hh>

#include "dlaf/cusolver/error.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"
#include "dlaf/util_blas.h"
#include "dlaf/util_cublas.h"

namespace dlaf {
namespace tile {
namespace internal {
template <typename T>
// TODO: This uses the "legacy" API. What does legacy mean in this case? Will
// it be removed?
struct CublasPotrf;

template <>
struct CublasPotrf<float> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnSpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnSpotrf_bufferSize(std::forward<Args>(args)...));
  }
};

template <>
struct CublasPotrf<double> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnDpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnDpotrf_bufferSize(std::forward<Args>(args)...));
  }
};

template <>
struct CublasPotrf<std::complex<float>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnCpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnCpotrf_bufferSize(std::forward<Args>(args)...));
  }
};

template <>
struct CublasPotrf<std::complex<double>> {
  template <typename... Args>
  static void call(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnZpotrf(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static void callBufferSize(Args&&... args) {
    DLAF_CUSOLVER_CALL(cusolverDnZpotrf_bufferSize(std::forward<Args>(args)...));
  }
};
}

namespace internal {
template <class T>
class CusolverPotrfInfo {
  // TODO: Combine buffers? Reuse upper/lower part of matrix? Workspace
  // only size 1?
  memory::MemoryView<T, Device::GPU> workspace_;
  memory::MemoryView<int, Device::GPU> info_;

public:
  CusolverPotrfInfo(int workspace_size) : workspace_(workspace_size), info_() {}

  T* workspace() {
    return workspace_();
  }
  int* info() {
    return info_();
  }
};
}

template <class T>
auto potrf(cusolverDnHandle_t handle, const blas::Uplo uplo, const matrix::Tile<T, Device::GPU>& a) {
  DLAF_ASSERT(square_size(a), a);
  const int n = a.size().rows();

  int workspace_size;
  internal::CublasPotrf<T>::callBufferSize(handle, util::blasToCublas(uplo), n,
                                           util::blasToCublasCast(a.ptr()), a.ld(), &workspace_size);
  internal::CusolverPotrfInfo<T> info{workspace_size};
  internal::CublasPotrf<T>::call(handle, util::blasToCublas(uplo), n, util::blasToCublasCast(a.ptr()),
                                 a.ld(), util::blasToCublasCast(info.workspace()), workspace_size,
                                 info.info());

  // TODO: When and where to check the result?
  return info;
}
}
}

#endif
