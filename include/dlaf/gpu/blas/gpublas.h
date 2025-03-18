//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//
#pragma once

/// @file
/// Provides gpublas wrappers for BLAS operations.

#ifdef DLAF_WITH_GPU
#include <complex>
#include <cstddef>
#include <utility>

#include <whip.hpp>

#include <dlaf/gpu/blas/api.h>
#include <dlaf/gpu/blas/error.h>
#include <dlaf/util_cublas.h>

#ifdef DLAF_WITH_HIP

#include <pika/async_cuda/detail/cuda_event_callback.hpp>

#include <dlaf/memory/memory_view.h>

#define DLAF_GET_ROCBLAS_WORKSPACE(f)                                                                   \
  [&]() {                                                                                               \
    std::size_t workspace_size;                                                                         \
    DLAF_GPUBLAS_CHECK_ERROR(                                                                           \
        rocblas_start_device_memory_size_query(static_cast<rocblas_handle>(handle)));                   \
    DLAF_ROCBLAS_WORKSPACE_CHECK_ERROR(rocblas_##f(handle, std::forward<Args>(args)...));               \
    DLAF_GPUBLAS_CHECK_ERROR(rocblas_stop_device_memory_size_query(static_cast<rocblas_handle>(handle), \
                                                                   &workspace_size));                   \
    return ::dlaf::memory::MemoryView<std::byte, Device::GPU>(to_int(workspace_size));                  \
  }();

namespace dlaf::tile::internal {
inline void extendROCBlasWorkspace(cublasHandle_t handle,
                                   ::dlaf::memory::MemoryView<std::byte, Device::GPU>&& workspace) {
  whip::stream_t stream;
  DLAF_GPUBLAS_CHECK_ERROR(cublasGetStream(handle, &stream));
  auto f = [workspace = std::move(workspace)](whip::error_t status) { whip::check_error(status); };
  pika::cuda::experimental::detail::add_event_callback(std::move(f), stream);
}
}

#define DLAF_DEFINE_GPUBLAS_OP(Name, Type, f)                                                           \
  template <>                                                                                           \
  struct Name<Type> {                                                                                   \
    template <typename... Args>                                                                         \
    static void call(cublasHandle_t handle, Args&&... args) {                                           \
      auto workspace = DLAF_GET_ROCBLAS_WORKSPACE(f);                                                   \
      DLAF_GPUBLAS_CHECK_ERROR(rocblas_set_workspace(static_cast<rocblas_handle>(handle), workspace(),  \
                                                     to_sizet(workspace.size())));                      \
      DLAF_GPUBLAS_CHECK_ERROR(rocblas_##f(handle, std::forward<Args>(args)...));                       \
      DLAF_GPUBLAS_CHECK_ERROR(rocblas_set_workspace(static_cast<rocblas_handle>(handle), nullptr, 0)); \
      ::dlaf::tile::internal::extendROCBlasWorkspace(handle, std::move(workspace));                     \
    }                                                                                                   \
  }

#elif defined(DLAF_WITH_CUDA)

#define DLAF_DEFINE_GPUBLAS_OP(Name, Type, f)                                \
  template <>                                                                \
  struct Name<Type> {                                                        \
    template <typename... Args>                                              \
    static void call(Args&&... args) {                                       \
      DLAF_GPUBLAS_CHECK_ERROR(cublas##f##_v2(std::forward<Args>(args)...)); \
    }                                                                        \
  }

#endif

#define DLAF_DECLARE_GPUBLAS_OP(Name) \
  template <typename T>               \
  struct Name

#ifdef DLAF_WITH_HIP
#define DLAF_MAKE_GPUBLAS_OP(Name, f)                      \
  DLAF_DECLARE_GPUBLAS_OP(Name);                           \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, s##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, d##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, c##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, z##f)

#define DLAF_MAKE_GPUBLAS_SYHE_OP(Name, f)                   \
  DLAF_DECLARE_GPUBLAS_OP(Name);                             \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, ssy##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, dsy##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, che##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, zhe##f)

#elif defined(DLAF_WITH_CUDA)
#define DLAF_MAKE_GPUBLAS_OP(Name, f)                      \
  DLAF_DECLARE_GPUBLAS_OP(Name);                           \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, S##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, D##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, C##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, Z##f)

#define DLAF_MAKE_GPUBLAS_SYHE_OP(Name, f)                   \
  DLAF_DECLARE_GPUBLAS_OP(Name);                             \
  DLAF_DEFINE_GPUBLAS_OP(Name, float, Ssy##f);               \
  DLAF_DEFINE_GPUBLAS_OP(Name, double, Dsy##f);              \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<float>, Che##f); \
  DLAF_DEFINE_GPUBLAS_OP(Name, std::complex<double>, Zhe##f)
#endif

namespace dlaf::gpublas::internal {

// Level 1
DLAF_MAKE_GPUBLAS_OP(Axpy, axpy);

// Level 2
DLAF_MAKE_GPUBLAS_OP(Gemv, gemv);

DLAF_MAKE_GPUBLAS_OP(Trmv, trmv);

// Level 3
DLAF_MAKE_GPUBLAS_OP(Gemm, gemm);

DLAF_MAKE_GPUBLAS_SYHE_OP(Hemm, mm);

DLAF_MAKE_GPUBLAS_SYHE_OP(Her2k, r2k);

DLAF_MAKE_GPUBLAS_SYHE_OP(Herk, rk);

#if defined(DLAF_WITH_CUDA)
DLAF_MAKE_GPUBLAS_OP(Trmm, trmm);
#elif defined(DLAF_WITH_HIP)

#if ROCBLAS_VERSION_MAJOR >= 3 && defined(ROCBLAS_V3)
DLAF_MAKE_GPUBLAS_OP(Trmm, trmm);
#else
DLAF_MAKE_GPUBLAS_OP(Trmm, trmm_outofplace);
#endif

#endif

DLAF_MAKE_GPUBLAS_OP(Trsm, trsm);
}
#endif
