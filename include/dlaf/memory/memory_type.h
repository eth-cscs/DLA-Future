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

#ifdef DLAF_WITH_GPU
#include <whip.hpp>
#endif

#include <dlaf/common/assert.h>

namespace dlaf::memory::internal {

enum class MemoryType { Host, Device, Managed, Unified };

#ifdef DLAF_WITH_CUDA
inline MemoryType get_memory_type(const void* p) {
  cudaPointerAttributes attributes{};
  cudaError_t status = cudaPointerGetAttributes(&attributes, p);
  if (status == cudaErrorInvalidValue) {
    // If Cuda returns cudaErrorInvalidValue we assume it's due
    // to Cuda not recognizing a non-Cuda allocated pointer as
    // host memory, and we assume the type is host.
    return MemoryType::Host;
  }
  else if (status != cudaSuccess) {
    throw whip::exception(status);
  }

  switch (attributes.type) {
    case cudaMemoryTypeUnregistered:
      [[fallthrough]];
    case cudaMemoryTypeHost:
      return MemoryType::Host;
    case cudaMemoryTypeDevice:
      return MemoryType::Device;
    case cudaMemoryTypeManaged:
      return MemoryType::Managed;
    default:
      return DLAF_UNREACHABLE(MemoryType);
  }
}
#elif defined(DLAF_WITH_HIP)
// Note that hipMemoryTypeManaged is not available in older versions, but it is
// already available in 5.3 so we don't do a separate check for availability.
inline MemoryType get_memory_type(const void* p) {
  hipPointerAttribute_t attributes{};
  hipError_t status = hipPointerGetAttributes(&attributes, p);
  if (status == hipErrorInvalidValue) {
    // If HIP returns hipErrorInvalidValue we assume it's due
    // to HIP not recognizing a non-HIP allocated pointer as
    // host memory, and we assume the type is host.
    return MemoryType::Host;
  }
  else if (status != hipSuccess) {
    throw whip::exception(status);
  }

  switch (attributes.type) {
#if HIP_VERSION >= 60000000
    case hipMemoryTypeUnregistered:
      [[fallthrough]];
#endif
    case hipMemoryTypeHost:
      return MemoryType::Host;
    case hipMemoryTypeArray:
      [[fallthrough]];
    case hipMemoryTypeDevice:
      return MemoryType::Device;
    case hipMemoryTypeUnified:
      return MemoryType::Unified;
    case hipMemoryTypeManaged:
      return MemoryType::Managed;
    default:
      return DLAF_UNREACHABLE(MemoryType);
  }
}
#else
inline MemoryType get_memory_type(const void*) noexcept {
  return MemoryType::Host;
}
#endif

}
