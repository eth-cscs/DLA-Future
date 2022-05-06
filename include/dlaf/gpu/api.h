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

#if defined(DLAF_WITH_HIP)

#include <hip/hip_runtime.h>

#define cublasGetStream                  hipblasGetStream
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaDeviceSynchronize            hipDeviceSynchronize
#define cudaError_t                      hipError_t
#define cudaGetErrorString               hipGetErrorString
#define cudaMemcpy2D                     hipMemcpy2D
#define cudaMemcpy2DAsync                hipMemcpy2DAsync
#define cudaMemcpyDefault                hipMemcpyDefault
#define cudaMemcpyDeviceToDevice         hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost           hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice           hipMemcpyHostToDevice
#define cudaMemset2DAsync                hipMemset2DAsync
#define cudaPointerAttributes            hipPointerAttribute_t
#define cudaSetDevice                    hipSetDevice
#define cudaStreamCreateWithPriority     hipStreamCreateWithPriority
#define cudaStreamDestroy                hipStreamDestroy
#define cudaStreamNonBlocking            hipStreamNonBlocking
#define cudaStream_t                     hipStream_t
#define cudaSuccess                      hipSuccess

#elif defined(DLAF_WITH_CUDA)

#include <cuda_runtime.h>

#endif
