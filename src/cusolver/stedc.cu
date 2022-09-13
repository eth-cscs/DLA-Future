//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/lapack/tile.h"
#include "dlaf/util_cuda.h"

namespace dlaf::tile::internal {

constexpr unsigned tridiag_kernel_sz = 32;

template <class T>
__global__ void expandTridiagonalToLowerTriangular(SizeType n, const T* diag, const T* offdiag,
                                                   SizeType ld_evecs, T* evecs) {
  const SizeType i = blockIdx.x * tridiag_kernel_sz + threadIdx.x;
  const SizeType j = blockIdx.y * tridiag_kernel_sz + threadIdx.y;

  // only the lower triangular part is set
  if (i >= n || j >= n || j > i)
    return;

  const SizeType idx = i + j * ld_evecs;

  if (i == j) {
    evecs[idx] = diag[i];
  }
  else if (i == j + 1) {
    evecs[idx] = offdiag[j];
  }
  else {
    evecs[idx] = T(0);
  }
}

__global__ void assertSyevdInfo(int* info) {
  if (*info != 0) {
    printf("Error SYEVD: info != 0 (%d)\n", *info);
    __trap();
  }
}

// `evals` [in / out] on entry holds the diagonal of the tridiagonal matrix, on exit holds the
// eigenvalues in the first column
//
// `evecs` [out] first holds the tridiagonal tile converted to an expanded form - lower triangular
// matrix, on exit it holds the eigenvectors tridiag holds the eigenvectors on exit
//
// Example of using cusolverDnXsyevd() is provided here:
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xsyevd/cusolver_Xsyevd_example.cu
template <class T>
void stedc(cusolverDnHandle_t handle, const Tile<T, Device::GPU>& tridiag,
           const Tile<T, Device::GPU>& evecs) {
  SizeType n = tridiag.size().rows();
  T* evals_ptr = tridiag.ptr(TileElementIndex(0, 0));
  const T* offdiag_ptr = tridiag.ptr(TileElementIndex(0, 1));
  SizeType ld_evecs = evecs.ld();
  T* evecs_ptr = evecs.ptr();

  // Expand from compact tridiagonal form into lower triangular form
  cudaStream_t stream;
  DLAF_GPULAPACK_CHECK_ERROR(cusolverDnGetStream(handle, &stream));
  const unsigned un = to_uint(n);
  dim3 nr_threads(tridiag_kernel_sz, tridiag_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(un, tridiag_kernel_sz), util::ceilDiv(un, tridiag_kernel_sz));
  expandTridiagonalToLowerTriangular<<<nr_blocks, nr_threads, 0, stream>>>(n, evals_ptr, offdiag_ptr,
                                                                           ld_evecs, evecs_ptr);

  // Determine additional memory needed and solve the symmetric eigenvalue problem
  cusolverDnParams_t params;
  DLAF_GPULAPACK_CHECK_ERROR(cusolverDnCreateParams(&params));
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;  // compute both eigenvalues and eigenvectors
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;     // the symmetric matrix is stored in the lower part
  cudaDataType dtype = (std::is_same<T, float>::value) ? CUDA_R_32F : CUDA_R_64F;

  size_t workspaceInBytesOnDevice;
  size_t workspaceInBytesOnHost;
  DLAF_GPULAPACK_CHECK_ERROR(
      cusolverDnXsyevd_bufferSize(handle, params, jobz, uplo, n, dtype, evecs_ptr, ld_evecs, dtype,
                                  evals_ptr, dtype, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  void* bufferOnDevice = memory::internal::getUmpireDeviceAllocator().allocate(workspaceInBytesOnDevice);
  void* bufferOnHost = memory::internal::getUmpireHostAllocator().allocate(workspaceInBytesOnHost);

  // Note: `info` has to be stored on device!
  memory::MemoryView<int, Device::GPU> info(1);
  DLAF_GPULAPACK_CHECK_ERROR(cusolverDnXsyevd(handle, params, jobz, uplo, n, dtype, evecs_ptr, ld_evecs,
                                              dtype, evals_ptr, dtype, bufferOnDevice,
                                              workspaceInBytesOnDevice, bufferOnHost,
                                              workspaceInBytesOnHost, info()));
  assertSyevdInfo<<<1, 1, 0, stream>>>(info());

  auto extend_info = [info = std::move(info), bufferOnDevice, bufferOnHost, params](cudaError_t status) {
    DLAF_GPU_CHECK_ERROR(status);
    memory::internal::getUmpireDeviceAllocator().deallocate(bufferOnDevice);
    memory::internal::getUmpireHostAllocator().deallocate(bufferOnHost);
    DLAF_GPULAPACK_CHECK_ERROR(cusolverDnDestroyParams(params));
  };
  pika::cuda::experimental::detail::add_event_callback(std::move(extend_info), stream);
}

DLAF_GPU_STEDC_ETI(, float);
DLAF_GPU_STEDC_ETI(, double);

}
