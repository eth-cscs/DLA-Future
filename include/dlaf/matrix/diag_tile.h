#pragma once

#if DLAF_WITH_CUDA
#include <cuda_runtime.h>
#include "dlaf/cuda/error.h"
#include "dlaf/util_cublas.h"
#endif

#include "dlaf/common/assert.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/types.h"

namespace dlaf {
namespace matrix {

template <class T>
void addToDiag(const Tile<T, Device::CPU>& tile, T val) {
  TileElementSize ts = tile.size();
  DLAF_ASSERT(ts.rows() == ts.cols(), "The tile must be square!");

  // Iterate over the diagonal of the tile
  for (SizeType i = 0; i < ts.rows(); ++i) {
    tile(TileElementIndex(i, i)) += val;
  }
}

#if DLAF_WITH_CUDA

namespace internal {

template <class T>
__global__ void diagHelper(T* ptr, SizeType ld, T val) {
  SizeType row = blockIdx.y * blockDim.y + threadIdx.y;
  SizeType col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row == col)
    ptr[row + col * ld] += val;
}

}

template <class T>
void addToDiag(const Tile<T, Device::GPU>& tile, T val, cudaStream_t stream) {
  TileElementSize ts = tile.size();

  dim3 dimBlock(16, 16);
  dim3 dimGrid((ts.cols() + dimBlock.x - 1) / dimBlock.x, (ts.rows() + dimBlock.y - 1) / dimBlock.y);
  internal::diagHelper<<<dimGrid, dimBlock, stream>>>(util::blasToCublasCast(tile.ptr()), tile.ld(),
                                                      util::blasToCublasCast(val));
}

#endif

}
}
