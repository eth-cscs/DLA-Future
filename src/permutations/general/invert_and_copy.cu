//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/permutations/general/invert_and_copy.h"

namespace dlaf::permutations::internal {

struct Dist1D {
  SizeType len;
  SizeType blk;
  comm::IndexT_MPI src_rank;
  comm::IndexT_MPI rank;
  comm::IndexT_MPI nranks;
};

__device__ SizeType rankGlobalIndex(const Dist1D& dist, SizeType idx) {
  return (idx / dist.blk + dist.src_rank) % dist.nranks;
}

__device__ SizeType localIndexFromGlobalIndex(const Dist1D& dist, SizeType idx) {
  return (idx / dist.nranks) * dist.blk + idx % dist.blk;
}

constexpr unsigned invert_and_copy_kernel_sz = 256;

__global__ void invertAndCopyArrKernel(Dist1D dist, const SizeType* in_ptr, SizeType* out_ptr) {
  SizeType idx = blockIdx.x * invert_and_copy_kernel_sz + threadIdx.x;
  if (idx >= dist.len)
    return;

  const SizeType in_gl = in_ptr[idx];
  if (rankGlobalIndex(in_gl) == dist.rank) {
    SizeType out_loc = localIndexFromGlobalIndex(idx_gl);
    out_ptr[out_loc] = idx;
  }
}

template <Coord C>
void invertAndCopyArr(const matrix::Distribution& dist_2d, const SizeType* in_ptr, SizeType* out_ptr,
                      whip::stream_t stream) {
  Dist1D dist{
      dist_2d.size().get<C>(),             // len
      dist_2d.blockSize().get<C>(),        // blk
      dist_2d.sourceRankIndex().get<C>(),  // src_rank
      dist_2d.rankIndex().get<C>(),        // rank
      dist_2d.commGridSize().get<C>()      // nranks
  };

  dim3 nr_threads(invert_and_copy_kernel_sz);
  dim3 nr_blocks(util::ceilDiv(to_uint(dist.len), invert_and_copy_kernel_sz));
  invertAndCopyArrKernel<<<nr_blocks, nr_threads, 0, stream>>>(dist, in_ptr, out_ptr);
}

DLAF_GPU_INVERT_AND_COPY_ETI(, Coord::Row);
DLAF_GPU_INVERT_AND_COPY_ETI(, Coord::Col);

}
