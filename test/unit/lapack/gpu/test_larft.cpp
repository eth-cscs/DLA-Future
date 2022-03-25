//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/lapack/gpu/larft.h"

#include <gtest/gtest.h>

#include "dlaf/common/vector.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf/util_matrix.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using dlaf::matrix::util::internal::getter_random;
using dlaf::matrix::test::createTile;

template <class T>
struct LarftTestGPU : public ::testing::Test {};

TYPED_TEST_SUITE(LarftTestGPU, MatrixElementTypes);

std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> configs{
    // n, k, mb, nb, v_start
    {0, 0, 1, 1},       {0, 4, 4, 4},         {4, 0, 5, 5},                       // Empty
    {6, 6, 6, 6},       {5, 3, 5, 3},         {7, 3, 10, 13},      {3, 7, 8, 8},  // Very Small
    {127, 35, 128, 36}, {128, 128, 128, 128}, {96, 127, 256, 256},                // A bit larger
};

TYPED_TEST(LarftTestGPU, CorrectnessLocal) {
  // Note:
  // There are two main differences of GPU larft compared to LAPACK implementation:
  // 1) the first element of the reflectors has to be set to 1. (implicit considered as one in LAPACK)
  // 2) The lower part of the T factor is 0. (untouched in LAPACK)
  using T = TypeParam;
  using CT = const T;

  cudaStream_t stream;
  DLAF_CUDA_CALL(cudaStreamCreate(&stream));
  cublasHandle_t handle;
  DLAF_CUBLAS_CALL(cublasCreate(&handle));
  DLAF_CUBLAS_CALL(cublasSetStream(handle, stream));
  for (const auto& [n, k, ldv, ldt] : configs) {
    getter_random<T> random_value(25698 + k * n);
    auto el = [&random_value](const TileElementIndex& index) {
      if (index.row() == index.col())
        return T{1};
      return random_value();
    };
    auto v_h = createTile<CT, Device::CPU>(el, {n, k}, ldv);
    auto v_d = createTile<T, Device::GPU>({n, k}, ldv);

    matrix::internal::copy_o(v_h, v_d);

    common::internal::vector<T> taus(k);
    for (SizeType j = 0; j < k && j < n; ++j) {
      const auto norm = blas::nrm2(n - j, v_h.ptr({j, j}), 1);
      taus[j] = 2 / (norm * norm);
    }

    // CPU Tile is set to zero, to guarantee that the element of the lower part are 0.
    auto zero = [](const TileElementIndex&) { return T{0}; };
    auto t_h = createTile<T, Device::CPU>(zero, {k, k}, ldt);
    auto t_d = createTile<T, Device::GPU>({k, k}, ldt);

    lapack::larft(lapack::Direction::Forward, lapack::StoreV::Columnwise, n, k, v_h.ptr(), v_h.ld(),
                  taus.data(), t_h.ptr(), t_h.ld());

    gpulapack::larft(handle, n, k, v_d.ptr(), v_d.ld(), taus.data(), t_d.ptr(), t_d.ld());
    DLAF_CUDA_CALL(cudaStreamSynchronize(stream));

    SCOPED_TRACE(::testing::Message()
                 << "Comparison test n=" << n << " k=" << k << " ldv=" << ldv << " ldt=" << ldt);

    const auto error = (k + 1) * TypeUtilities<T>::error;
    CHECK_TILE_NEAR(t_h, t_d, error, error);
  }
  DLAF_CUBLAS_CALL(cublasDestroy(handle));
  DLAF_CUDA_CALL(cudaStreamDestroy(stream));
}
