//
// Distributed Linear Algebra with Future (DLAF)
//
// Copyright (c) 2018-2022, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "dlaf/lapack/gpu/lacpy.h"

#include <gtest/gtest.h>
#include <whip.hpp>

#include "dlaf/blas/enum_output.h"
#include "dlaf/gpu/blas/error.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/util_types.h"

using namespace dlaf;
using namespace dlaf::test;
using dlaf::matrix::test::createTile;

template <class T>
struct LacpyTestGPU : public ::testing::Test {};

TYPED_TEST_SUITE(LacpyTestGPU, MatrixElementTypes);

std::vector<std::tuple<SizeType, SizeType, SizeType, SizeType>> configs{
    // m, n, lda, ldb
    {0, 0, 1, 1},        {0, 4, 4, 4},         {4, 0, 5, 5},                       // Empty
    {6, 6, 6, 6},        {5, 3, 5, 6},         {7, 3, 10, 13},      {3, 7, 8, 8},  // Very Small
    {127, 35, 128, 200}, {128, 128, 128, 128}, {96, 127, 256, 256},                // A bit larger
};

TYPED_TEST(LacpyTestGPU, CorrectnessLocal) {
  using T = TypeParam;

  using blas::Uplo;

  whip::stream_t stream;
  whip::stream_create(&stream);
  cublasHandle_t handle;
  DLAF_GPUBLAS_CHECK_ERROR(cublasCreate(&handle));
  DLAF_GPUBLAS_CHECK_ERROR(cublasSetStream(handle, stream));

  auto zero = [](const TileElementIndex&) { return T(0); };

  auto el = [](const TileElementIndex& ij) {
    if (ij.row() == ij.col())
      return T(ij.row());
    else
      return T(ij.row() - ij.col());
  };

  for (const auto& [m, n, lda, ldb] : configs) {
    for (const auto uplo : {Uplo::Lower, Uplo::Upper, Uplo::General}) {
      // Reference
      auto tile_input = createTile<const T, Device::CPU>(el, {m, n}, lda);
      auto tile_result = createTile<T, Device::CPU>(zero, {m, n}, ldb);

      lapack::lacpy(uplo, m, n, tile_input.ptr(), tile_input.ld(), tile_result.ptr(), tile_result.ld());

      // Test
      auto tile_src = createTile<const T, Device::GPU>(el, {m, n}, lda);
      auto tile_dst = createTile<T, Device::GPU>(zero, {m, n}, ldb);

      gpulapack::lacpy(uplo, m, n, tile_src.ptr(), tile_src.ld(), tile_dst.ptr(), tile_dst.ld(), stream);
      whip::stream_synchronize(stream);

      // Verify
      SCOPED_TRACE(::testing::Message() << "Comparison test m=" << m << " n=" << n << " lda=" << lda
                                        << " ldb=" << ldb << " uplo=" << uplo);

      const auto error = TypeUtilities<T>::error;
      CHECK_TILE_NEAR(tile_result, tile_dst, error, error);
    }
  }

  DLAF_GPUBLAS_CHECK_ERROR(cublasDestroy(handle));
  whip::stream_destroy(stream);
}
