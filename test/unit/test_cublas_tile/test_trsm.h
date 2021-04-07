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

#include <cuda_runtime.h>
#include <functional>
#include <sstream>
#include <tuple>
#include "gtest/gtest.h"
#include "dlaf/blas/enum_output.h"
#include "dlaf/cublas/error.h"
#include "dlaf/cublas_tile.h"
#include "dlaf/matrix/copy_tile.h"
#include "dlaf/matrix/tile.h"
#include "dlaf/memory/memory_view.h"
#include "dlaf_test/matrix/util_tile.h"
#include "dlaf_test/matrix/util_tile_blas.h"
#include "dlaf_test/util_types.h"

namespace dlaf {
namespace test {

using namespace dlaf;
using namespace dlaf::matrix;
using namespace dlaf::matrix::test;
using namespace dlaf::tile;
using namespace testing;

template <class ElementIndex, class T, class CT = const T>
void testTrsm(blas::Side side, blas::Uplo uplo, blas::Op op, blas::Diag diag, SizeType m, SizeType n,
              SizeType extra_lda, SizeType extra_ldb) {
  const TileElementSize size_a =
      side == blas::Side::Left ? TileElementSize(m, m) : TileElementSize(n, n);
  const TileElementSize size_b(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;

  std::stringstream s;
  s << "TRSM: " << side << ", " << uplo << ", " << op << ", " << diag << ", m = " << m << ", n = " << n
    << ", lda = " << lda << ", ldb = " << ldb;
  SCOPED_TRACE(s.str());

  const T alpha = TypeUtilities<T>::element(-1.2, .7);

  std::function<T(const TileElementIndex&)> el_op_a, el_b, res_b;

  if (side == blas::Side::Left)
    std::tie(el_op_a, el_b, res_b) = getLeftTriangularSystem<ElementIndex, T>(uplo, op, diag, alpha, m);
  else
    std::tie(el_op_a, el_b, res_b) = getRightTriangularSystem<ElementIndex, T>(uplo, op, diag, alpha, n);

  auto a = createTile<CT>(el_op_a, size_a, lda, op);
  auto b = createTile<T>(el_b, size_b, ldb);

  Tile<T, Device::GPU> a0d(size_a, memory::MemoryView<T, Device::GPU>(lda * size_a.cols()), lda);
  Tile<T, Device::GPU> bd(size_b, memory::MemoryView<T, Device::GPU>(ldb * size_b.cols()), ldb);

  copy(a, a0d);
  copy(b, bd);

  Tile<CT, Device::GPU> ad(std::move(a0d));

  cublasHandle_t handle;
  DLAF_CUBLAS_CALL(cublasCreate(&handle));
  cublasTrsm(handle, side, uplo, op, diag, alpha, ad, bd);
  DLAF_CUDA_CALL(cudaDeviceSynchronize());
  DLAF_CUBLAS_CALL(cublasDestroy(handle));

  copy(bd, b);

  CHECK_TILE_NEAR(res_b, b, 10 * (m + 1) * TypeUtilities<T>::error,
                  10 * (m + 1) * TypeUtilities<T>::error);
}

}
}
