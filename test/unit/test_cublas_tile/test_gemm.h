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
#include <sstream>
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

template <class T, class CT = const T>
void testGemm(blas::Op op_a, blas::Op op_b, SizeType m, SizeType n, SizeType k, SizeType extra_lda,
              SizeType extra_ldb, SizeType extra_ldc) {
  const TileElementSize size_a =
      (op_a == blas::Op::NoTrans) ? TileElementSize(m, k) : TileElementSize(k, m);
  const TileElementSize size_b =
      (op_b == blas::Op::NoTrans) ? TileElementSize(k, n) : TileElementSize(n, k);
  const TileElementSize size_c(m, n);

  const SizeType lda = std::max<SizeType>(1, size_a.rows()) + extra_lda;
  const SizeType ldb = std::max<SizeType>(1, size_b.rows()) + extra_ldb;
  const SizeType ldc = std::max<SizeType>(1, size_c.rows()) + extra_ldc;

  std::stringstream s;
  s << "GEMM: " << op_a << ", " << op_a;
  s << ", m = " << m << ", n = " << n << ", k = " << k;
  s << ", lda = " << lda << ", ldb = " << ldb << ", ldc = " << ldc;
  SCOPED_TRACE(s.str());

  // Note: The tile elements are chosen such that:
  // - op_a(a)_ik = .9 * (i+1) / (k+.5) * exp(I*(2*i-k)),
  // - op_b(b)_kj = .8 * (k+.5) / (j+2) * exp(I*(k+j)),
  // - c_ij = 1.2 * i / (j+1) * exp(I*(-i+j)),
  // where I = 0 for real types or I is the complex unit for complex types.
  // Therefore the result should be:
  // res_ij = beta * c_ij + Sum_k(alpha * op_a(a)_ik * op_b(b)_kj)
  //        = beta * c_ij + gamma * (i+1) / (j+2) * exp(I*(2*i+j)),
  // where gamma = .72 * k * alpha.
  auto el_op_a = [](const TileElementIndex& index) {
    double i = index.row();
    double k = index.col();
    return TypeUtilities<T>::polar(.9 * (i + 1) / (k + .5), 2 * i - k);
  };
  auto el_op_b = [](const TileElementIndex& index) {
    double k = index.row();
    double j = index.col();
    return TypeUtilities<T>::polar(.8 * (k + .5) / (j + 2), k + j);
  };
  auto el_c = [](const TileElementIndex& index) {
    double i = index.row();
    double j = index.col();
    return TypeUtilities<T>::polar(1.2 * i / (j + 1), -i + j);
  };

  T alpha = TypeUtilities<T>::element(-1.2, .7);
  T beta = TypeUtilities<T>::element(1.1, .4);

  T gamma = TypeUtilities<T>::element(.72 * k, 0) * alpha;
  auto res_c = [beta, el_c, gamma](const TileElementIndex& index) {
    double i = index.row();
    double j = index.col();
    return beta * el_c(index) + gamma * TypeUtilities<T>::polar((i + 1) / (j + 2), 2 * i + j);
  };

  auto a = createTile<CT>(el_op_a, size_a, lda, op_a);
  auto b = createTile<CT>(el_op_b, size_b, ldb, op_b);
  auto c = createTile<T>(el_c, size_c, ldc);

  Tile<T, Device::GPU> a0d(size_a, memory::MemoryView<T, Device::GPU>(lda * size_a.cols()), lda);
  Tile<T, Device::GPU> b0d(size_b, memory::MemoryView<T, Device::GPU>(ldb * size_b.cols()), ldb);
  Tile<T, Device::GPU> cd(size_c, memory::MemoryView<T, Device::GPU>(ldc * size_c.cols()), ldc);

  copy(a, a0d);
  copy(b, b0d);
  copy(c, cd);

  Tile<CT, Device::GPU> ad(std::move(a0d));
  Tile<CT, Device::GPU> bd(std::move(b0d));

  cublasHandle_t handle;
  DLAF_CUBLAS_CALL(cublasCreate(&handle));
  cublasGemm(handle, op_a, op_b, alpha, ad, bd, beta, cd);
  DLAF_CUDA_CALL(cudaDeviceSynchronize());
  DLAF_CUBLAS_CALL(cublasDestroy(handle));

  copy(cd, c);

  // Check result against analytical result.
  CHECK_TILE_NEAR(res_c, c, 2 * (k + 1) * TypeUtilities<T>::error,
                  2 * (k + 1) * TypeUtilities<T>::error);
}
}
}
